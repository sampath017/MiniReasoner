"""Reusable SimpleStories, Phase 4 arithmetic, KV-cache, and interactive chat benchmarks."""

import math
from pathlib import Path
import re
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from ..data.arithmetic import (
    ArithmeticExample,
    canonical_problem,
    collate_sft_batch,
    encode_sft_split,
    load_or_restore_arithmetic_dataset,
    normalize_answer_format,
    prepare_arithmetic_dataset,
    render_examples,
    score_arithmetic_completion,
    score_strict_scratchpad_rollout,
    select_stratified_eval_problems,
)
from ..training.common import loss_for
from .generate import generate_batch_greedy, generate_batch_greedy_with_meta, generate_text
from .tokenizer import EOS


DEFAULT_STORY_PROMPTS = (
    "Once upon a time, there was a little",
    "One day, Lily found a secret",
    "Mia had a little dog named",
    "Jack and his best friend went to the",
    "In the middle of the dark forest,",
)

PRE_SFT_BASELINE_PROMPTS = (
    {"op": "add_1d", "a": 3, "b": 5, "answer": 8, "prompt": "What is 3 + 5?\nAnswer:"},
    {"op": "add_1d", "a": 7, "b": 8, "answer": 15, "prompt": "What is 7 + 8?\nAnswer:"},
    {"op": "add_2d", "a": 12, "b": 15, "answer": 27, "prompt": "What is 12 + 15?\nAnswer:"},
    {"op": "add_2d_carry", "a": 47, "b": 8, "answer": 55, "prompt": "What is 47 + 8?\nAnswer:"},
    {"op": "add_2d_carry", "a": 38, "b": 24, "answer": 62, "prompt": "What is 38 + 24?\nAnswer:"},
    {"op": "sub_1d", "a": 9, "b": 4, "answer": 5, "prompt": "What is 9 - 4?\nAnswer:"},
    {"op": "sub_2d", "a": 25, "b": 10, "answer": 15, "prompt": "What is 25 - 10?\nAnswer:"},
    {"op": "sub_2d_borrow", "a": 52, "b": 17, "answer": 35, "prompt": "What is 52 - 17?\nAnswer:"},
    {"op": "sub_2d_borrow", "a": 43, "b": 18, "answer": 25, "prompt": "What is 43 - 18?\nAnswer:"},
    {"op": "add_zero", "a": 14, "b": 0, "answer": 14, "prompt": "What is 14 + 0?\nAnswer:"},
)


@torch.inference_mode()
def generate_timed(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 150,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    use_cache: bool = True,
    device: torch.device | None = None,
) -> dict:
    """Run timed text completion with optional KV caching and return text + throughput stats."""
    active_device = device or next(model.parameters()).device
    start = time.perf_counter()
    completion = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        device=active_device,
        temperature=temperature,
        top_k=0 if temperature <= 0.0 else top_k,
        top_p=1.0 if temperature <= 0.0 else top_p,
        repetition_penalty=repetition_penalty,
        use_cache=use_cache,
    )
    elapsed = time.perf_counter() - start
    n_gen = len(tokenizer.encode(completion, add_special_tokens=False).ids) if completion else 0
    return {
        "prompt": prompt,
        "completion": completion,
        "full_text": prompt + completion,
        "tokens_generated": n_gen,
        "elapsed_seconds": elapsed,
        "tokens_per_second": n_gen / elapsed if elapsed > 0 else 0.0,
        "device": active_device.type,
    }


def benchmark_throughput(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    device: torch.device,
    *,
    prompt: str = "Once upon a time, there was a little",
    token_lengths: tuple[int, ...] = (30, 60, 100),
    use_cache: bool = True,
) -> list[dict]:
    """Print and return generation throughput across target completion lengths."""
    rows = []
    mode_tag = "KV-Cache ON" if use_cache else "KV-Cache OFF"
    print("=" * 72)
    print(f"THROUGHPUT BENCHMARK ({device.type.upper()} | {mode_tag})")
    print("=" * 72)
    print(f"{'Target Tokens':<15}{'Tokens Generated':<20}{'Elapsed Time (s)':<20}{'Throughput (tok/s)'}")
    print("-" * 72)
    for target in token_lengths:
        res = generate_timed(model, tokenizer, prompt, max_new_tokens=target, temperature=0.0, use_cache=use_cache, device=device)
        print(f"{target:<15}{res['tokens_generated']:<20}{res['elapsed_seconds']:<20.2f}{res['tokens_per_second']:.2f}")
        rows.append({"target": target, **res})
    print("=" * 72)
    return rows


def evaluate_simplestories_test(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    model_config,
    device: torch.device,
    *,
    n_stories: int | None = None,
    n_cloze: int | None = None,
    sample_prompts: tuple[str, ...] = DEFAULT_STORY_PROMPTS,
) -> dict:
    """Evaluate held-out SimpleStories test PPL, BPB, RoPE position scaling, Cloze accuracy, and completions."""
    target_stories = n_stories if n_stories is not None else (250 if device.type == "cuda" else 64)
    target_cloze = n_cloze if n_cloze is not None else (100 if device.type == "cuda" else 40)

    test_stories: list[str] = []
    skip_reason = None
    try:
        from datasets import load_dataset
        ds = load_dataset("SimpleStories/SimpleStories", split="test", streaming=True)
        for row in ds:
            text = (row.get("story") or "").strip()
            if text:
                test_stories.append(text)
            if len(test_stories) >= max(target_stories, target_cloze * 3):
                break
    except Exception as exc:
        skip_reason = f"{type(exc).__name__}: {exc}"

    eval_stories = test_stories[:target_stories]
    if not eval_stories:
        print("=" * 70)
        print(f"SKIPPING HELD-OUT TEST EVALUATION ({skip_reason or 'no test stories found'})")
        print("=" * 70)
        return {"skipped": True, "reason": skip_reason}

    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    buckets = {"1-256": [0.0, 0], "257-512": [0.0, 0], "513-1024": [0.0, 0]}

    with torch.inference_mode():
        for story in eval_stories:
            ids = tokenizer.encode(story, add_special_tokens=False).ids[: model_config.context_length]
            if len(ids) < 2:
                continue
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor(ids[1:], dtype=torch.long, device=device)
            logits = model(x)[0].float()
            per_tok = torch.nn.functional.cross_entropy(logits, y, reduction="none")
            n_pred = len(ids) - 1
            total_nll += float(per_tok.sum().item())
            total_tokens += n_pred
            total_bytes += max(1, len(tokenizer.decode(ids[1:]).encode("utf-8")))
            for label, lo, hi in (("1-256", 0, 256), ("257-512", 256, 512), ("513-1024", 512, n_pred)):
                if n_pred > lo:
                    span = per_tok[lo : min(hi, n_pred)]
                    buckets[label][0] += float(span.sum().item())
                    buckets[label][1] += span.numel()

    mean_ce = total_nll / max(1, total_tokens)
    mean_ppl = math.exp(mean_ce)
    bpb = total_nll / max(1e-12, math.log(2.0) * total_bytes)

    print("=" * 70)
    print("HELD-OUT TEST EVALUATION | SimpleStories/SimpleStories (test split)")
    print("=" * 70)
    print(f"Stories Scored:          {len(eval_stories):,}")
    print(f"Predicted Tokens:        {total_tokens:,} ({total_bytes:,} UTF-8 bytes | {total_bytes / max(1, total_tokens):.2f} bytes/tok)")
    print(f"Test Cross-Entropy Loss: {mean_ce:.4f}")
    print(f"Test Perplexity (PPL):   {mean_ppl:.2f}")
    print(f"Test Bits-Per-Byte(BPB): {bpb:.4f}")
    print("-" * 70)
    print("RoPE Context-Length Position Scaling:")
    for label, (nll_sum, tok_cnt) in buckets.items():
        if tok_cnt > 0:
            ce_b = nll_sum / tok_cnt
            print(f"  Tokens {label:<10} | count={tok_cnt:<7,} | Loss={ce_b:.4f} | PPL={math.exp(ce_b):.2f}")
    print("=" * 70)

    # 4-choice Story Cloze
    rng = np.random.default_rng(42)
    split_sents = [[s.strip() for s in re.split(r"(?<=[.!?])\s+", st.strip()) if s.strip()] for st in test_stories]
    eligible = [s for s in split_sents if len(s) >= 4]
    cloze_acc = None
    if len(eligible) >= 4:
        distractors = [s[min(3, len(s) - 1)] for s in eligible]
        cloze_items = []
        for idx, sents in enumerate(eligible[:target_cloze]):
            t_pos = min(3, len(sents) - 1)
            cand = [i for i in range(len(distractors)) if i != idx]
            chosen = rng.choice(cand, size=3, replace=False)
            cloze_items.append((" ".join(sents[:t_pos]), [sents[t_pos]] + [distractors[int(j)] for j in chosen], 0))

        pad_id = tokenizer.token_to_id(EOS) or 0
        correct = 0
        with torch.inference_mode():
            for q_idx, (ctx, opts, gold_idx) in enumerate(cloze_items):
                ctx_ids = tokenizer.encode(ctx, add_special_tokens=False).ids
                opt_seqs, opt_spans = [], []
                for opt in opts:
                    opt_ids = tokenizer.encode(" " + opt, add_special_tokens=False).ids
                    max_ctx = max(1, model_config.context_length - len(opt_ids))
                    seq = (ctx_ids[-max_ctx:] + opt_ids)[-model_config.context_length :]
                    opt_seqs.append(seq)
                    opt_spans.append((len(seq) - len(opt_ids), len(seq)))
                max_len = max(len(s) for s in opt_seqs)
                batch_x = torch.full((4, max_len), pad_id, dtype=torch.long, device=device)
                for row_i, seq in enumerate(opt_seqs):
                    batch_x[row_i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                logits = model(batch_x).float()
                scores = [
                    float(torch.nn.functional.cross_entropy(logits[r, s - 1 : e - 1], batch_x[r, s:e]).item())
                    for r, (s, e) in enumerate(opt_spans)
                ]
                correct += int(int(np.argmin(scores)) == gold_idx)
        cloze_acc = correct / max(1, len(cloze_items)) * 100.0
        print(f"\n4-Choice Story Cloze Accuracy: {correct}/{len(cloze_items)} ({cloze_acc:.1f}%) [Random baseline: 25.0%]")

    print("\n" + "=" * 70)
    print(f"STORY COMPLETION SANITY CHECK (Greedy on {device.type.upper()})")
    print("=" * 70)
    for idx, sp in enumerate(sample_prompts, start=1):
        res = generate_timed(model, tokenizer, sp, max_new_tokens=80, temperature=0.0, device=device)
        print(f"[{idx}] ({res['tokens_per_second']:.1f} tok/s) {res['full_text'].strip()}")
    return {"test_ce": mean_ce, "test_ppl": mean_ppl, "test_bpb": bpb, "story_mc_acc": cloze_acc}


def run_pre_sft_arithmetic_baseline(model: torch.nn.Module, tokenizer: Tokenizer, device: torch.device) -> list[dict]:
    """Score the fixed 10-item zero-shot addition/subtraction baseline on a Phase 3 base checkpoint."""
    print("=" * 85)
    print("ARITHMETIC BASELINE EVALUATION (Pre-Trained Base Model)")
    print("=" * 85)
    results = []
    for item in PRE_SFT_BASELINE_PROMPTS:
        res = generate_timed(model, tokenizer, item["prompt"], max_new_tokens=15, temperature=0.0, device=device)
        raw = res["completion"].strip()
        verdict = score_arithmetic_completion(raw, item["answer"], answer_format="direct")
        results.append({
            "prompt": item["prompt"].replace("\n", " "),
            "expected": item["answer"],
            "model_output": raw[:30].replace("\n", " "),
            **verdict,
        })
    print(f"{'Prompt':<25}{'Target':<10}{'Raw Model Output':<30}{'Exact Match':<14}{'Valid Format'}")
    print("-" * 85)
    for r in results:
        print(f"{r['prompt']:<25}{r['expected']:<10}{r['model_output']:<30}{str(r['exact_match']):<14}{str(r['valid_format'])}")
    ok = sum(int(r["exact_match"]) for r in results)
    fmt = sum(int(r["valid_format"]) for r in results)
    print("-" * 85)
    print(f"Exact Match Accuracy: {ok}/{len(results)} ({ok / len(results) * 100:.1f}%)")
    print(f"Valid Format Rate:    {fmt}/{len(results)} ({fmt / len(results) * 100:.1f}%)")
    return results


def load_eval_arithmetic_dataset(
    arith_manifest: dict,
    *,
    allow_fallback_build: bool = True,
) -> tuple[Path, dict[str, list[ArithmeticExample]], dict]:
    """Load/restore matching Phase 4 arithmetic dataset from disk/Drive, with offline deterministic fallback."""
    ops = arith_manifest["ops"]
    digits = int(arith_manifest["max_digits"])
    neg = bool(arith_manifest["allow_negative"])
    samples = int(arith_manifest["samples_per_op"])
    seed = int(arith_manifest["seed"])
    fmt = normalize_answer_format(arith_manifest.get("answer_format", "direct"))
    multistep = bool(arith_manifest.get("include_multistep", False))

    try:
        return load_or_restore_arithmetic_dataset(
            ops=ops,
            max_digits=digits,
            allow_negative=neg,
            samples_per_op=samples,
            seed=seed,
            answer_format=fmt,
            include_multistep=multistep,
            use_drive=True,
        )
    except Exception as exc:
        if not allow_fallback_build:
            raise
        print(f"Drive/disk dataset unavailable ({type(exc).__name__}); rebuilding deterministic splits locally...")
        return prepare_arithmetic_dataset(
            ops=ops,
            max_digits=digits,
            allow_negative=neg,
            samples_per_op=samples,
            val_fraction=float(arith_manifest.get("val_fraction", 0.1)),
            test_fraction=float(arith_manifest.get("test_fraction", 0.1)),
            seed=seed,
            answer_format=fmt,
            include_multistep=multistep,
            sync_drive=False,
        )


def evaluate_arithmetic_split(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    model_config,
    arith_manifest: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
    eval_split: str = "val",
    allow_sealed_test: bool = False,
    max_examples: int | None = None,
    batch_size: int = 32,
    use_kv_cache: bool = True,
    max_misses_to_show: int = 15,
    base_model: torch.nn.Module | None = None,
    base_tokenizer: Tokenizer | None = None,
    strict_trace: bool = False,
) -> dict:
    """Benchmark SFT/RL checkpoint on `val` (or `test` in Phase 6) with masked loss and exact-match/strict-trace breakdown."""
    if eval_split == "test" and not allow_sealed_test:
        raise RuntimeError("Sealed `test` split is locked for Phase 6. Set allow_sealed_test=True only for release verification.")

    arith_dir, arith_splits, loaded_manifest = load_eval_arithmetic_dataset(arith_manifest)
    am_format = normalize_answer_format(loaded_manifest.get("answer_format", arith_manifest.get("answer_format", "direct")))
    am_multistep = bool(loaded_manifest.get("include_multistep", arith_manifest.get("include_multistep", False)))
    am_digits = int(loaded_manifest.get("max_digits", arith_manifest.get("max_digits", 3)))

    examples = list(arith_splits[eval_split])
    if max_examples is not None:
        examples = select_stratified_eval_problems(
            examples, int(max_examples), seed=int(loaded_manifest.get("seed", arith_manifest.get("seed", 42))),
        )

    eos_id = tokenizer.token_to_id(EOS)
    encoded = encode_sft_split(examples, tokenizer, context_length=model_config.context_length, eos_id=eos_id)
    max_target_toks = max((sum(1 for t in y if t != -100) for _, y in encoded), default=24)
    max_new_tokens = max(24 if am_format == "direct" else 96, am_digits * 2 + 6, max_target_toks + 24)
    precision = "bf16" if (device.type == "cuda" and dtype == torch.bfloat16) else "fp32"

    print("=" * 88)
    print(
        f"PHASE 4 ARITHMETIC EVALUATION | split={eval_split} ({len(examples):,} rows) | "
        f"format={am_format} | multistep={am_multistep} | kv_cache={use_kv_cache}"
    )
    print(f"Dataset:  {arith_dir}")
    print(f"Hardware: {device.type.upper()} | batch_size={batch_size} | max_new_tokens={max_new_tokens}")
    print("=" * 88)

    t0 = time.perf_counter()
    total_nll = 0.0
    total_targets = 0
    with torch.inference_mode():
        for first in range(0, len(encoded), max(1, batch_size)):
            idxs = list(range(first, min(first + max(1, batch_size), len(encoded))))
            bx, by = collate_sft_batch(encoded, idxs, pad_id=eos_id, device=device)
            unmasked = int((by != -100).sum().item())
            if unmasked > 0:
                total_nll += float(loss_for(model, bx, by, precision).item()) * unmasked
                total_targets += unmasked
    eval_loss = total_nll / max(1, total_targets)
    print(f"Computed masked {eval_split} arith loss: {eval_loss:.4f} ({total_targets:,} targets in {time.perf_counter() - t0:.1f}s)")

    prompts = [ex.prompt for ex in examples]
    gen_start = time.perf_counter()
    next_report = max(200, len(prompts) // 10)

    def _report(done: int, total: int) -> None:
        nonlocal next_report
        if done >= next_report or done == total:
            el = time.perf_counter() - gen_start
            rate = done / max(1e-6, el)
            eta = (total - done) / max(1e-6, rate)
            print(f"  Decoded {done:,}/{total:,} ({done / total * 100:.1f}%) | {el:.1f}s | {rate:.1f} ex/s | ETA {eta:.1f}s", flush=True)
            next_report += max(200, total // 10)

    if use_kv_cache:
        meta_rows = generate_batch_greedy_with_meta(
            model, tokenizer, prompts, max_new_tokens=max_new_tokens,
            device=device, batch_size=batch_size, progress_callback=_report,
        )
        completions = [r["text"] for r in meta_rows]
        eos_flags = [r["ended_with_eos"] for r in meta_rows]
    else:
        completions = []
        eos_flags = []
        for idx, p in enumerate(prompts, start=1):
            text = generate_text(model, tokenizer, p, max_new_tokens=max_new_tokens, device=device, temperature=0.0, use_cache=False)
            completions.append(text)
            tok_cnt = len(tokenizer.encode(text, add_special_tokens=False).ids) if text else 0
            eos_flags.append(tok_cnt < max_new_tokens)
            _report(idx, len(prompts))
    gen_elapsed = time.perf_counter() - gen_start

    exact_hits = 0
    strict_hits = 0
    valid_hits = 0
    work_hits = 0
    eos_hits = 0
    op_stats: dict[str, list[int]] = {}
    cat_stats: dict[str, list[int]] = {}
    scored_rows: list[dict] = []
    passes: list[tuple[ArithmeticExample, str, dict]] = []
    misses: list[tuple[ArithmeticExample, str, dict]] = []

    use_strict = bool(strict_trace and am_format == "scratchpad")
    for ex, raw, ended_eos in zip(examples, completions, eos_flags):
        if use_strict:
            verdict = score_strict_scratchpad_rollout(raw, ex.answer, ended_with_eos=ended_eos)
            passed = bool(verdict["strict_pass"])
            work_hits += int(verdict["valid_work"])
            eos_hits += int(verdict["ended_with_eos"])
        else:
            verdict = score_arithmetic_completion(raw, ex.answer, answer_format=am_format)
            passed = bool(verdict["exact_match"])
        exact_hits += int(verdict["exact_match"])
        strict_hits += int(passed)
        valid_hits += int(verdict["valid_format"])
        op_e = op_stats.setdefault(ex.op, [0, 0])
        op_e[1] += 1
        op_e[0] += int(passed)
        cat_e = cat_stats.setdefault(ex.category, [0, 0])
        cat_e[1] += 1
        cat_e[0] += int(passed)
        row = {"op": ex.op, "category": ex.category, "prompt": ex.prompt, "expected": ex.answer, "completion": raw, **verdict}
        scored_rows.append(row)
        if passed:
            passes.append((ex, raw, verdict))
        else:
            misses.append((ex, raw, verdict))

    n_total = max(1, len(examples))
    print("\n" + "=" * 88)
    print(f"OVERALL {eval_split.upper()} RESULTS ({len(examples):,} problems in {gen_elapsed:.1f}s | {len(examples) / max(1e-6, gen_elapsed):.1f} ex/s)")
    print("=" * 88)
    if use_strict:
        print(f"Strict Trace Pass:    {strict_hits:,}/{len(examples):,} ({strict_hits / n_total * 100:.2f}%)")
    print(f"Exact Match Accuracy: {exact_hits:,}/{len(examples):,} ({exact_hits / n_total * 100:.2f}%)")
    print(f"Valid Format Rate:    {valid_hits:,}/{len(examples):,} ({valid_hits / n_total * 100:.2f}%)")
    if use_strict:
        print(f"Valid Work Trace:     {work_hits:,}/{len(examples):,} ({work_hits / n_total * 100:.2f}%) | EOS rate: {eos_hits / n_total * 100:.2f}%")
    print(f"Masked Arith Loss:    {eval_loss:.4f}")

    print("\nPer-Operation Breakdown:")
    print(f"  {'Op':<10}{'Correct':<12}{'Total':<10}{'Accuracy'}")
    print("  " + "-" * 42)
    for op_name, (ok, cnt) in sorted(op_stats.items()):
        print(f"  {op_name:<10}{ok:<12,}{cnt:<10,}{ok / cnt * 100:.2f}%")

    print("\nPer-Category Breakdown:")
    print(f"  {'Category':<24}{'Correct':<10}{'Total':<10}{'Accuracy'}")
    print("  " + "-" * 54)
    for cat_name, (ok, cnt) in sorted(cat_stats.items(), key=lambda kv: (kv[1][0] / kv[1][1], kv[0])):
        print(f"  {cat_name:<24}{ok:<10,}{cnt:<10,}{ok / cnt * 100:.2f}%")

    nontrivial_passes = [r for r in passes if "zero" not in r[0].category and "unit" not in r[0].category]
    sample_ok = nontrivial_passes or passes
    if sample_ok:
        print("\nSample Passes (up to 5):")
        for ex, raw, _ in sample_ok[:5]:
            print(f"  [PASS] {ex.category} | {ex.prompt.strip().replace(chr(10), ' ')} -> {raw.strip().replace(chr(10), ' // ')[:160]}")

    if misses:
        print(f"\nSample Misses (showing {min(len(misses), max_misses_to_show)} of {len(misses):,}):")
        print("-" * 88)
        for ex, raw, verdict in misses[:max_misses_to_show]:
            print(f"  [{ex.category}] {ex.prompt.strip().replace(chr(10), ' ')}")
            print(f"    Pred ({verdict['extracted']}): {raw.strip().replace(chr(10), ' // ')[:180]}")
            print(f"    Gold ({ex.answer})")

    if base_model is not None and base_tokenizer is not None and examples:
        cmp_n = min(24 if not use_strict else len(examples), len(examples))
        base_meta = generate_batch_greedy_with_meta(
            base_model, base_tokenizer, [ex.prompt for ex in examples[:cmp_n]], max_new_tokens=max_new_tokens, device=device, batch_size=batch_size,
        )
        base_flags = [
            bool(
                score_strict_scratchpad_rollout(bm["text"], ex.answer, ended_with_eos=bm["ended_with_eos"])["strict_pass"]
                if use_strict
                else score_arithmetic_completion(bm["text"], ex.answer, answer_format=am_format)["exact_match"]
            )
            for ex, bm in zip(examples[:cmp_n], base_meta)
        ]
        cur_flags = [
            bool(r.get("strict_pass") if use_strict else r["exact_match"])
            for r in scored_rows[:cmp_n]
        ]
        base_ok = sum(int(b) for b in base_flags)
        cur_ok = sum(int(c) for c in cur_flags)
        fixed = sum(1 for b, c in zip(base_flags, cur_flags) if not b and c)
        broke = sum(1 for b, c in zip(base_flags, cur_flags) if b and not c)
        print("\n" + "-" * 88)
        print(
            f"Paired Comparison ({cmp_n:,} items): Parent={base_ok}/{cmp_n} ({base_ok / cmp_n * 100:.1f}%) -> "
            f"Current={cur_ok}/{cmp_n} ({cur_ok / cmp_n * 100:.1f}%) | fixed={fixed}, broke={broke} (net={fixed - broke:+d})"
        )
    print("=" * 88)

    return {
        "eval_loss": eval_loss,
        "strict_pass": strict_hits / n_total,
        "exact_match": exact_hits / n_total,
        "valid_format": valid_hits / n_total,
        "valid_work": work_hits / n_total if use_strict else exact_hits / n_total,
        "op_stats": op_stats,
        "cat_stats": cat_stats,
        "rows": scored_rows,
        "examples": examples,
        "answer_format": am_format,
        "max_new_tokens": max_new_tokens,
    }


def evaluate_wording_generalization(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    examples: list[ArithmeticExample],
    *,
    device: torch.device,
    answer_format: str = "direct",
    n_problems: int = 12,
    max_new_tokens: int = 96,
    use_kv_cache: bool = True,
) -> dict[str, tuple[int, int]]:
    """Test alternate phrasings (`Calculate`, `equation`, `word_op`, `word_numbers`, conversational) on held-out binary ops."""
    fmt = normalize_answer_format(answer_format)
    use_conv = fmt != "direct"
    binary_examples = [ex for ex in examples if len(ex.canonical_id.split(":")) == 3]
    probes = []
    seen_ids: set[str] = set()
    for ex in select_stratified_eval_problems(binary_examples, max(n_problems * 2, n_problems), seed=42):
        if ex.canonical_id not in seen_ids:
            seen_ids.add(ex.canonical_id)
            parts = ex.canonical_id.split(":")
            probes.append(canonical_problem(parts[0], int(parts[1]), int(parts[2])))
            if len(probes) >= n_problems:
                break

    variant_examples = [
        vex
        for prob in probes
        for vex in render_examples(prob, include_variants=True, answer_format=fmt, include_conversational=use_conv)
        if not vex.variant.endswith("_commuted")
    ]
    if use_kv_cache:
        preds = generate_batch_greedy(model, tokenizer, [v.prompt for v in variant_examples], max_new_tokens=max_new_tokens, device=device, batch_size=32)
    else:
        preds = [
            generate_text(model, tokenizer, v.prompt, max_new_tokens=max_new_tokens, device=device, temperature=0.0, use_cache=False)
            for v in variant_examples
        ]

    stats: dict[str, list[int]] = {}
    for vex, raw in zip(variant_examples, preds):
        verdict = score_arithmetic_completion(raw, vex.answer, answer_format=fmt)
        st = stats.setdefault(vex.variant, [0, 0])
        st[1] += 1
        st[0] += int(verdict["exact_match"])

    print("=" * 66)
    print(f"WORDING GENERALIZATION ({len(probes)} held-out problems, format={fmt})")
    print("=" * 66)
    for vname, (ok, cnt) in sorted(stats.items()):
        print(f"  {vname:<22} {ok:>3}/{cnt:<3} ({ok / max(1, cnt) * 100:5.1f}%)")
    return {k: (v[0], v[1]) for k, v in stats.items()}


def benchmark_kv_cache_parity(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompts: list[str],
    *,
    device: torch.device,
    max_new_tokens: int = 160,
    batch_size: int = 32,
) -> dict:
    """Compare uncached sequential, KV-cached sequential, and batched KV-cached greedy decoding for exact parity and speedup."""
    t0 = time.perf_counter()
    uncached = [generate_text(model, tokenizer, p, max_new_tokens=max_new_tokens, device=device, temperature=0.0, use_cache=False) for p in prompts]
    t_uncached = time.perf_counter() - t0

    t1 = time.perf_counter()
    cached_seq = [generate_text(model, tokenizer, p, max_new_tokens=max_new_tokens, device=device, temperature=0.0, use_cache=True) for p in prompts]
    t_cached = time.perf_counter() - t1

    t2 = time.perf_counter()
    cached_batch = generate_batch_greedy(model, tokenizer, prompts, max_new_tokens=max_new_tokens, device=device, batch_size=batch_size)
    t_batch = time.perf_counter() - t2

    seq_match = (uncached == cached_seq)
    batch_match = (uncached == cached_batch)
    n_tok = sum(len(tokenizer.encode(s, add_special_tokens=False).ids) for s in uncached if s)

    print("=" * 78)
    print(f"KV-CACHE PARITY & SPEEDUP ({len(prompts)} prompts | {n_tok} output tokens | {device.type.upper()})")
    print("=" * 78)
    print(f"Uncached Sequential (`use_cache=False`): {t_uncached:6.2f}s | {n_tok / max(1e-6, t_uncached):6.1f} tok/s | Baseline")
    print(f"KV-Cached Sequential (`use_cache=True`): {t_cached:6.2f}s | {n_tok / max(1e-6, t_cached):6.1f} tok/s | {t_uncached / max(1e-6, t_cached):5.2f}x | parity={seq_match}")
    print(f"KV-Cached Batched (bs={batch_size:<2}):             {t_batch:6.2f}s | {n_tok / max(1e-6, t_batch):6.1f} tok/s | {t_uncached / max(1e-6, t_batch):5.2f}x | parity={batch_match}")
    print("=" * 78)
    return {"uncached_s": t_uncached, "cached_s": t_cached, "batched_s": t_batch, "seq_parity": seq_match, "batch_parity": batch_match}


def create_math_chat_helpers(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    *,
    device: torch.device,
    answer_format: str = "direct",
    use_cache: bool = True,
):
    """Build `(ask_math, chat_math)` helpers tuned for either direct (`arith_v2`) or CoT (`arith_v3`) SFT."""
    fmt = normalize_answer_format(answer_format)
    default_max = 160 if fmt == "scratchpad" else 32

    def _format_prompt(question: str) -> str:
        q = question.strip()
        if fmt == "scratchpad":
            if q.endswith("Answer:"):
                q = q[: -len("Answer:")].strip()
            return q + "\n"
        return q if q.endswith("Answer:") else (q + "\nAnswer:")

    def ask_math(
        question: str,
        *,
        expected: int | None = None,
        max_new_tokens: int = default_max,
        temperature: float = 0.0,
        verbose: bool = True,
    ) -> dict:
        res = generate_timed(
            model, tokenizer, _format_prompt(question),
            max_new_tokens=max_new_tokens, temperature=temperature, use_cache=use_cache, device=device,
        )
        raw = res["completion"].strip()
        verdict = score_arithmetic_completion(raw, expected if expected is not None else 0, answer_format=fmt)
        is_correct = verdict["exact_match"] if expected is not None else None
        if verbose:
            tag = ""
            if is_correct is True:
                tag = " [PASS]"
            elif is_correct is False:
                tag = f" [MISS | expected {expected}]"
            print(f"Q: {question}{tag}")
            for line in raw.splitlines():
                print(f"   {line}")
            if not verdict["valid_format"]:
                print(f"   (invalid format | {res['tokens_generated']} tok | {res['elapsed_seconds'] * 1000:.0f}ms)")
            else:
                print(
                    f"   -> Extracted: {verdict['extracted']} "
                    f"({res['tokens_generated']} tok | {res['elapsed_seconds'] * 1000:.0f}ms | {res['tokens_per_second']:.1f} tok/s)"
                )
            print("-" * 78)
        return {
            "mode": "math",
            "question": question,
            "completion": raw,
            "extracted": verdict["extracted"],
            "valid_format": verdict["valid_format"],
            "correct": is_correct,
            "elapsed_seconds": res["elapsed_seconds"],
        }

    def chat_math():
        print(f"Interactive MiniReasoner Math Chat (format={fmt}, kv_cache={use_cache}, device={device.type.upper()}). Type 'exit' or empty line to quit.")
        while True:
            try:
                user_q = input("\nMath prompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExited chat.")
                break
            if not user_q or user_q.lower() in {"exit", "quit", "q"}:
                print("Exited chat.")
                break
            ask_math(user_q)

    return ask_math, chat_math


_MATH_EXPR_RE = re.compile(r"\d\s*[+\-*/%=]\s*\(?\s*-?\d")
_MATH_VERB_RE = re.compile(
    r"\b(?:calculate|compute|evaluate|solve|remainder|modulo|mod|sum|difference|product|quotient|"
    r"plus|minus|times|divided by|multiplied by|subtract|multiply|divide|what is|how much is)\b",
    re.IGNORECASE,
)


def is_math_prompt(prompt: str) -> bool:
    """Heuristically distinguish arithmetic/math queries from open-ended story prompts."""
    text = prompt.strip()
    if not text:
        return False
    if text.endswith("Answer:") or _MATH_EXPR_RE.search(text):
        return True
    if any(ch.isdigit() for ch in text) and _MATH_VERB_RE.search(text):
        return True
    return bool(re.search(r"\b(?:plus|minus|times|divided by|multiplied by)\b", text, re.IGNORECASE))


def create_ask_helper(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    *,
    device: torch.device,
    answer_format: str = "direct",
    use_cache: bool = True,
):
    """Build a unified `ask(prompt, ...)` callable for both story continuations and arithmetic queries."""
    ask_math, _ = create_math_chat_helpers(
        model, tokenizer, device=device, answer_format=answer_format, use_cache=use_cache,
    )
    fmt = normalize_answer_format(answer_format)
    default_math_max = 160 if fmt == "scratchpad" else 32

    def ask(
        prompt: str,
        *,
        mode: str = "auto",
        expected: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        verbose: bool = True,
    ) -> dict:
        if mode not in ("auto", "math", "story"):
            raise ValueError(f"Unknown ask mode {mode!r}; expected 'auto', 'math', or 'story'.")
        chosen = mode if mode != "auto" else ("math" if (expected is not None or is_math_prompt(prompt)) else "story")
        if chosen == "math":
            return ask_math(
                prompt,
                expected=expected,
                max_new_tokens=default_math_max if max_new_tokens is None else max_new_tokens,
                temperature=0.0 if temperature is None else temperature,
                verbose=verbose,
            )
        story_temp = 0.7 if temperature is None else temperature
        story_max = 120 if max_new_tokens is None else max_new_tokens
        res = generate_timed(
            model,
            tokenizer,
            prompt,
            max_new_tokens=story_max,
            temperature=story_temp,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_cache=use_cache,
            device=device,
        )
        if verbose:
            print(res["full_text"].strip())
            print(
                f"   ({res['tokens_generated']} tok | {res['elapsed_seconds'] * 1000:.0f}ms | {res['tokens_per_second']:.1f} tok/s)"
            )
            print("-" * 78)
        return {
            "mode": "story",
            "question": prompt,
            "completion": res["completion"],
            "full_text": res["full_text"],
            "elapsed_seconds": res["elapsed_seconds"],
        }

    return ask
