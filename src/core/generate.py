from contextlib import nullcontext
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .checkpoint import load_checkpoint
from .model import GPT, GPTConfig
from .tokenizer import EOS


def prepare_prompt_tokens(tokenizer: Tokenizer, prompt: str, context_length: int) -> list[int]:
    """Encode a prompt and print a truncation warning if it exceeds `context_length`."""
    if context_length < 1:
        raise ValueError("context_length must be positive")
    eos_id = tokenizer.token_to_id(EOS)
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not ids:
        if eos_id is None:
            raise ValueError(f"Tokenizer is missing required EOS token {EOS!r}")
        ids = [eos_id]
    if len(ids) > context_length:
        total_tokens = len(ids)
        dropped = total_tokens - context_length
        ids = ids[-context_length:]
        cut_preview = tokenizer.decode(ids[:12])
        print(
            f"Warning: Prompt length ({total_tokens:,} tokens) exceeds context_length ({context_length:,}); "
            f"dropping the first {dropped:,} tokens (cutting at token index {dropped:,} and keeping "
            f"the last {context_length:,} tokens starting at: {cut_preview!r}).",
            flush=True,
        )
    return ids


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    seen_tokens: set[int] | None = None,
) -> int:
    """Select next token ID from final-step logits with optional temperature, top-k, top-p, and repetition penalty."""
    logits = logits.float().clone()
    if repetition_penalty != 1.0 and seen_tokens:
        for prev_token in seen_tokens:
            if logits[prev_token] < 0:
                logits[prev_token] *= repetition_penalty
            else:
                logits[prev_token] /= repetition_penalty
    if temperature <= 0.0 or top_k == 1:
        return int(logits.argmax().item())
    scaled = logits / temperature
    if top_k > 0:
        v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
        scaled[scaled < v[-1]] = -float("Inf")
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        scaled[sorted_indices[sorted_indices_to_remove]] = -float("Inf")
    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.inference_mode()
def generate_text(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    *,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    use_cache: bool = True,
) -> str:
    """Decode continuation text from an already-loaded model and tokenizer with optional KV caching."""
    was_training = model.training
    model.eval()
    try:
        context_length = model.config.context_length
        ids = prepare_prompt_tokens(tokenizer, prompt, context_length)
        generated: list[int] = []
        eos_id = tokenizer.token_to_id(EOS)
        past_kvs = None
        logits = None

        for step in range(max_new_tokens):
            if use_cache and past_kvs is not None and past_kvs[0][0].size(2) < context_length:
                x = torch.tensor([[ids[-1]]], dtype=torch.long, device=device)
                logits, past_kvs = model(x, past_kvs=past_kvs, use_cache=True)
            else:
                window = ids[-context_length:]
                x = torch.tensor([window], dtype=torch.long, device=device)
                if use_cache and len(window) < context_length:
                    logits, past_kvs = model(x, use_cache=True)
                else:
                    logits = model(x)
                    past_kvs = None

            seen = set(ids[-context_length:]) if repetition_penalty != 1.0 else None
            next_id = sample_next_token(
                logits[0, -1],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seen_tokens=seen,
            )
            if next_id == eos_id:
                break
            ids.append(next_id)
            generated.append(next_id)
            if step == max_new_tokens - 1:
                break
        return tokenizer.decode(generated)
    finally:
        if was_training:
            model.train()


@torch.inference_mode()
def generate_batch_greedy(
    model: GPT,
    tokenizer: Tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    device: torch.device,
    *,
    batch_size: int = 32,
    progress_callback=None,
) -> list[str]:
    """Decode greedy completions across equal-prompt-length batches using KV caching (exact parity with `generate_text`)."""
    if not prompts:
        return []
    was_training = model.training
    model.eval()
    try:
        context_length = model.config.context_length
        eos_id = tokenizer.token_to_id(EOS)
        if eos_id is None:
            raise ValueError(f"Tokenizer is missing required EOS token {EOS!r}")
        encoded = [prepare_prompt_tokens(tokenizer, p, context_length) for p in prompts]

        by_len: dict[int, list[int]] = {}
        for idx, ids in enumerate(encoded):
            by_len.setdefault(len(ids), []).append(idx)

        outputs = [""] * len(prompts)
        completed = 0
        total = len(prompts)

        for prompt_len in sorted(by_len):
            group = by_len[prompt_len]
            for start in range(0, len(group), batch_size):
                chunk = group[start : start + batch_size]
                bs = len(chunk)
                x0 = torch.tensor([encoded[i] for i in chunk], dtype=torch.long, device=device)
                finished = torch.zeros(bs, dtype=torch.bool, device=device)
                generated_rows: list[list[int]] = [[] for _ in range(bs)]

                can_cache = (prompt_len + max_new_tokens) <= context_length
                past_kvs = None
                cur_seqs = None
                if can_cache:
                    logits, past_kvs = model(x0, use_cache=True)
                else:
                    cur_seqs = [list(encoded[i]) for i in chunk]
                    logits = model(x0)

                for step in range(max_new_tokens):
                    next_ids = logits[:, -1].float().argmax(dim=-1)
                    newly_done = next_ids == eos_id
                    next_list = next_ids.tolist()
                    fin_list = finished.tolist()
                    for row_i, tok_id in enumerate(next_list):
                        if not fin_list[row_i]:
                            if tok_id != eos_id:
                                generated_rows[row_i].append(tok_id)
                            if cur_seqs is not None:
                                cur_seqs[row_i].append(tok_id)
                    finished = finished | newly_done
                    if bool(finished.all().item()) or step == max_new_tokens - 1:
                        break
                    if can_cache:
                        step_x = torch.where(finished, eos_id, next_ids)[:, None]
                        logits, past_kvs = model(step_x, past_kvs=past_kvs, use_cache=True)
                    else:
                        step_x = torch.tensor(
                            [seq[-context_length:] for seq in cur_seqs],
                            dtype=torch.long,
                            device=device,
                        )
                        logits = model(step_x)

                for row_i, orig_idx in enumerate(chunk):
                    outputs[orig_idx] = tokenizer.decode(generated_rows[row_i])
                completed += bs
                if progress_callback is not None:
                    progress_callback(completed, total)

        return outputs
    finally:
        if was_training:
            model.train()


@torch.inference_mode()
def complete(
    checkpoint_path: Path,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    log_path: Path | None = None,
) -> str:
    if log_path is not None:
        from ..infra.tracking import tee_to_log

        log_ctx = tee_to_log(log_path)
    else:
        log_ctx = nullcontext()
    with log_ctx:
        checkpoint = load_checkpoint(checkpoint_path)
        tokenizer = Tokenizer.from_str(checkpoint["tokenizer_json"])
        config = GPTConfig(**checkpoint["model_config"])
        model = GPT(config).to(device).eval()
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        return generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
            device,
            temperature=temperature,
            top_p=top_p,
        )
