"""Unit and integration tests for Phase 4 Arithmetic SFT and src reorganization."""

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
import torch

from src.core.checkpoint import build_model_config, load_checkpoint
from src.core.model import GPT, GPTConfig
from src.core.tokenizer import EOS, sha256
from src.data.arithmetic import (
    GENERATOR_VERSION,
    SCRATCHPAD_GENERATOR_VERSION,
    build_arithmetic_dataset,
    canonical_problem,
    collate_sft_batch,
    dataset_slug,
    encode_sft_example,
    format_work_trace,
    load_or_restore_arithmetic_dataset,
    prepare_arithmetic_dataset,
    score_arithmetic_completion,
)
from src.infra import azure_blob, checkpoint_store
from src.training import sft


def _make_byte_tokenizer() -> Tokenizer:
    vocab = {EOS: 0}
    for i in range(256):
        ch = bytes([i]).decode("latin-1")
        if ch not in vocab:
            vocab[ch] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def test_arithmetic_all_ops_digits_and_splits():
    """Verify add/sub/mul/div (+ mod), negative integers, 3- and 4-digit scaling, and zero leakage."""
    assert canonical_problem("add", -7, 8).canonical_id == canonical_problem("add", 8, -7).canonical_id
    assert canonical_problem("mul", -3, 12).canonical_id == canonical_problem("mul", 12, -3).canonical_id
    assert canonical_problem("sub", 15, 7).canonical_id != canonical_problem("sub", 7, 15).canonical_id
    assert canonical_problem("sub", 7, 15).answer == -8
    assert canonical_problem("div", -144, 12).answer == -12
    assert canonical_problem("div", -144, -12).answer == 12
    assert canonical_problem("mod", 58, 8).answer == 2

    splits, manifest = build_arithmetic_dataset(samples_per_op=400, seed=42)  # Defaults: 4 ops, 3 digits, signed
    assert manifest["generator_version"] == GENERATOR_VERSION
    assert manifest["ops"] == ["add", "sub", "mul", "div"]
    assert manifest["max_digits"] == 3 and manifest["allow_negative"] is True
    assert any(ex.answer < 0 for ex in splits["train"]), "Expected negative arithmetic examples by default"
    assert set(manifest["op_counts"]["val"]) == {"add", "sub", "mul", "div"}

    ids_by_split = {name: {ex.canonical_id for ex in items} for name, items in splits.items()}
    assert ids_by_split["train"].isdisjoint(ids_by_split["val"])
    assert ids_by_split["train"].isdisjoint(ids_by_split["test"])
    assert ids_by_split["val"].isdisjoint(ids_by_split["test"])

    for items in splits.values():
        for ex in items:
            a, b = (int(v) for v in ex.canonical_id.split(":")[1:])
            if ex.op == "add":
                expected = a + b
            elif ex.op == "sub":
                expected = a - b
            elif ex.op == "mul":
                expected = a * b
            else:
                expected = (-1 if (a < 0) ^ (b < 0) else 1) * (abs(a) // abs(b))
            assert ex.answer == expected and ex.answer_text == str(expected)

    # Verify > 3 digits (e.g. 4 digits) + mod remain cleanly supported
    splits_4d, man_4d = build_arithmetic_dataset(
        ops=("add", "sub", "mul", "div", "mod"), max_digits=4, samples_per_op=200, seed=7,
    )
    assert man_4d["max_digits"] == 4 and "mod" in man_4d["ops"]
    assert any(max(abs(int(v)) for v in ex.canonical_id.split(":")[1:]) >= 1000 for ex in splits_4d["train"])
    ids_4d = {k: {ex.canonical_id for ex in v} for k, v in splits_4d.items()}
    assert ids_4d["train"].isdisjoint(ids_4d["val"]) and ids_4d["train"].isdisjoint(ids_4d["test"])
    print("  [PASSED] All 4 ops (+ mod), 3+ digits, negatives, and leakage-free canonical splits")


def test_disk_persistence_and_guard():
    """Verify prepared dataset saves to disk cleanly with SHA-256 verification and training fails loudly if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        data_dir = tmp / "arith_bundle"

        try:
            load_or_restore_arithmetic_dataset(dataset_dir=data_dir, use_drive=False)
            raise AssertionError("Expected FileNotFoundError when dataset has not been prepared yet")
        except FileNotFoundError:
            pass

        saved_dir, splits, manifest = prepare_arithmetic_dataset(
            output_dir=data_dir, samples_per_op=120, seed=42, sync_drive=False,
        )
        assert saved_dir == data_dir
        for fname in ("train.jsonl", "val.jsonl", "test.jsonl", "manifest.json"):
            assert (data_dir / fname).is_file()

        _, reloaded, rel_man = load_or_restore_arithmetic_dataset(dataset_dir=data_dir, use_drive=False)
        assert len(reloaded["train"]) == len(splits["train"])
        assert rel_man["files"]["val.jsonl"]["sha256"] == manifest["files"]["val.jsonl"]["sha256"]
    print("  [PASSED] Standalone dataset persistence, SHA-256 verification, and missing-data guard")


def test_sft_masking_and_verifier():
    """Verify prompt -100 masking, EOS target preservation, and strict deterministic verifier."""
    tok = _make_byte_tokenizer()
    eos_id = tok.token_to_id(EOS)
    prompt = "What is -144 / 12?\nAnswer:"
    prompt_len = len(tok.encode(prompt, add_special_tokens=False).ids)
    x, y = encode_sft_example(tok, prompt, "-12", context_length=128, eos_id=eos_id)

    assert len(x) == len(y)
    assert all(val == -100 for val in y[: prompt_len - 1])
    ans_tokens = tok.encode("-12", add_special_tokens=False).ids + [eos_id]
    assert y[prompt_len - 1 :] == ans_tokens

    bx, by = collate_sft_batch([(x, y), (x[:10], y[:10])], [0, 1], pad_id=eos_id, device=torch.device("cpu"))
    assert bx.shape == by.shape == (2, len(x))
    assert (by[1, 10:] == -100).all()

    assert score_arithmetic_completion("0", 0) == {"valid_format": True, "extracted": 0, "exact_match": True, "reward": 1.0}
    assert score_arithmetic_completion(" -12\n", -12)["exact_match"] is True
    assert score_arithmetic_completion("55", 54) == {"valid_format": True, "extracted": 55, "exact_match": False, "reward": 0.0}
    for bad in ("", "   ", "55 56", "The answer is 55", "1: What is 2?"):
        res = score_arithmetic_completion(bad, 55)
        assert res["valid_format"] is False and res["reward"] == 0.0
    print("  [PASSED] Prompt -100 masking and deterministic arithmetic verifier")


def test_rope_context_extension_for_sft():
    """Verify SFT can load base weights cleanly even when extending context_length (e.g. 1024 -> 2048)."""
    tok = _make_byte_tokenizer()
    vocab_size = tok.get_vocab_size()
    base_cfg = GPTConfig(vocab_size=vocab_size, context_length=64, n_layer=2, n_head=4, n_embd=64)
    base_model = GPT(base_cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "base.pt"
        torch.save({
            "version": 1,
            "model_config": asdict(base_cfg),
            "model": base_model.state_dict(),
            "tokenizer_json": tok.to_str(),
            "step": 26047,
            "metrics": [{"step": 26047, "val_loss": 1.13}],
        }, base_path)

        ckpt = load_checkpoint(base_path)
        ext_cfg = build_model_config(ckpt, context_length=128)
        assert ext_cfg.context_length == 128
        ext_model = GPT(ext_cfg)
        ext_model.load_state_dict(ckpt["model"])

        x = torch.randint(0, vocab_size, (2, 110))
        y = torch.randint(0, vocab_size, (2, 110))
        loss = torch.nn.functional.cross_entropy(ext_model(x).flatten(0, 1), y.flatten())
        loss.backward()
        assert torch.isfinite(loss)

        cfg_2048 = build_model_config({"model_config": {"vocab_size": 1024, "context_length": 1024}}, context_length=2048)
        assert cfg_2048.context_length == 2048
    print("  [PASSED] SFT RoPE weight compatibility across extended context_length")


class _FakeBlobClient:
    def __init__(self, container, blob_name: str):
        self.container = container
        self.blob_name = blob_name

    def upload_blob(self, stream, *, overwrite: bool = False, metadata: dict | None = None):
        payload = stream.read()
        self.container.data[self.blob_name] = payload
        self.container.blobs[self.blob_name] = len(payload)


class _FakeContainer:
    url = "https://example.blob.core.windows.net/minireasoner"

    def __init__(self):
        self.blobs: dict[str, int] = {}
        self.data: dict[str, bytes] = {}

    def list_blob_names(self, *, name_starts_with: str = "", results_per_page=None):
        return [name for name in sorted(self.blobs) if name.startswith(name_starts_with)]

    def get_blob_client(self, blob_name: str):
        return _FakeBlobClient(self, blob_name)


class _FakeRunHandle:
    def __init__(self, run_id: str):
        self.id = run_id
        self.tags: dict[str, str] = {}
        self.metrics: list[tuple[int, dict]] = []
        self.artifacts: dict[str, bytes] = {}

    def log(self, metrics: dict, *, step: int) -> None:
        self.metrics.append((step, dict(metrics)))

    def flush(self) -> None:
        pass

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def log_artifact(self, path, artifact_path: str = "logs") -> None:
        p = Path(path)
        self.artifacts[f"{artifact_path}/{p.name}"] = p.read_bytes()


def test_run_sft_end_to_end():
    """Verify full SFT pilot loop consuming prepared data from disk with retention & Azure/MLflow outputs."""
    tok = _make_byte_tokenizer()
    vocab_size = tok.get_vocab_size()
    base_cfg = GPTConfig(vocab_size=vocab_size, context_length=64, n_layer=2, n_head=4, n_embd=64)
    base_model = GPT(base_cfg)
    container = _FakeContainer()
    run_handle = _FakeRunHandle("sft_run_test")

    @contextmanager
    def fake_track_run(*args, **kwargs):
        yield run_handle

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        logs_dir = tmp / "logs"
        data_dir = tmp / "prepared_arith"
        prepare_arithmetic_dataset(
            output_dir=data_dir, samples_per_op=100, seed=42, sync_drive=False,
        )

        base_ckpt_path = tmp / "run_base" / "latest_step_26047.pt"
        base_ckpt_path.parent.mkdir(parents=True)
        torch.save({
            "version": 1,
            "model_config": asdict(base_cfg),
            "model": base_model.state_dict(),
            "tokenizer_json": tok.to_str(),
            "step": 26047,
            "metrics": [{"step": 26047, "val_loss": 1.1303}],
        }, base_ckpt_path)

        tok_dir = tmp / "tokenized"
        tok_dir.mkdir()
        tok.save(str(tok_dir / "tokenizer.json"))
        arr = np.arange(512, dtype=np.uint16) % vocab_size
        arr.tofile(tok_dir / "train.bin")
        arr.tofile(tok_dir / "val.bin")
        manifest = {
            "version": 1,
            "vocab_size": vocab_size,
            "tokenizer_sha256": sha256(tok_dir / "tokenizer.json"),
            "train": {"tokens": len(arr), "sha256": sha256(tok_dir / "train.bin")},
            "val": {"tokens": len(arr), "sha256": sha256(tok_dir / "val.bin")},
        }

        with (
            patch.dict("os.environ", {"DATABRICKS_HOST": "https://example.cloud.databricks.com", "DATABRICKS_TOKEN": "tok"}),
            patch.object(azure_blob, "container_client", return_value=container),
            patch.object(checkpoint_store, "container_client", return_value=container),
            patch("src.training.sft.ensure_tokenized_corpus", return_value=manifest),
            patch("src.training.common.track_run", fake_track_run),
        ):
            res = sft.run_sft(
                preset="sft_pilot",
                data_dir=data_dir,
                no_drive=True,
                base_checkpoint=base_ckpt_path,
                context_length=96,
                steps=6,
                batch_size=4,
                eval_every=3,
                retention_ratio=0.35,
                device=torch.device("cpu"),
                logs_dir=logs_dir,
                retention_tokenized_dir=tok_dir,
            )

        assert res["status"] == "Completed" and res["step"] == 6
        assert "val_exact_match" in res and "val_valid_format" in res
        assert set(res["op_accuracy"]) == {"add", "sub", "mul", "div"}
        assert res["story_val_loss"] is not None
        assert res["parent_checkpoint_step"] == 26047
        assert run_handle.tags["stage"] == "sft"
        assert run_handle.tags["context_length"] == "96"
        assert len(list(logs_dir.iterdir())) == 1
        assert any(k.endswith(".pt") for k in container.blobs)
        assert any("/logs/train_" in k and k.endswith(".log") for k in container.blobs)
        assert any(k.startswith("logs/train_") for k in run_handle.artifacts)
    print("  [PASSED] End-to-end Phase 4 SFT pilot reading prepared dataset with 4 ops & retention")


def _assert_valid_work_trace(work_line: str, expected_answer: int) -> None:
    assert work_line.startswith("Work: "), f"Missing Work prefix: {work_line!r}"
    parts = [p.strip() for p in work_line[len("Work: "):].split(";")]
    assert parts and all(parts)
    last_val = None
    for part in parts:
        lhs, rhs = part.split("=", 1)
        rhs_val = int(rhs)
        if lhs == "neg":
            assert last_val is not None and rhs_val == -last_val
        elif "/" in lhs:
            a_s, b_s = lhs.split("/", 1)
            a_i = int(a_s)
            b_i = int(b_s.strip("()"))
            assert a_i % b_i == 0 and (a_i // b_i == rhs_val or (-1 if (a_i < 0) ^ (b_i < 0) else 1) * (abs(a_i) // abs(b_i)) == rhs_val)
        else:
            cleaned_lhs = lhs.replace("(", "").replace(")", "")
            assert eval(cleaned_lhs, {"__builtins__": {}}, {}) == rhs_val
        last_val = rhs_val
    assert last_val == expected_answer, f"{work_line!r} ended with {last_val}, expected {expected_answer}"


def test_scratchpad_traces_multistep_and_verifier():
    """Verify numeric Work traces, multi-step conversational generation, leakage-free splits, and parser."""
    edge_cases = [
        canonical_problem("mul", 177, 515),
        canonical_problem("mul", -95, 672),
        canonical_problem("mul", 6, 45),
        canonical_problem("mul", 0, -99),
        canonical_problem("add", -36, 545),
        canonical_problem("add", 89, 498),
        canonical_problem("add", -545, 36),
        canonical_problem("add", -120, -45),
        canonical_problem("sub", -494, -377),
        canonical_problem("sub", 475, 256),
        canonical_problem("sub", 1, 881),
        canonical_problem("sub", 36, -545),
        canonical_problem("div", 624, 16),
        canonical_problem("div", -998, -1),
        canonical_problem("div", 0, -17),
        canonical_problem("mod", 58, 8),
    ]
    for prob in edge_cases:
        _assert_valid_work_trace(format_work_trace(prob), prob.answer)

    slug = dataset_slug(answer_format="scratchpad", include_multistep=True)
    assert slug.startswith(f"{SCRATCHPAD_GENERATOR_VERSION}_scratchpad_multistep_")

    splits, manifest = build_arithmetic_dataset(
        samples_per_op=200, seed=42, answer_format="scratchpad", include_multistep=True,
    )
    assert manifest["generator_version"] == SCRATCHPAD_GENERATOR_VERSION
    assert manifest["answer_format"] == "scratchpad"
    assert manifest["include_multistep"] is True
    assert set(manifest["op_counts"]["val"]) == {"add", "sub", "mul", "div", "multi"}

    ids_by_split = {name: {ex.canonical_id for ex in items} for name, items in splits.items()}
    assert ids_by_split["train"].isdisjoint(ids_by_split["val"])
    assert ids_by_split["train"].isdisjoint(ids_by_split["test"])
    assert ids_by_split["val"].isdisjoint(ids_by_split["test"])

    val_categories = {ex.category for ex in splits["val"]}
    for expected_cat in ("multi_seq_sym", "multi_seq_word", "multi_combine_sym", "multi_combine_word", "multi_pemdas"):
        assert expected_cat in val_categories

    for items in splits.values():
        for ex in items:
            assert ex.prompt.endswith("\n") and not ex.prompt.endswith("\nAnswer:")
            work_line, ans_line = ex.answer_text.split("\n", 1)
            assert ans_line == f"Answer: {ex.answer}"
            _assert_valid_work_trace(work_line, ex.answer)
            verdict = score_arithmetic_completion(ex.answer_text, ex.answer, answer_format="scratchpad")
            assert verdict == {"valid_format": True, "extracted": ex.answer, "exact_match": True, "reward": 1.0}

    for bad in (
        "25",
        "Answer: 25",
        "Work: \nAnswer: 25",
        "Work: 10+10=20\nAnswer: 20 extra",
        "Work: 10+10=20\nAnswer: 20\nAnswer: 20",
    ):
        res = score_arithmetic_completion(bad, 25, answer_format="scratchpad")
        assert res["valid_format"] is False and res["reward"] == 0.0
    print("  [PASSED] Numeric scratchpad traces, multi-step conversational splits, and verifier")


def test_run_sft_scratchpad_end_to_end():
    """Verify full SFT loop with scratchpad + multi-step preset, dynamic eval token budget, and logging."""
    tok = _make_byte_tokenizer()
    vocab_size = tok.get_vocab_size()
    base_cfg = GPTConfig(vocab_size=vocab_size, context_length=256, n_layer=2, n_head=4, n_embd=64)
    base_model = GPT(base_cfg)
    container = _FakeContainer()
    run_handle = _FakeRunHandle("sft_scratchpad_test")

    @contextmanager
    def fake_track_run(*args, **kwargs):
        yield run_handle

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        logs_dir = tmp / "logs"
        data_dir = tmp / "prepared_scratchpad"
        prepare_arithmetic_dataset(
            output_dir=data_dir,
            samples_per_op=120,
            seed=42,
            answer_format="scratchpad",
            include_multistep=True,
            sync_drive=False,
        )

        base_ckpt_path = tmp / "run_base" / "latest_step_26047.pt"
        base_ckpt_path.parent.mkdir(parents=True)
        torch.save({
            "version": 1,
            "model_config": asdict(base_cfg),
            "model": base_model.state_dict(),
            "tokenizer_json": tok.to_str(),
            "step": 26047,
            "metrics": [{"step": 26047, "val_loss": 1.1303}],
        }, base_ckpt_path)

        tok_dir = tmp / "tokenized"
        tok_dir.mkdir()
        tok.save(str(tok_dir / "tokenizer.json"))
        arr = np.arange(512, dtype=np.uint16) % vocab_size
        arr.tofile(tok_dir / "train.bin")
        arr.tofile(tok_dir / "val.bin")
        manifest = {
            "version": 1,
            "vocab_size": vocab_size,
            "tokenizer_sha256": sha256(tok_dir / "tokenizer.json"),
            "train": {"tokens": len(arr), "sha256": sha256(tok_dir / "train.bin")},
            "val": {"tokens": len(arr), "sha256": sha256(tok_dir / "val.bin")},
        }

        with (
            patch.dict("os.environ", {"DATABRICKS_HOST": "https://example.cloud.databricks.com", "DATABRICKS_TOKEN": "tok"}),
            patch.object(azure_blob, "container_client", return_value=container),
            patch.object(checkpoint_store, "container_client", return_value=container),
            patch("src.training.sft.ensure_tokenized_corpus", return_value=manifest),
            patch("src.training.common.track_run", fake_track_run),
        ):
            res = sft.run_sft(
                preset="sft_scratchpad",
                data_dir=data_dir,
                samples_per_op=120,
                no_drive=True,
                base_checkpoint=base_ckpt_path,
                context_length=256,
                steps=4,
                batch_size=4,
                eval_every=2,
                retention_ratio=0.25,
                device=torch.device("cpu"),
                logs_dir=logs_dir,
                retention_tokenized_dir=tok_dir,
            )

        assert res["status"] == "Completed" and res["step"] == 4
        assert res["answer_format"] == "scratchpad" and res["include_multistep"] is True
        assert set(res["op_accuracy"]) == {"add", "sub", "mul", "div", "multi"}
        assert run_handle.tags["arith_format"] == "scratchpad"
    print("  [PASSED] End-to-end Phase 4 SFT scratchpad & multi-step loop")


def test_prepare_data_cot_alias_and_benchmark_helpers():
    """Verify `cot` alias, stratified evaluation subsets, `--stage corpus --no-drive`, and KV-cache helpers."""
    from src import main as cli_main
    from src.core.benchmark import (
        benchmark_kv_cache_parity,
        create_ask_helper,
        create_math_chat_helpers,
        evaluate_arithmetic_split,
        evaluate_wording_generalization,
        is_math_prompt,
    )
    from src.data.arithmetic import normalize_answer_format

    assert normalize_answer_format("cot") == "scratchpad"
    assert normalize_answer_format("scratchpad") == "scratchpad"
    assert normalize_answer_format("direct") == "direct"

    tok = _make_byte_tokenizer()
    cfg = GPTConfig(vocab_size=tok.get_vocab_size(), context_length=128, n_layer=2, n_head=4, n_embd=64)
    model = GPT(cfg).eval()
    device = torch.device("cpu")

    parity = benchmark_kv_cache_parity(
        model,
        tok,
        ["What is 12 + 34?\n", "Calculate 99 - 41.\n"],
        device=device,
        max_new_tokens=16,
        batch_size=2,
    )
    assert parity["seq_parity"] is True
    assert parity["batch_parity"] is True

    ask_cot, _ = create_math_chat_helpers(model, tok, device=device, answer_format="cot", use_cache=True)
    cot_out = ask_cot("What is 2 + 2?", max_new_tokens=8, verbose=False)
    assert isinstance(cot_out["completion"], str) and cot_out["elapsed_seconds"] >= 0.0

    ask_direct, _ = create_math_chat_helpers(model, tok, device=device, answer_format="direct", use_cache=False)
    direct_out = ask_direct("What is 2 + 2?", max_new_tokens=8, verbose=False)
    assert isinstance(direct_out["completion"], str) and direct_out["elapsed_seconds"] >= 0.0

    assert is_math_prompt("What is 47 + 8?") is True
    assert is_math_prompt("What is twelve plus eight?") is True
    assert is_math_prompt("Start with 100, subtract 35, and multiply by 4.") is True
    assert is_math_prompt("Once upon a time") is False
    assert is_math_prompt("Once upon a time, 3 little foxes lived in a forest.") is False

    ask = create_ask_helper(model, tok, device=device, answer_format="direct", use_cache=True)
    story_out = ask("Once upon a time", max_new_tokens=8, verbose=False)
    assert story_out["mode"] == "story" and story_out["full_text"].startswith("Once upon a time")
    math_out = ask("What is 47 + 8?", max_new_tokens=8, verbose=False)
    assert math_out["mode"] == "math" and "valid_format" in math_out
    forced_story = ask("2 + 2", mode="story", max_new_tokens=6, verbose=False)
    assert forced_story["mode"] == "story"
    try:
        ask("hello", mode="bad")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid ask() mode")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "arith_dir"
        _, splits, manifest = prepare_arithmetic_dataset(
            output_dir=out_dir, samples_per_op=100, seed=42, sync_drive=False,
        )
        with patch("src.core.benchmark.load_eval_arithmetic_dataset", return_value=(out_dir, splits, manifest)):
            eval_out = evaluate_arithmetic_split(
                model,
                tok,
                cfg,
                manifest,
                device=device,
                dtype=torch.float32,
                max_examples=12,
                batch_size=6,
                use_kv_cache=True,
                max_misses_to_show=0,
            )
        assert len(eval_out["examples"]) == 12
        assert set(eval_out["op_stats"]) == {"add", "div", "mul", "sub"}

        # Sorted val rows start with ~100 add examples; wording probes must still cover all 4 ops when n_problems=4.
        with patch("src.core.benchmark.canonical_problem", wraps=canonical_problem) as spy_canon:
            evaluate_wording_generalization(
                model, tok, splits["val"], device=device, answer_format="direct", n_problems=4, max_new_tokens=4, use_kv_cache=True,
            )
        probed_ops = {call.args[0] for call in spy_canon.call_args_list}
        assert probed_ops == {"add", "sub", "mul", "div"}

    with patch.object(cli_main, "ensure_tokenized_corpus", return_value={}) as mock_corpus:
        cli_main.main(["prepare-data", "--stage", "corpus", "--no-drive"])
        assert mock_corpus.call_args.kwargs["sync_drive"] is False
    print("  [PASSED] `cot` alias, stratified benchmark/wording subsets, and `--stage corpus --no-drive`")


if __name__ == "__main__":
    print("Running Phase 4 Arithmetic SFT test suite...")
    test_arithmetic_all_ops_digits_and_splits()
    test_disk_persistence_and_guard()
    test_sft_masking_and_verifier()
    test_rope_context_extension_for_sft()
    test_run_sft_end_to_end()
    test_scratchpad_traces_multistep_and_verifier()
    test_run_sft_scratchpad_end_to_end()
    test_prepare_data_cot_alias_and_benchmark_helpers()
    print("All tests in test_sft.py passed successfully!")
