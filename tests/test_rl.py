"""Unit and integration tests for Phase 5 GRPO RL and strict binary scratchpad verifier."""

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

from src.core.generate import generate_batch_greedy_with_meta, generate_batch_rollouts, generate_text
from src.core.model import GPT, GPTConfig
from src.core.tokenizer import EOS, sha256
from src.data.arithmetic import (
    build_arithmetic_dataset,
    load_rl_hard_dataset,
    prepare_arithmetic_dataset,
    score_strict_scratchpad_rollout,
    verify_work_trace,
)
from src.infra import azure_blob, checkpoint_store
from src.infra.checkpoint_store import CheckpointStore
from src.training import grpo
from src.training.grpo import compute_group_advantages, prepare_hard_rl_dataset


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


class _FakeBlobHandle:
    def __init__(self, container, name: str):
        self.container = container
        self.blob_name = name

    def upload_blob(self, stream, *, overwrite: bool = False, metadata: dict | None = None):
        payload = stream.read() if hasattr(stream, "read") else bytes(stream)
        self.container.blobs[self.blob_name] = payload


class _FakeContainer:
    url = "https://example.blob.core.windows.net/minireasoner"

    def __init__(self):
        self.blobs: dict[str, bytes] = {}

    def list_blob_names(self, *, name_starts_with: str = "", results_per_page=None):
        return [name for name in sorted(self.blobs) if name.startswith(name_starts_with)]

    def get_blob_client(self, blob_name: str):
        return _FakeBlobHandle(self, blob_name)

    def download_blob(self, blob_name: str):
        class _DL:
            def __init__(self, data: bytes):
                self.data = data

            def readinto(self, stream):
                stream.write(self.data)

        return _DL(self.blobs[blob_name])


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


def test_strict_binary_verifier_all_conditions():
    """Verify strictly binary {0.0, 1.0} rewards across EOS, regex, trace equations, tautologies, and final match."""
    # Condition 1-4 all pass
    v1 = score_strict_scratchpad_rollout(
        "Work: 515*100=51500; 515*70=36050; 515*7=3605; 51500+36050=87550; 87550+3605=91155\nAnswer: 91155",
        91155,
        ended_with_eos=True,
    )
    assert v1["strict_pass"] is True and v1["reward"] == 1.0

    # Signed neg= and exact division
    v_neg = score_strict_scratchpad_rollout("Work: 928+2=930; neg=-930\nAnswer: -930", -930, ended_with_eos=True)
    assert v_neg["reward"] == 1.0
    v_div = score_strict_scratchpad_rollout("Work: 860/860=1; neg=-1\nAnswer: -1", -1, ended_with_eos=True)
    assert v_div["reward"] == 1.0
    v_sum = score_strict_scratchpad_rollout(
        "Work: 7*100=700; 910-700=210; 7*30=210; 100+30=130\nAnswer: 130", 130, ended_with_eos=True,
    )
    assert v_sum["reward"] == 1.0

    # Condition 1 failure: missing <|eos|> even with valid text & right answer
    v_no_eos = score_strict_scratchpad_rollout("Work: 47+8=55\nAnswer: 55", 55, ended_with_eos=False)
    assert v_no_eos["strict_pass"] is False and v_no_eos["reward"] == 0.0

    # Trailing junk or missing Work
    assert score_strict_scratchpad_rollout("Work: 2+2=4\nAnswer: 4\njunk", 4, ended_with_eos=True)["reward"] == 0.0
    assert score_strict_scratchpad_rollout("4", 4, ended_with_eos=True)["reward"] == 0.0

    # Condition 2 failure: intermediate slip even when final clause & Answer get lucky
    v_slip = score_strict_scratchpad_rollout("Work: 515*70=35050; 2+2=4\nAnswer: 4", 4, ended_with_eos=True)
    assert v_slip["valid_work"] is False and v_slip["reward"] == 0.0

    # Condition 2 failure: tautologies with no operator or neg= at start
    for taut in ("Work: 91155=91155\nAnswer: 91155", "Work: -5=-5\nAnswer: -5", "Work: (42)=42\nAnswer: 42", "Work: neg=-5\nAnswer: -5"):
        assert score_strict_scratchpad_rollout(taut, int(taut.split()[-1]), ended_with_eos=True)["reward"] == 0.0

    # Inexact or zero division + disallowed AST constructs (no eval)
    assert score_strict_scratchpad_rollout("Work: 7/2=3\nAnswer: 3", 3, ended_with_eos=True)["reward"] == 0.0
    assert score_strict_scratchpad_rollout("Work: 5/0=0\nAnswer: 0", 0, ended_with_eos=True)["reward"] == 0.0
    assert score_strict_scratchpad_rollout("Work: 2**3=8\nAnswer: 8", 8, ended_with_eos=True)["reward"] == 0.0
    assert score_strict_scratchpad_rollout("Work: True+1=2\nAnswer: 2", 2, ended_with_eos=True)["reward"] == 0.0
    assert score_strict_scratchpad_rollout("Work: __import__('os').getpid()=1\nAnswer: 1", 1, ended_with_eos=True)["reward"] == 0.0

    # Condition 3 failure: trace last RHS != Answer
    assert score_strict_scratchpad_rollout("Work: 2+2=4\nAnswer: 5", 5, ended_with_eos=True)["reward"] == 0.0

    # Condition 4 failure: wrong ground truth
    assert score_strict_scratchpad_rollout("Work: 2+2=4\nAnswer: 4", 5, ended_with_eos=True)["reward"] == 0.0

    # Verify every single generated arith_v3 trace passes strict verifier
    splits, _ = build_arithmetic_dataset(samples_per_op=120, seed=42, answer_format="scratchpad", include_multistep=True)
    for ex in splits["val"]:
        res = score_strict_scratchpad_rollout(ex.answer_text, ex.answer, ended_with_eos=True)
        assert res["reward"] == 1.0, f"Generated arith_v3 sample failed strict verifier: {ex.answer_text!r}"
        ok, _ = verify_work_trace(ex.answer_text.splitlines()[0].removeprefix("Work: "))
        assert ok is True
    print("  [PASSED] Strict binary {0.0, 1.0} trace verifier across all 4 conditions and arith_v3 corpus")


def test_batched_rollouts_and_group_advantages():
    """Verify KV-cached rollout shapes/EOS tokens, greedy parity, and GRPO group advantage normalization."""
    tok = _make_byte_tokenizer()
    eos_id = tok.token_to_id(EOS)
    cfg = GPTConfig(vocab_size=tok.get_vocab_size(), context_length=64, n_layer=2, n_head=4, n_embd=64)
    model = GPT(cfg).eval()
    device = torch.device("cpu")

    prompts = ["What is 12 + 34?\n", "What is 9 - 2?\n", "Compute 3 * 5 then add 2.\n"]
    seq_texts = [generate_text(model, tok, p, 12, device, temperature=0.0, use_cache=True) for p in prompts]
    meta_rows = generate_batch_greedy_with_meta(model, tok, prompts, 12, device, batch_size=2)
    assert [m["text"] for m in meta_rows] == seq_texts

    torch.manual_seed(7)
    rollouts = generate_batch_rollouts(
        model, tok, prompts, 10, device, num_return_sequences=4, batch_size=8, temperature=0.9, top_k=20, top_p=0.95,
    )
    assert len(rollouts) == len(prompts) * 4
    for idx, r in enumerate(rollouts):
        assert r["prompt_idx"] == idx // 4 and r["rollout_idx"] == idx % 4
        if r["ended_with_eos"]:
            assert r["rollout_ids"][-1] == eos_id
            assert len(r["rollout_ids"]) == len(r["generated_ids"]) + 1
        else:
            assert r["rollout_ids"] == r["generated_ids"]
            assert len(r["generated_ids"]) == 10

    # Regression: uncached bucket fallback (prompt_len + max_new_tokens > context_length) when rows hit EOS at different steps.
    uncached_rollouts = generate_batch_rollouts(
        model, tok, prompts, 70, device, num_return_sequences=4, batch_size=8, temperature=1.0, top_k=20, top_p=0.95,
    )
    assert len(uncached_rollouts) == len(prompts) * 4

    adv = compute_group_advantages([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], group_size=4)
    assert len(adv) == 12
    assert adv[0] > 0 and adv[1] < 0 and abs(sum(adv[:4])) < 1e-6
    assert adv[4:] == [0.0] * 8
    print("  [PASSED] Batched KV-cached rollout sampling, greedy parity, and GRPO group normalization")


def test_hard_failure_mining_and_end_to_end_grpo():
    """Verify leakage-free mining of failed train problems and full GRPO update/checkpoint/resume loop."""
    tok = _make_byte_tokenizer()
    vocab_size = tok.get_vocab_size()
    cfg = GPTConfig(vocab_size=vocab_size, context_length=128, n_layer=2, n_head=4, n_embd=64)
    sft_model = GPT(cfg)
    container = _FakeContainer()
    run_handle = _FakeRunHandle("grpo_run_test")

    @contextmanager
    def fake_track_run(*args, **kwargs):
        yield run_handle

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        logs_dir = tmp / "logs"
        data_dir = tmp / "prepared_v3"
        rl_dir = tmp / "rl_hard"
        _, splits, _ = prepare_arithmetic_dataset(
            output_dir=data_dir, samples_per_op=100, seed=42, answer_format="scratchpad", include_multistep=True, sync_drive=False,
        )

        # Save an SFT checkpoint whose parent_run_id is the grandparent base run (must NOT leak into GRPO parent_run_id!).
        sft_ckpt_dir = tmp / "e57bf52b_sft"
        sft_ckpt_dir.mkdir(parents=True)
        sft_ckpt_path = sft_ckpt_dir / "latest_step_10000.pt"
        torch.save({
            "version": 1,
            "stage": "sft",
            "parent_run_id": "23a72f3a_grandparent_base",
            "parent_checkpoint_step": 26047,
            "model_config": asdict(cfg),
            "model": sft_model.state_dict(),
            "tokenizer_json": tok.to_str(),
            "step": 10000,
            "metrics": [{"step": 10000, "val_loss": 0.0022}],
        }, sft_ckpt_path)

        tok_dir = tmp / "tokenized"
        tok_dir.mkdir()
        tok.save(str(tok_dir / "tokenizer.json"))
        arr = np.arange(512, dtype=np.uint16) % vocab_size
        arr.tofile(tok_dir / "train.bin")
        arr.tofile(tok_dir / "val.bin")
        corpus_manifest = {
            "version": 1,
            "vocab_size": vocab_size,
            "tokenizer_sha256": sha256(tok_dir / "tokenizer.json"),
            "train": {"tokens": len(arr), "sha256": sha256(tok_dir / "train.bin")},
            "val": {"tokens": len(arr), "sha256": sha256(tok_dir / "val.bin")},
        }

        mined_dir, hard_items, hard_manifest = prepare_hard_rl_dataset(
            preset="grpo_pilot",
            sft_run_id="e57bf52b_sft",
            sft_checkpoint=sft_ckpt_path,
            data_dir=data_dir,
            rl_data_dir=rl_dir,
            samples_per_op=100,
            scan_limit=12,
            scan_batch_size=8,
            no_drive=True,
            device=torch.device("cpu"),
        )
        assert mined_dir == rl_dir and len(hard_items) >= 12
        assert hard_manifest["parent_run_id"] == "e57bf52b_sft" and hard_manifest["parent_checkpoint_step"] == 10000
        heldout_ids = {ex.canonical_id for ex in splits["val"]} | {ex.canonical_id for ex in splits["test"]}
        assert {ex.canonical_id for ex in hard_items}.isdisjoint(heldout_ids), "Hard RL dataset leaked val/test IDs!"
        reloaded, _ = load_rl_hard_dataset(rl_dir)
        assert len(reloaded) == len(hard_items)

        with (
            patch.dict("os.environ", {"DATABRICKS_HOST": "https://example.cloud.databricks.com", "DATABRICKS_TOKEN": "tok"}),
            patch.object(azure_blob, "container_client", return_value=container),
            patch.object(checkpoint_store, "container_client", return_value=container),
            patch("src.training.grpo.ensure_tokenized_corpus", return_value=corpus_manifest),
            patch("src.training.common.track_run", fake_track_run),
        ):
            res = grpo.run_grpo(
                preset="grpo_pilot",
                sft_run_id="e57bf52b_sft",
                sft_checkpoint=sft_ckpt_path,
                data_dir=data_dir,
                rl_data_dir=rl_dir,
                samples_per_op=100,
                steps=2,
                stop_after=1,
                prompts_per_step=2,
                group_size=4,
                mini_batch_size=4,
                eval_every=1,
                no_drive=True,
                precision_request="fp32",
                device=torch.device("cpu"),
                logs_dir=logs_dir,
                retention_tokenized_dir=tok_dir,
                checkpoints_dir=tmp / "cache",
            )

            assert res["status"] == "Stopped" and res["step"] == 1
            assert res["parent_run_id"] == "e57bf52b_sft" and res["parent_checkpoint_step"] == 10000
            assert run_handle.tags["stage"] == "rl" and run_handle.tags["algorithm"] == "grpo"
            assert run_handle.tags["parent_run_id"] == "e57bf52b_sft"
            assert run_handle.tags["reward_spec"] == "strict_binary_eos_work_answer_v1"
            assert any("val_loss" in m for _, m in run_handle.metrics)
            assert any("kl_div" in m for _, m in run_handle.metrics)
            assert len(CheckpointStore("minireasoner", "grpo_run_test")._versions()) == 1

            # Verify actual step 1 -> step 2 resume with mocked container and mixed-reward gradient signal
            toggle = {"idx": 0}

            def mixed_reward_verifier(raw_output, expected_answer, *, ended_with_eos=True):
                r = 1.0 if toggle["idx"] % 2 == 0 else 0.0
                toggle["idx"] += 1
                return {
                    "ended_with_eos": bool(ended_with_eos),
                    "valid_format": True,
                    "valid_work": bool(r),
                    "trace_ends_at_answer": bool(r),
                    "last_work_val": expected_answer if r else 0,
                    "extracted": expected_answer if r else 0,
                    "exact_match": bool(r),
                    "strict_pass": bool(r),
                    "reward": r,
                }

            with patch("src.training.grpo.score_strict_scratchpad_rollout", side_effect=mixed_reward_verifier):
                res_resume = grpo.run_grpo(
                    preset="grpo_pilot",
                    sft_run_id="e57bf52b_sft",
                    sft_checkpoint=sft_ckpt_path,
                    data_dir=data_dir,
                    rl_data_dir=rl_dir,
                    samples_per_op=100,
                    steps=2,
                    prompts_per_step=2,
                    group_size=4,
                    mini_batch_size=4,
                    eval_every=1,
                    resume=True,
                    mlflow_run_id="grpo_run_test",
                    no_drive=True,
                    precision_request="fp32",
                    device=torch.device("cpu"),
                    logs_dir=logs_dir,
                    retention_tokenized_dir=tok_dir,
                    checkpoints_dir=tmp / "cache",
                )
            assert res_resume["status"] == "Completed" and res_resume["step"] == 2
            assert res_resume["val_strict_pass"] > 0.0
            step2_metrics = next(m for s, m in run_handle.metrics if s == 2 and "frac_mixed_groups" in m)
            assert step2_metrics["frac_mixed_groups"] == 1.0
            resumed_versions = CheckpointStore("minireasoner", "grpo_run_test")._versions()
            assert len(resumed_versions) == 2
            latest_blob = container.download_blob(CheckpointStore("minireasoner", "grpo_run_test").files.prefix + max(resumed_versions, key=lambda v: v.step).name)
            import io
            buf = io.BytesIO()
            latest_blob.readinto(buf)
            buf.seek(0)
            resumed_state = torch.load(buf, map_location="cpu", weights_only=True)["model"]
            assert any(not torch.equal(resumed_state[k], v) for k, v in sft_model.state_dict().items()), "Expected mixed rewards to update weights"
    # Verify strict wording generalization + Phase 6 multi-model comparison & provenance helpers cleanly
    from src.core.benchmark import compare_arithmetic_results, compare_simplestories_results, evaluate_wording_generalization
    from src.infra.eval_setup import summarize_release_provenance

    with tempfile.TemporaryDirectory() as p6_dir:
        tmp_p6 = Path(p6_dir)
        _, p6_splits, p6_manifest = prepare_arithmetic_dataset(
            output_dir=tmp_p6 / "arith",
            samples_per_op=100,
            seed=42,
            answer_format="scratchpad",
            include_multistep=True,
            sync_drive=False,
        )
        wording = evaluate_wording_generalization(
            sft_model, tok, p6_splits["test"], device=torch.device("cpu"),
            answer_format="scratchpad", n_problems=4, max_new_tokens=6, use_kv_cache=True, strict_trace=True,
        )
        assert "canonical" in wording
        fake_rows_a = [
            {"op": "add", "category": "add_2d", "prompt": "Q1", "expected": 2, "completion": "Work: 1+1=2\nAnswer: 2", "exact_match": True, "strict_pass": True},
            {"op": "mul", "category": "mul_3d", "prompt": "Q2", "expected": 6, "completion": "bad", "exact_match": False, "strict_pass": False},
        ]
        fake_rows_b = [
            {"op": "add", "category": "add_2d", "prompt": "Q1", "expected": 2, "completion": "bad", "exact_match": False, "strict_pass": False},
            {"op": "mul", "category": "mul_3d", "prompt": "Q2", "expected": 6, "completion": "Work: 2*3=6\nAnswer: 6", "exact_match": True, "strict_pass": True},
        ]
        comp = compare_arithmetic_results(
            {
                "SFT": {"eval_loss": 0.01, "strict_pass": 0.5, "exact_match": 0.5, "valid_format": 1.0, "valid_work": 0.5, "op_stats": {"add": [1, 1], "mul": [0, 1]}, "rows": fake_rows_a},
                "RL": {"eval_loss": 0.02, "strict_pass": 0.5, "exact_match": 0.5, "valid_format": 1.0, "valid_work": 0.5, "op_stats": {"add": [0, 1], "mul": [1, 1]}, "rows": fake_rows_b},
            },
            baseline_name="SFT",
            max_examples_to_show=2,
        )
        assert comp["paired"]["RL"] == {"both_pass": 0, "fixed": 1, "broke": 1, "both_fail": 0, "net": 0, "op_deltas": {"add": (0, 1), "mul": (1, 0)}}
        compare_simplestories_results({"Base": {"test_ce": 1.15, "test_ppl": 3.16, "test_bpb": 0.55, "story_mc_acc": 87.5}})
        ckpt_file = tmp_p6 / "ckpt.pt"
        ckpt_file.write_bytes(b"checkpoint-bytes")
        prov = summarize_release_provenance(
            [{"label": "SFT", "run_id": "run_sft", "checkpoint_path": ckpt_file, "bundle": {"stage": "sft", "step": 10000, "checkpoint": {"parent_run_id": "base", "parent_checkpoint_step": 26047}}}],
            arith_dir=tmp_p6 / "arith",
            arith_manifest=p6_manifest,
        )
        assert prov[0]["sha256"] == sha256(ckpt_file)
    print("  [PASSED] Hard-failure mining, strict GRPO training loop, Azure naming & resume")


if __name__ == "__main__":
    print("Running Phase 5 GRPO RL test suite...")
    test_strict_binary_verifier_all_conditions()
    test_batched_rollouts_and_group_advantages()
    test_hard_failure_mining_and_end_to_end_grpo()
    print("All tests in test_rl.py passed successfully!")
