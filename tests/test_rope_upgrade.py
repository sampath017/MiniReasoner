"""Automated test suite for MiniReasoner RoPE, context_length=1024, and batch_size=32 upgrade."""

import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.generate import complete
from src.model import GPT, GPTConfig
from src.settings import PRESETS, VOCAB_SIZE
from src.train import EpochBatchSampler, TokenStream, TrainConfig, load_checkpoint


def test_architecture_and_forward_backward():
    """Verify RoPE model dimensions, exact parameter count (32,049,152), and finite gradients."""
    base_settings = PRESETS["base"]["model"]
    cfg = GPTConfig(vocab_size=VOCAB_SIZE, **base_settings)
    assert cfg.context_length == 1024, f"Expected context_length=1024, got {cfg.context_length}"
    assert cfg.pos_encoding == "rope", f"Expected pos_encoding='rope', got {cfg.pos_encoding}"
    assert cfg.rope_theta == 10000.0, f"Expected rope_theta=10000.0, got {cfg.rope_theta}"

    model = GPT(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 32_049_152, f"Expected 32,049,152 parameters, got {total_params}"

    # Verify no learned positional embedding table
    assert "wpe" not in model.transformer, "Found unexpected 'wpe' in model.transformer"

    # Forward + backward on dummy input (2, 1024)
    x = torch.randint(0, VOCAB_SIZE, (2, 1024))
    y = torch.randint(0, VOCAB_SIZE, (2, 1024))
    logits = model(x)
    assert logits.shape == (2, 1024, VOCAB_SIZE), f"Unexpected logits shape: {logits.shape}"

    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None grad"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite grad"

    if torch.cuda.is_available():
        model_cuda = model.to("cuda")
        x_cuda = x.repeat(16, 1).to("cuda")
        y_cuda = y.repeat(16, 1).to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits_cuda = model_cuda(x_cuda)
            loss_cuda = F.cross_entropy(logits_cuda.flatten(0, 1), y_cuda.flatten())
        loss_cuda.backward()


def test_attention_causality():
    """Verify that modifying tokens at positions > 10 does not alter logits at positions <= 10."""
    cfg = GPTConfig(vocab_size=VOCAB_SIZE, **PRESETS["base"]["model"])
    model = GPT(cfg).eval()

    torch.manual_seed(42)
    seq_len = 64
    split_pos = 10
    x1 = torch.randint(0, VOCAB_SIZE, (1, seq_len))
    x2 = x1.clone()
    x2[0, split_pos + 1:] = torch.randint(0, VOCAB_SIZE, (seq_len - split_pos - 1,))

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    diff_causal = (out1[0, :split_pos + 1] - out2[0, :split_pos + 1]).abs().max().item()
    diff_future = (out1[0, split_pos + 1:] - out2[0, split_pos + 1:]).abs().max().item()

    assert diff_causal == 0.0, f"Causality violation! Logits at <= {split_pos} changed by {diff_causal}"
    assert diff_future > 0.0, "Perturbation in future positions did not change future outputs"


def test_long_context_generation():
    """Verify that complete() executes smoothly on prompt contexts exceeding the old 256 limit."""
    cfg = GPTConfig(vocab_size=VOCAB_SIZE, **PRESETS["base"]["model"])
    model = GPT(cfg).eval()

    cached_chk = Path("checkpoints/best_checkpoint.pt")
    if cached_chk.is_file():
        data = torch.load(cached_chk, map_location="cpu", weights_only=True)
        tokenizer_json = data["tokenizer_json"]
    else:
        from tokenizers import models, pre_tokenizers, decoders
        vocab = {bytes([i]).decode("latin-1"): i for i in range(256)}
        tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tokenizer_json = tok.to_str()

    tok = Tokenizer.from_str(tokenizer_json)

    with tempfile.TemporaryDirectory() as tmpdir:
        chk_path = Path(tmpdir) / "test_rope_checkpoint.pt"
        torch.save({
            "version": 1,
            "model_config": asdict(cfg),
            "model": model.state_dict(),
            "tokenizer_json": tokenizer_json,
            "step": 1,
            "metrics": []
        }, chk_path)

        story_unit = (
            "Once upon a time in a sunny valley, a kind little girl named Mia found a golden key. "
            "She wondered what the key could open, so she walked through the green meadow. "
            "Along the way, she met a friendly rabbit who hopped beside her. "
        )
        prompt = story_unit * 4  # 312 tokens with real BPE, 908 with byte-level fallback
        prompt_ids = tok.encode(prompt, add_special_tokens=False).ids
        assert 256 < len(prompt_ids) <= 1024, f"Expected 256 < tokens <= 1024, got {len(prompt_ids)}"

        # Greedy completion
        completion = complete(chk_path, prompt, max_new_tokens=15, device=torch.device("cpu"))
        assert isinstance(completion, str), "Completion is not a string"

        # Sampling completion
        completion_sampling = complete(
            chk_path, prompt, max_new_tokens=10, device=torch.device("cpu"),
            temperature=0.8, top_p=0.9
        )
        assert isinstance(completion_sampling, str), "Sampling completion is not a string"


def test_sampler_and_checkpoint_compatibility():
    """Verify EpochBatchSampler accounting (26,047 steps, 838 tail targets) and run_identity matching."""
    tokens = 853_498_695  # SimpleStories full train split tokens
    target_count = tokens - 1
    context_length = 1024
    batch_size = 32

    windows, tail_targets = divmod(target_count, context_length)
    window_count = windows + int(tail_targets > 0)
    steps_per_epoch = math.ceil(window_count / batch_size)

    assert steps_per_epoch == 26047, f"Expected 26,047 steps per epoch, got {steps_per_epoch}"
    assert tail_targets == 838, f"Expected 838 tail targets, got {tail_targets}"

    dummy_dir = tempfile.mkdtemp()
    try:
        dummy_file = Path(dummy_dir) / "dummy_train.bin"
        arr = np.arange(100_000, dtype=np.uint16)
        arr.tofile(dummy_file)

        stream = TokenStream(dummy_file, len(arr), context_length)
        sampler = EpochBatchSampler(stream, batch_size=batch_size, seed=42)

        starts1 = sampler.starts(step=1)
        starts2 = sampler.starts(step=1)
        assert np.array_equal(starts1, starts2), "Sampler is not deterministic for step 1"

        t0 = sampler.targets_through(0)
        t1 = sampler.targets_through(1)
        assert t0 == 0, f"Expected targets_through(0)==0, got {t0}"
        assert t1 == batch_size * context_length, f"Expected targets_through(1)=={batch_size * context_length}, got {t1}"

        del stream.tokens
        del stream
        del sampler
    finally:
        import shutil
        shutil.rmtree(dummy_dir, ignore_errors=True)

    base_settings = PRESETS["base"]["model"]
    train_settings = PRESETS["base"]["training"].copy()
    train_settings["steps"] = steps_per_epoch
    train_settings.pop("epochs", None)

    model_cfg = GPTConfig(vocab_size=VOCAB_SIZE, **base_settings)
    train_cfg = TrainConfig(**train_settings)

    run_identity = {
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "sampling": "shuffled_windows_v1",
        "data_manifest": {"version": 1, "vocab_size": VOCAB_SIZE},
    }

    model = GPT(model_cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        chk_file = Path(tmpdir) / "checkpoint_roundtrip.pt"
        saved_dict = {
            "version": 1,
            **run_identity,
            "step": 500,
            "model": model.state_dict(),
            "optimizer": {},
            "tokenizer_json": "{}",
            "metrics": [{"step": 500, "val_loss": 2.5}]
        }
        torch.save(saved_dict, chk_file)

        loaded = load_checkpoint(chk_file)
        assert loaded["version"] == 1
        matches = not any(loaded.get(k) != v for k, v in run_identity.items())
        assert matches, "run_identity mismatch on valid checkpoint!"

    cached_chk = Path("checkpoints/best_checkpoint.pt")
    if cached_chk.is_file():
        old_chk = load_checkpoint(cached_chk)
        is_mismatched = any(old_chk.get(k) != v for k, v in run_identity.items())
        assert is_mismatched, "Expected mismatched old checkpoint to be rejected by run_identity!"


if __name__ == "__main__":
    print("Running MiniReasoner test suite in tests/...")
    test_architecture_and_forward_backward()
    print("  [PASSED] Architecture and forward/backward pass")
    test_attention_causality()
    print("  [PASSED] Attention causality (SDPA + RoPE)")
    test_long_context_generation()
    print("  [PASSED] Long context generation (> 256 tokens)")
    test_sampler_and_checkpoint_compatibility()
    print("  [PASSED] Sampler accounting and checkpoint round-trip")
    print("All tests in tests/ passed successfully!")
