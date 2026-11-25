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


@torch.inference_mode()
def generate_text(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """Decode continuation text from an already-loaded model and tokenizer."""
    was_training = model.training
    model.eval()
    try:
        context_length = model.config.context_length
        ids = prepare_prompt_tokens(tokenizer, prompt, context_length)
        generated = []
        eos_id = tokenizer.token_to_id(EOS)
        for _ in range(max_new_tokens):
            x = torch.tensor([ids[-context_length:]], device=device)
            logits = model(x)[0, -1].float()
            if temperature <= 0.0:
                next_id = int(logits.argmax().item())
            else:
                scaled_logits = logits / temperature
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    scaled_logits[indices_to_remove] = -float("Inf")
                probs = torch.softmax(scaled_logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            if next_id == eos_id:
                break
            ids.append(next_id)
            generated.append(next_id)
        return tokenizer.decode(generated)
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
