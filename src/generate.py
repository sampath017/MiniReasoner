from pathlib import Path

import torch
from tokenizers import Tokenizer

from .model import GPT, GPTConfig
from .tokenizer import EOS
from .train import load_checkpoint


@torch.inference_mode()
def complete(
    checkpoint_path: Path,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    checkpoint = load_checkpoint(checkpoint_path)
    tokenizer = Tokenizer.from_str(checkpoint["tokenizer_json"])
    config = GPTConfig(**checkpoint["model_config"])
    model = GPT(config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not ids or len(ids) > config.context_length:
        raise ValueError(f"Prompt must encode to between 1 and {config.context_length} tokens")
    generated = []
    eos_id = tokenizer.token_to_id(EOS)
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-config.context_length :]], device=device)
        logits = model(x)[0, -1]
        if temperature <= 0.0:
            next_id = logits.argmax().item()
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
            next_id = torch.multinomial(probs, num_samples=1).item()
        if next_id == eos_id:
            break
        ids.append(next_id)
        generated.append(next_id)
    return tokenizer.decode(generated)
