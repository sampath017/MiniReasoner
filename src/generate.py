from pathlib import Path

import torch
from tokenizers import Tokenizer

from .model import GPT, GPTConfig
from .tokenizer import EOS
from .train import load_checkpoint


@torch.inference_mode()
def complete(checkpoint_path: Path, prompt: str, max_new_tokens: int, device: torch.device) -> str:
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
        next_id = model(x)[0, -1].argmax().item()
        if next_id == eos_id:
            break
        ids.append(next_id)
        generated.append(next_id)
    return tokenizer.decode(generated)
