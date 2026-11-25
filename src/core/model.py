from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    context_length: int = 1024
    n_layer: int = 10
    n_head: int = 8
    n_embd: int = 512
    rope_theta: float = 10000.0
    pos_encoding: str = "rope"

    def __post_init__(self):
        if min(self.vocab_size, self.context_length, self.n_layer, self.n_head, self.n_embd) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        head_dim = self.n_embd // self.n_head
        if self.pos_encoding == "rope":
            if head_dim % 2 != 0:
                raise ValueError("head_dim must be even for RoPE")
            if self.rope_theta <= 0:
                raise ValueError("rope_theta must be positive")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Keep standard FP32 computation for RoPE rotations to avoid BF16 precision loss
    x_fp32 = x.float()
    out = (x_fp32 * cos) + (rotate_half(x_fp32) * sin)
    return out.to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "head_dim must be even for RoPE"
        self.dim = dim
        self.theta = float(theta)
        self.max_seq_len = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int, device: torch.device | None = None):
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]  # (1, 1, seq_len, dim)
        sin = emb.sin()[None, None, :, :]  # (1, 1, seq_len, dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.max_seq_len = seq_len

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(2)
        if seq_len > self.max_seq_len:
            self._build_cache(max(seq_len, self.max_seq_len * 2), device=q.device)
        elif self.cos.device != q.device:
            self._build_cache(self.max_seq_len, device=q.device)
        cos = self.cos[:, :, :seq_len, :]
        sin = self.sin[:, :, :seq_len, :]
        return apply_rotary_pos_emb(q, cos, sin), apply_rotary_pos_emb(k, cos, sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.rotary = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.context_length,
            theta=config.rope_theta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, n_embd = x.shape
        qkv = self.c_attn(x).reshape(batch, length, 3, self.n_head, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind()
        q, k = self.rotary(query, key)
        attended = F.scaled_dot_product_attention(q, k, value, is_causal=True)
        return self.c_proj(attended.transpose(1, 2).reshape(batch, length, n_embd))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            "ln_f": nn.LayerNorm(config.n_embd),
        })
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for block in self.transformer.h:
            nn.init.normal_(block.attn.c_proj.weight, std=residual_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=residual_std)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        length = input_ids.size(1)
        if length > self.config.context_length:
            raise ValueError("Input exceeds the configured context length")
        hidden = self.transformer.wte(input_ids)
        for block in self.transformer.h:
            hidden = block(hidden)
        return self.lm_head(self.transformer.ln_f(hidden))
