import torch
from typing import Tuple

# ---- Rotary helpers ----
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: (..., dim)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q,k shape: (B, n_heads, T, head_dim)
    # cos, sin shape: (T, 1, rotary_dim) or (T, rotary_dim)
    # Split last dim in two for rotate_half trick; implementation uses elementwise
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

def build_rotary_cos_sin(max_seq: int, head_dim: int, device: torch.device, dtype: torch.dtype):
    # standard rotary frequency schedule
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(max_seq, device=device, dtype=dtype)
    freqs = torch.einsum("t,d->t d", t, inv_freq)  # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    cos = emb.cos().unsqueeze(1)  # (T, 1, head_dim)
    sin = emb.sin().unsqueeze(1)  # (T, 1, head_dim)
    return cos, sin