from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from llm.rope import apply_rotary_pos_emb, build_rotary_cos_sin

# ---- LLM-grade attention module ----
class LLMAttention(nn.Module):
    """
    Multi-head self-attention with:
      - fused QKV (single linear)
      - RoPE rotary embeddings (applied to q,k)
      - uses torch.nn.functional.scaled_dot_product_attention (benefits from FlashAttention)
      - supports past_key_value caching for autoregressive inference
    """
    def __init__(self,
                 dim: int,
                 n_heads: int,
                 max_seq_len: int = 2048,
                 rotary_fraction: float = 1.0, # fraction of head_dim to apply RoPE to (1.0 = full)
                 dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        rotary_dim = int(self.head_dim * rotary_fraction)
        if rotary_dim % 2 == 1:
            rotary_dim -= 1
        self.rotary_dim = rotary_dim

        self.register_buffer("_cos", torch.empty(0), persistent=False)
        self.register_buffer("_sin", torch.empty(0), persistent=False)

    def _ensure_rotary(self, x: torch.Tensor):
        if self._cos.numel() == 0 or self._cos.shape[0] < self.max_seq_len or self._cos.device != x.device:
            cos, sin = build_rotary_cos_sin(self.max_seq_len, self.rotary_dim, device=x.device, dtype=x.dtype)
            self._cos = cos  # (T, 1, rotary_dim)
            self._sin = sin

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                is_causal: bool = True,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        def reshape_heads(z):
            return z.view(B, T, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # RoPE application
        if self.rotary_dim > 0:
            self._ensure_rotary(q)
            q_rot_part = q[..., :self.rotary_dim]
            k_rot_part = k[..., :self.rotary_dim]
            
            # Slice and prepare cos/sin for current sequence length
            cos = self._cos[:T].squeeze(1) # (T, rotary_dim)
            sin = self._sin[:T].squeeze(1)

            q_rotated, k_rotated = apply_rotary_pos_emb(q_rot_part, k_rot_part, cos, sin)
            q = torch.cat([q_rotated, q[..., self.rotary_dim:]], dim=-1)
            k = torch.cat([k_rotated, k[..., self.rotary_dim:]], dim=-1)

        # Handle KV cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v)

        # Use scaled_dot_product_attention for optimal performance (FlashAttention dispatch)
        q_flat = q.reshape(B * self.n_heads, -1, self.head_dim)
        k_flat = k.reshape(B * self.n_heads, -1, self.head_dim)
        v_flat = v.reshape(B * self.n_heads, -1, self.head_dim)
        
        out_flat = F.scaled_dot_product_attention(q_flat, k_flat, v_flat,
                                                  attn_mask=attn_mask,
                                                  dropout_p=self.dropout if self.training else 0.0,
                                                  is_causal=is_causal)

        out = out_flat.view(B, self.n_heads, T, self.head_dim).transpose(1, 2).reshape(B, T, C)
        out = self.out(out)
        return out, present_kv