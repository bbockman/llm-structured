import torch
import torch.nn as nn
import torch.nn.functional as F
from llm.blocks import TransformerBlock
from typing import Optional, Tuple

class GPT(nn.Module):
    """
    Minimal GPT-style model:
      - token embedding
      - N TransformerBlocks (using RoPE for position encoding)
      - final LayerNorm and linear head (with tied weights)
    """
    def __init__(self,
                 vocab_size: int,
                 n_layers: int,
                 dim: int,
                 n_heads: int,
                 max_seq_len: int = 2048,
                 rotary_fraction: float = 1.0,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, n_heads=n_heads, mlp_ratio=mlp_ratio,
                             rotary_fraction=rotary_fraction, max_seq_len=max_seq_len, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(dim, eps=1e-5)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # weight tying (saves memory and is standard practice)
        self.head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.LongTensor, attn_mask: Optional[torch.Tensor] = None, past_key_values: Optional[Tuple] = None):
        B, T = input_ids.shape
        x = self.token_emb(input_ids)  # (B, T, dim)

        presents = []
        for i, block in enumerate(self.blocks):
            past_kv = None
            if past_key_values is not None:
                past_kv = past_key_values[i]
            x, present_kv = block(x, attn_mask=attn_mask, is_causal=True, past_kv=past_kv)
            presents.append(present_kv)

        x = self.final_ln(x)
        logits = self.head(x)
        return logits, presents

    @torch.no_grad()
    def generate(self, input_ids: torch.LongTensor, max_new_tokens: int = 50, eos_token_id: Optional[int] = None, temperature: float = 1.0):
        # Simple autoregressive loop using cached KV pairs
        # ... (implementation as provided in original document)
        device = input_ids.device
        B = input_ids.shape[0]
        past = [None] * len(self.blocks)

        out = input_ids
        for _ in range(max_new_tokens):
            logits, presents = self.forward(out[:, -1:].long(), past_key_values=past)
            # logits shape (B, 1, V)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B,1)
            out = torch.cat([out, next_token], dim=1)

            # update cache
            past = presents

            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break
        return out