import torch
import torch.nn as nn
import torch.nn.functional as F
from llm.mmhsa import LLMAttention


class FeedForward(nn.Module):
    # Multi-layer perceptron / feed-forward network
    def __init__(self, dim, hidden_dim, activation=F.gelu, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 n_heads: int,
                 mlp_ratio: float = 4.0,
                 rotary_fraction: float = 1.0,
                 max_seq_len: int = 2048,
                 dropout: float = 0.0):
        super().__init__()
        # Pre-LN: normalization applied BEFORE attention/MLP
        self.ln1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = LLMAttention(dim, n_heads, max_seq_len=max_seq_len, rotary_fraction=rotary_fraction, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim, eps=1e-5)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim, hidden_dim, activation=F.gelu, dropout=dropout)

    def forward(self, x, attn_mask=None, is_causal=True, past_kv=None):
        # Attention + Residual
        residual = x
        x_ln = self.ln1(x)
        attn_out, present_kv = self.attn(x_ln, attn_mask=attn_mask, is_causal=is_causal, past_kv=past_kv)
        x = residual + attn_out
        
        # MLP + Residual
        x = x + self.mlp(self.ln2(x))
        return x, present_kv