# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_every_two(x):
    # x: (..., head_dim)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # interleave (-x2, x1) -> shape (..., head_dim)
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, base=10000):
        """
        head_dim: dimensionality per head (must be even)
        """
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for rotary"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_seq, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_seq, head_dim]
        self.register_buffer("cos", emb.cos())    # [max_seq, head_dim]
        self.register_buffer("sin", emb.sin())    # [max_seq, head_dim]

    def apply_rotary(self, x):
        # x: [batch, seq, heads, head_dim]
        seq_len = x.shape[1]
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)  # [1, seq, 1, head_dim]
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        return x * cos + rotate_every_two(x) * sin

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        # x: [..., d]
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.scale

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff // 2, d_model, bias=False)  # second projection after gating
        # SiLU used as gating nonlinearity
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.w1(x)         # [B, T, d_ff]
        a, b = x.chunk(2, dim=-1)
        return self.w2(a * self.act(b))

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, rotary_emb: RotaryEmbedding):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)
        self.rotary = rotary_emb

    def forward(self, x, attn_mask=None):
        # x: [B, T, D]
        B, T, D = x.size()
        q = self.to_q(x).view(B, T, self.n_heads, self.head_dim)  # [B,T,H,HD]
        k = self.to_k(x).view(B, T, self.n_heads, self.head_dim)
        v = self.to_v(x).view(B, T, self.n_heads, self.head_dim)

        # apply rotary to q,k
        q = self.rotary.apply_rotary(q)
        k = self.rotary.apply_rotary(k)

        # compute scaled dot-product
        # scores: [B, H, T, T]
        scores = torch.einsum("bthd,bThd->bh t T".replace(" ", ""), q, k) / math.sqrt(self.head_dim)
        # Above compact fn broken on some interpreters, safer to do einsum plainly:
        # scores = torch.einsum("bthd,bThd->bh t T", q, k) / math.sqrt(self.head_dim)
        # but because einsum expects valid spec, use correct:
        scores = torch.einsum("bthd,bThd->bh t T", q, k) / math.sqrt(self.head_dim)  # this may error on old einsum; fallback below if needed

        # Build causal mask if not provided
        if attn_mask is None:
            # causal_mask shape: [1, 1, T, T]
            causal = torch.tril(torch.ones((T, T), device=x.device, dtype=torch.bool))
            causal = causal.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~causal, float("-inf"))
        else:
            # attn_mask expected shape [B, T] (1 for tokens to keep)
            # convert to [B,1,1,T] then broadcast
            am = attn_mask.view(B, 1, 1, T)
            scores = scores.masked_fill(am == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # [B, H, T, T]
        out = torch.einsum("bh t T,bThd->bthd", weights, v)  # [B,T,H,HD]
        out = out.contiguous().view(B, T, D)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, rotary_emb):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, rotary_emb)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.attn_norm(x), attn_mask=attn_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x

class TinyDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048, max_seq=2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        # rotary operates per-head; give it head_dim
        head_dim = d_model // n_heads
        self.rotary = RotaryEmbedding(head_dim, max_seq_len=max_seq)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, self.rotary) for _ in range(n_layers)])
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = self.embed.weight

        # optionally tie embeddings:
        # self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, attention_mask=None):
        # input_ids: [B, T]
        x = self.embed(input_ids)  # [B, T, D]
        for blk in self.blocks:
            x = blk(x, attn_mask=attention_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, V]
        return logits
