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

from flash_attn.modules.mha import FlashMHA
import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, rotary_emb):
        super().__init__()
        self.rotary = rotary_emb

        self.attn = FlashMHA(
            embed_dim=d_model,
            num_heads=n_heads,
            causal=True,
            dropout=0.0,
            bias=False,
            use_rotary_emb=True     # FlashMHA will call rotary_emb_fn
        )

    def forward(self, x, attn_mask=None):
        """
        x: (B, T, D)
        attn_mask: (B, T) bool mask, True=pad/masked, False=valid
        """

        # FlashAttention will call rotary_emb_fn(q, k)
        return self.attn(
            x,
            key_padding_mask=attn_mask,
            rotary_emb_fn=self.rotary.apply_rotary_flash
        )



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

from torch.utils.checkpoint import checkpoint
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
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scale"""
        std = 0.02
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)
        # Don't init lm_head.weight - it's tied to embed!
        
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: [B, T]
        out = self.embed(input_ids)  # [B, T, D]
        for blk in self.blocks:
            out = checkpoint(blk, out, attention_mask, use_reentrant=False)
        x = self.ln_f(out)
        logits = self.lm_head(x)  # [B, T, V]
        return logits
    
    def compute_loss(self, input_ids, attention_mask=None, labels=None):
        """
        Compute loss more efficiently by computing logits and loss together.
        This can save memory compared to materializing full logits then computing loss.
        """
        out = self.embed(input_ids)
        for blk in self.blocks:
            out = checkpoint(blk, out, attention_mask, use_reentrant=False)
        x = self.ln_f(out)
        
        if labels is None:
            # Inference mode - return full logits
            return self.lm_head(x)
        
        # Training mode - compute loss directly
        # Shift for next-token prediction
        shift_hidden = x[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Compute logits only for non-padding positions
        shift_logits = self.lm_head(shift_hidden)
        
        # Flatten for loss computation
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100  # Standard ignore index
        )
        
        return loss
