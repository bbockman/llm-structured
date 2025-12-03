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
    def __init__(self, head_dim, max_seq_len, base=10000):
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
        B, T, D = x.size()

        # Projections
        q = self.to_q(x).view(B, T, self.n_heads, self.head_dim)
        k = self.to_k(x).view(B, T, self.n_heads, self.head_dim)
        v = self.to_v(x).view(B, T, self.n_heads, self.head_dim)

        # Apply rotary embeddings before SDPA
        q = self.rotary.apply_rotary(q)
        k = self.rotary.apply_rotary(k)

        # SDPA expects: (B, num_heads, T, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Prepare attention mask: (B, 1, 1, T)
        if attn_mask is not None:
            attn_mask = attn_mask.view(B, 1, 1, T).bool()

        # SDPA handles causal masking internally when is_causal=True
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=True
        )

        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.to_out(out)

from utils import rms


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, rotary_emb,
        debug_rms=False,
        debug_branch=False,
        use_rezero=True,     
        branch_scale=1.0,    
        alpha_init=0.0,      
    ):
        super().__init__()

        self.attn_norm = RMSNorm(d_model)
        self.mlp_norm  = RMSNorm(d_model)

        self.attn = CausalSelfAttention(d_model, n_heads, rotary_emb)
        self.mlp  = SwiGLU(d_model, d_ff)

        self.use_rezero   = use_rezero
        self.branch_scale = branch_scale

        if use_rezero:
            self.alpha_attn = nn.Parameter(torch.full((1,), alpha_init))
            self.alpha_mlp  = nn.Parameter(torch.full((1,), alpha_init))
        else:
            self.register_buffer("alpha_attn", torch.tensor(1.0))
            self.register_buffer("alpha_mlp",  torch.tensor(1.0))

        self.debug_rms = debug_rms
        self.debug_branch = debug_branch 
        self._debug_done = False
         
        self.layer_id = "?"

    def forward(self, x, attn_mask=None):
        rm_log = self.debug_rms and (not self._debug_done)
        br_log = self.debug_branch and (not self._debug_done)

        if rm_log:
            print(f"[RMS] layer_in {self.layer_id}: {rms(x):.4f}")

        attn_out = self.attn(self.attn_norm(x), attn_mask=attn_mask)
        mlp_out  = self.mlp(self.mlp_norm(x))

        attn_scale = self.branch_scale * self.alpha_attn
        mlp_scale  = self.branch_scale * self.alpha_mlp

        attn_delta = attn_scale * attn_out
        mlp_delta  = mlp_scale  * mlp_out

        x_out = x + attn_delta + mlp_delta

        if rm_log:
            print(f"[RMS] layer_out {self.layer_id}: {rms(x_out):.4f}")
            self._debug_done = True and not br_log

        if br_log:
            with torch.no_grad():
                # (B, T, D) -> (B*T, D) -> mean L2 per token
                attn_delta = attn_delta.flatten(0, 1).norm(dim=-1).mean().item()
                mlp_delta  = mlp_delta.flatten(0, 1).norm(dim=-1).mean().item()
                ratio = mlp_delta / (attn_delta + 1e-8)

            print(
                f"[Δ] layer {self.layer_id}: "
                f"attn={attn_delta:.4f}, mlp={mlp_delta:.4f}, ratio={ratio:.2f}"
            )
            # self._debug_branches_done = True

        return x_out


from torch.utils.checkpoint import checkpoint

class TinyDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=12, d_ff=3072, max_seq=1024):
        super().__init__()
        assert d_model % n_heads == 0
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        head_dim = d_model // n_heads
        self.rotary = RotaryEmbedding(head_dim, max_seq_len=max_seq)

        # Depth-aware base scale: each of the 2 branches per layer
        # has effective initial scale ≈ branch_scale (if alpha_init=1)
        branch_scale = 1.0 / math.sqrt(2.0 * n_layers)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads,
                d_ff,
                self.rotary,
                branch_scale=branch_scale,
                alpha_init=1.0,  # effective init ≈ 1/sqrt(2 * n_layers)
            )
            for _ in range(n_layers)
        ])
        for i, block in enumerate(self.blocks):
            block.layer_id = i  # for debugging

        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scale"""
        std = 0.02
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)
        
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: [B, T]
        out = self.embed(input_ids)  # [B, T, D]
        for blk in self.blocks:
            out = checkpoint(blk, out, attention_mask, use_reentrant=False)
        x = self.ln_f(out)
        logits = self.lm_head(x)  # [B, T, V]
        return logits
    
    def compute_loss(self, input_ids, pad_id=None, attention_mask=None, labels=None):
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
            ignore_index=pad_id  
        )
        
        return loss
    
    @property
    def max_seq_len(self):
        return self.max_seq_len

def get_current_model(max_seq_len=1024, vocab_size=8192, dmodel=1024, n_layers=12, n_heads=16, d_ff=4096):
    return TinyDecoder(
        vocab_size=vocab_size,
        d_model=dmodel,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq=max_seq_len
    )