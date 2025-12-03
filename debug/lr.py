attn_params = []
mlp_params = []
other_params = []

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if "attn." in name:
        attn_params.append(p)
    elif "mlp." in name:
        mlp_params.append(p)
    else:
        other_params.append(p)

optimizer = torch.optim.AdamW(
    [
        {"params": attn_params, "lr": lr},           # full LR for attention
        {"params": mlp_params, "lr": lr * 0.7},      # slightly smaller LR for MLP
        {"params": other_params, "lr": lr},          # embeddings, norms, lm_head...
    ],
    weight_decay=1e-2,
)
