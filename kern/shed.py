import math

def cosine_with_warmup(step, warmup_steps, total_steps, base_lr, min_lr):
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))

    if step >= total_steps:
        return min_lr

    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine
