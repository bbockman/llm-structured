def cosine_lr(t, T, eta_max, eta_min):
    if T <= 1:
        return eta_min  # or eta_max; define your convention
    cos_inner = math.pi * t / (T - 1)
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(cos_inner))

T = total_updates  # e.g. num_epochs * steps_per_epoch

for t in range(T):  # t is your "global step index"
    lr = cosine_lr(t, T, eta_max, eta_min)
    for group in optimizer.param_groups:
        group["lr"] = lr
    # run one update (forward/backward/step) here...

import math
from torch.optim.lr_scheduler import LambdaLR

def min_max_min_cosine_lambda(current_step, total_steps, min_lr, max_lr):
    # Progress from 0 to 1
    progress = current_step / total_steps
    # Cosine curve: goes from min -> max -> min
    # f(progress) = min_lr + (max_lr - min_lr) * (1 - cos(pi * progress)) / 2
    return min_lr + (max_lr - min_lr) * (1 - math.cos(math.pi * progress)) / 2

# Usage:
optimizer = ...  # your optimizer
total_steps = ...  # total number of steps
min_lr = 1e-5
max_lr = 1e-3

scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda step: min_max_min_cosine_lambda(step, total_steps, min_lr, max_lr) / max_lr
)