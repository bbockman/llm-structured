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