import torch
import pynvml

def init_gpu_monitor():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return handle

def get_gpu_stats(handle):
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "gpu_util": util.gpu,
        "mem_used": mem_info.used / 1024 ** 2,
        "mem_total": mem_info.total / 1024 ** 2
    }

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
