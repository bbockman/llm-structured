import pynvml
import torch

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

def rms(x):
    with torch.no_grad():
        return x.pow(2).mean().sqrt().item()
    
def get_tensor_memory():
    import gc
    import torch
    from collections import defaultdict

    summary = defaultdict(lambda: {"count": 0, "total_mb": 0.0})
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                key = (str(obj.dtype), tuple(obj.shape), str(obj.device))
                size_mb = obj.element_size() * obj.nelement() / 1024**2
                summary[key]["count"] += 1
                summary[key]["total_mb"] += size_mb
        except Exception:
            pass

    print("\n--- Tensor Memory Summary ---")
    for key, val in summary.items():
        dtype, shape, device = key
        print(f"{val['count']:3d}x {dtype} {shape} on {device}: {val['total_mb']:.2f} MB")
    print("--- End of Summary ---\n")

def print_param_ids(model):
    print("\n--- Model Parameter IDs ---")
    for name, param in model.named_parameters():
        print(f"{name}: id={id(param.data)}, shape={tuple(param.shape)}, device={param.device}")
    print("--- End ---\n")