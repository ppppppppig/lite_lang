import torch


def get_total_gpu_memory() -> float:

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    total_memory = 0.0
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    total_memory += device.total_memory

    return total_memory / (1024**3)


def get_available_gpu_memory() -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    free_memory = 0.0
    reserved = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    free_memory = total - reserved

    return free_memory / (1024**3)
