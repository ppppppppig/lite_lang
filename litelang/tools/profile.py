import torch
import time

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

def measure_function_time(func):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start_time = time.perf_counter()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行原函数
        torch.cuda.synchronize()
        end_time = time.perf_counter()  # 记录结束时间
        print(f"函数 {func.__name__} 执行耗时: {end_time - start_time:.8f} 秒")
        return result
    return wrapper
