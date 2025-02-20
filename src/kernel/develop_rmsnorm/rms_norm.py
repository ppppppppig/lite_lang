import torch
import triton.language as tl
import triton
from triton.runtime import driver

device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
NUM_SM = props.multi_processor_count

def torch_rmsnorm(x, gamma, variance_epilson):
    x = x.to(torch.float32)
    gamma = gamma.to(torch.float32)
    variance = torch.sum(x * x, dim=-1, keepdim=True) / x.size(-1)
    middle = x / torch.sqrt(variance + variance_epilson)
    res = middle * gamma
    print(f"11: {res}")
    return res

@triton.jit
def rmsnorm_kernel(x_ptr, gamma_ptr, out_ptr, variance_epsilon,
                   x_stride_0, x_stride_1,
                   gamma_stride_0,
                   x_size_x, x_size_y,
                   gamma_size_x,
                   BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pros = tl.num_programs(0)
    
    for i in range(pid, x_size_x, num_pros):
        
        x = tl.load(x_ptr +  i * x_stride_0 + tl.arange(0, BLOCK) * x_stride_1, mask=tl.arange(0, BLOCK) < x_size_y, other=0).to(tl.float32)
        
        gamma = tl.load(gamma_ptr + tl.arange(0, BLOCK) * gamma_stride_0, mask=tl.arange(0, BLOCK) < gamma_size_x, other=0).to(tl.float32)
        sum = tl.sum(x * x) / x_size_y
        middle = x / tl.sqrt(sum + variance_epsilon)
        out = middle * gamma
        out = out.to(x_ptr.dtype.element_ty)
        tl.static_print(out.dtype)
        tl.store( out_ptr + i * x_stride_0 + tl.arange(0, BLOCK) * x_stride_1,out,  mask=tl.arange(0, BLOCK) < x_size_y)


def next_pow_of_2(n: int):
    if n < 0:
        return 1
    elif (n and (n - 1)) == 0:
        return n
    else:
        return 1 << (n - 1).bit_length()

def rmsnorm(x, gamma, variance_epsilon):
    assert x.shape[1] == gamma.shape[0], \
    f"gamma shape 0: {gamma.shape[0]} not match x shape 1: {x.shape[1]}"
    
    assert x.ndim == 2, "shapesize shoule equal to 2"
    
    block = next_pow_of_2(x.size(-1))
    out = torch.empty_like(x)
    
    
    grid = NUM_SM
    
    rmsnorm_kernel[grid,](
        x, gamma, out, variance_epsilon,
        x.stride(0), x.stride(1),
        gamma.stride(0),
        x.shape[0], x.shape[1],
        gamma.shape[0],
        num_stages=3, num_warps=4,
        BLOCK=block
    )
    print(f"out: {out}")
    return out


def test_rmsnorm(M, N, variance_epsilon=1e-5):
    x = torch.normal(size=(M, N), mean=0.3, std=0.5, device="cuda", dtype=torch.float32)
    gamma = torch.normal(size=(N,), mean=0.3, std=0.5, device="cuda", dtype=torch.float32)
    
    res1 = torch_rmsnorm(x, gamma, variance_epsilon)
    
    res2 = rmsnorm(x, gamma, variance_epsilon)
    
    print(f"max diff: {torch.max(res1 - res2)}")
    
    assert torch.allclose(res1, res2, atol=1e-3), "diff large"
    
test_rmsnorm(100, 2048)