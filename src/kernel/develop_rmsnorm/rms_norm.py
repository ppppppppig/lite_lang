import torch
import triton.language as tl
import triton

def torch_rmsnorm(x, gamma, variance_epilson):
    x = x.to(torch.float32)
    gamma = gamma.to(torch.float32)
    variance = torch.sum(x * x, dim=-1) / x.size(-1)
    middle = x / torch.sqrt(variance + variance_epilson)
    res = torch.matmul(middle, gamma)

@triton.jit
def rmsnorm_kernel(x_ptr, gamma_ptr, out_ptr, variance_epsilon,
                   x_stride_0, x_stride_1,
                   gamma_stride_0,
                   x_size_x, x_size_y,
                   gamma_size_x,
                   BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    
    input_m_offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x_input = tl.load(x_ptr + input_m_offsets * x_stride_0, mask=(input_m_offsets < x_size_x), other=-float('inf'))
    pow = x_input * x_input
    sum = tl.sum(pow, axis=0)
    variance = sum / x_size_y
    middle = x_input / tl.sqrt(variance + variance_epsilon)
    gamma_input = tl.load(gamma_ptr)
    res = middle * gamma_input
    tl.store(res, out_ptr + input_m_offsets[:, None] * x_stride_0 , mask=input_m_offsets < x_size_x)


def rmsnorm(x, gamma, variance_epsilon):
    assert x.shape[1] == gamma.shape[0], \
    f"gamma shape 0: {gamma.shape[0]} not match x shape 1: {x.shape[1]}"
    
    assert x.ndim == 2, "shapesize shoule equal to 2"
    
    block = 32
    out = torch.empty_like(x)
    
    grid = (((x.shape[0] + block - 1) // block),)
    
    rmsnorm_kernel[grid](
        x, gamma, out, variance_epsilon,
        x.stride(0), x.stride(1),
        gamma.stride(0),
        x.shape[0], x.shape[1],
        gamma.shape[0],
        num_stages=3, num_warps=4,
        BLOCK=block
    )
    return out


def test_rmsnorm(M, N, variance_epsilon=1e-5):
    x = torch.normal(size=(M, N), mean=0.3, std=0.5)
    gamma = torch.normal(size=(N,), mean=0.3, std=0.5)
    
    res1 = torch_rmsnorm(x, gamma, variance_epsilon)
    
    res2 = rmsnorm(x, gamma, variance_epsilon)
    
    print(f"max diff: {torch.max(res1 - res2)}")
    
    assert torch.allclose(res1, res2, atol=1e-3), "diff large"
    
test_rmsnorm(2048, 2048)