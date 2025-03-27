import torch

import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_kernel(
    input_ptr,
    output_ptr,
    stride_input_m,
    stride_input_n,
    stride_output_m,
    stride_output_n,
    size_m,
    size_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # stride_input_m = tl.cast(stride_input_m, dtype=tl.int32)
    # stride_output_m = tl.cast(stride_output_m, dtype=tl.int64)

    tid = tl.program_id(0)
    input_m_offsets = tid * BLOCK_M + tl.arange(0, BLOCK_M)
    output_m_offsets = tid * BLOCK_M + tl.arange(0, BLOCK_M)

    pid = tl.program_id(1)
    input_n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    output_n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    up_offsets = input_m_offsets[:, None] * stride_input_m + (input_n_offsets[None, :] + size_n) * stride_input_n
    gate_offsets = input_m_offsets[:, None] * stride_input_m + input_n_offsets[None, :] * stride_input_n
    res_offsets = output_m_offsets[:, None] * stride_output_m + output_n_offsets[None, :] * stride_output_n

    up = tl.load(
        input_ptr + up_offsets,
        mask=(input_n_offsets < size_n)[None, :] * (input_m_offsets < size_m)[:, None],
        other=0.0,
    )
    gate = tl.load(
        input_ptr + gate_offsets,
        mask=(input_n_offsets < size_n)[None, :] * (input_m_offsets < size_m)[:, None],
        other=0.0,
    ).to(tl.float32)

    gate = gate / (1 + tl.exp(-gate))
    gate = gate.to(input_ptr.dtype.element_ty)

    tl.store(
        output_ptr + res_offsets,
        up * gate,
        mask=(output_n_offsets < size_n)[None, :] * (output_m_offsets < size_m)[:, None],
    )


def silu_and_mul_fwd(input: torch.Tensor, output):
    stride_input_m = input.stride(0)
    stride_input_n = input.stride(1)
    stride_output_m = output.stride(0)
    stride_output_n = output.stride(1)
    size_m = input.shape[0]
    size_n = input.shape[-1] // 2
    BLOCK_M = 128
    BLOCK_N = 128
    grid = (
        triton.cdiv(size_m, BLOCK_M),
        triton.cdiv(size_n, BLOCK_N),
    )
    _silu_and_mul_kernel[grid](
        input,
        output,
        stride_input_m,
        stride_input_n,
        stride_output_m,
        stride_output_n,
        size_m,
        size_n,
        BLOCK_M,
        BLOCK_N,
    )
    return


def torch_silu_and_mul(input: torch.Tensor):
    return torch.nn.functional.silu(input[:, 0 : (input.shape[-1] // 2)]) * input[:, (input.shape[-1] // 2) :]


def test_silu_and_mul(M, N, dtype, device="cuda"):
    # create data
    X = torch.randn((M, N), dtype=dtype, device=device)
    out = torch.empty((M, N // 2), dtype=dtype, device=device)
    # run
    silu_and_mul_fwd(X, out)
    y_ref = torch_silu_and_mul(X)

    # compare
    print("type:", out.dtype, y_ref.dtype)
    print("max delta:", torch.max(torch.abs(out - y_ref)))
    assert torch.allclose(out, y_ref, atol=1e-3, rtol=0)
    return  

if __name__ == '__main__':
    test_silu_and_mul(2048, 2048, torch.float16)