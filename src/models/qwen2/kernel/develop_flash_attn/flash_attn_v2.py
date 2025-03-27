
import triton
import triton.language as tl
import torch

import pdb
@triton.jit
def _fwd_kernel(
    Q, K, V,
    sm_scale,
    Out,
    padding_mask,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    stride_pmb,
    stride_pmh,
    seq_len: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    
    # 三维网格划分
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)
    pid_m = tl.program_id(0)

    # 计算当前处理的序列范围
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # 加载Q的当前块
    q_ptrs = Q + pid_batch * stride_qb + pid_head * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)

    # 初始化累加器
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # 遍历K/V的列块
    for block_n in range(0, seq_len, BLOCK_N):
        # 加载K块
        k_ptrs = K + pid_batch * stride_kb + pid_head * stride_kh + (block_n + offs_n[None, :]) * stride_kn + offs_d[:, None]
        k = tl.load(k_ptrs, mask=(block_n + offs_n[None, :]) < seq_len, other=0.0)

        # 计算QK
        qk = tl.dot(q, k) * sm_scale
        
        # 应用因果掩码
        if IS_CAUSAL:
            mask = (offs_m[:, None] >= (block_n + offs_n[None, :]))
            qk = tl.where(mask, qk, float("-inf"))
            
        # 应用padding掩码
        padding_mask_ptrs = padding_mask + pid_batch * stride_pmh + block_n + offs_n
        pm = tl.load(padding_mask_ptrs, mask=(block_n + offs_n) < seq_len, other=0)
        qk = tl.where(pm[None, :] == 1, qk, float("-inf"))

        # Stable softmax
        m_ij = tl.maximum(tl.max(qk, axis=1), m_i)
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, axis=1)

        # 更新累加器
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        # 加载V块并累加
        v_ptrs = V + pid_batch * stride_vb + pid_head * stride_vh + (block_n + offs_n[:, None]) * stride_vn + offs_d[None, :]
        v = tl.load(v_ptrs, mask=(block_n + offs_n[:, None]) < seq_len, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_ij

    # 写入最终结果
    acc = acc / l_i[:, None]
    o_ptrs = Out + pid_batch * stride_ob + pid_head * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < seq_len)


def triton_attention(Q, K, V, padding_mask, sm_scale=None):
    """
    对齐标准注意力函数的Triton实现
    输入格式:
    Q/K/V: (batch_size, num_heads, seq_len, d_model)
    padding_mask: (batch_size, seq_len) 
    """
    assert Q.shape == K.shape == V.shape
    batch, n_heads, seq_len, d_model = Q.shape
    
    # 自动计算缩放因子
    if sm_scale is None:
        sm_scale = 1.0 / (d_model ** 0.5)
    
    # 输出初始化
    Out = torch.empty_like(Q)
    
    # 配置执行网格
    BLOCK_M = 64 if d_model <= 64 else 32
    BLOCK_N = 64
    grid = ( triton.cdiv(seq_len, BLOCK_M), batch , n_heads)
    
    # 转换padding_mask格式
    padding_mask = padding_mask.to(torch.int32)
    
    # 调用内核
    _fwd_kernel[grid](
        Q, K, V,
        sm_scale,
        Out,
        padding_mask,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        padding_mask.stride(0), padding_mask.stride(1),
        seq_len=seq_len,
        IS_CAUSAL=True,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=d_model,
        num_warps=4,
        num_stages=2
    )
    return Out

# 标准的 Attention
def standard_softmax_attention(Q, K, V, padding_mask, sm_scale):
    """
    执行标准的PyTorch softmax和attention计算。
    """

    M = torch.tril(torch.ones(Q.size(-2), K.size(-2), device="cuda"))
    p = torch.matmul(Q * sm_scale, K.transpose(2, 3))
    for z in range(Q.size(0)):
        for h in range(Q.size(1)):
            p[z, h, :, :] = torch.where(M == 1, p[z, h, :, :], float("-inf") )
            p[z, h, :, :] = torch.where(padding_mask[z, None, None, :], p[z, h, :, :], float('-inf'))
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, V)
    return ref_out
    

if __name__ == "__main__":
    # 创建示例数据
    N_CTX, D_MODEL = 32, 64
    B, H = 1, 3
    SM_M = 101376
    dtype = torch.float16
    Q = torch.empty((B, H, N_CTX, D_MODEL), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2).requires_grad_()
    K = torch.empty((B, H, N_CTX, D_MODEL), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2).requires_grad_()
    V = torch.empty((B, H, N_CTX, D_MODEL), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2).requires_grad_()
    # torch.cuda.set_device(4)
    # Q = torch.load("q.pt").cuda()
    # K = torch.load("k.pt").cuda()
    # V = torch.load("v.pt").cuda()
    # print(f"Q: {Q}")
    # print(f"K: {K}")
    # print(f"V: {V}")
    padding_mask = torch.ones((B, N_CTX), dtype=torch.bool, device="cuda")
    sm_scale = 0.01
    # output = triton_attention(Q, K, V, padding_mask=padding_mask, sm_scale=sm_scale)
    # print(f"max: {torch.max(output)}")
    
    padding_mask[:, 0:1] = False
    # print(f"padding_mask: {padding_mask}")
    # padding_mask[0, 1] = False
    times = 1
    while times > 0:
        times -= 1
        output = triton_attention(Q, K, V, padding_mask=padding_mask, sm_scale=sm_scale)
        standard_out = standard_softmax_attention(Q, K, V, padding_mask=padding_mask, sm_scale=sm_scale)
        print(f"output: {output[:, :, :, :]}")
        print(f"max: {torch.max(output)}")
        print(f"standard_out: {standard_out[:, :, :, :]}")
        assert torch.allclose(output[:, :, 1:, :], standard_out[:, :, 1:, :], atol=1e-3), "Error: output and standard_out are not close"

