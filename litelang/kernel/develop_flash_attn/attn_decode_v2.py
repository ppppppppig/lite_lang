import triton
import triton.language as tl
import torch

import triton
import triton.language as tl
import torch


@triton.jit
def _decode_attn_fwd(
    Q,
    K,
    V,
    padding_mask,
    sm_scale,
    M,
    Out,
    temp_qk,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_qkz,
    stride_qkh,
    stride_qkm,
    stride_qkn,
    stride_pmz,
    stride_pms,
    Z,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 基础偏移计算
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    q_offset = batch_id * stride_qz + head_id * stride_qh
    vk_offset = batch_id * stride_kz + head_id * stride_kh
    qk_offset = batch_id * stride_qkz + head_id * stride_qkh
    mask_offset = batch_id * stride_pmz

    # 预定义固定形状的行掩码 (16 x BLOCK_N)
    row_mask = tl.arange(0, 16)[:, None] == 0
    row_mask = tl.broadcast_to(row_mask, (16, BLOCK_N))

    # 加载查询向量并填充到16行 (仅第0行有效)
    q = tl.zeros((16, HEAD_DIM), dtype=tl.float16)
    q0 = tl.load(Q + q_offset + tl.arange(0, HEAD_DIM) * stride_qk)
    row_mask_q = tl.arange(0, 16)[:, None] == 0
    q = tl.where(row_mask_q, q0[None, :], q)

    # 初始化累加器
    acc = tl.zeros((16, HEAD_DIM), dtype=tl.float32)
    m_i = tl.zeros([16], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([16], dtype=tl.float32) + 1.0

    # 分块处理键/值
    for start_n in range(0, N_CTX, BLOCK_N):
        cols_n = start_n + tl.arange(0, BLOCK_N)

        # 加载键块 [HEAD_DIM, BLOCK_N]
        k_ptrs = (
            K
            + vk_offset
            + cols_n * stride_kn
            + tl.arange(0, HEAD_DIM)[:, None] * stride_kk
        )
        k = tl.load(k_ptrs, mask=(cols_n < N_CTX)[None, :], other=0.0)

        # 计算注意力分数 [16, BLOCK_N]
        qk = tl.dot(q, k.to(tl.float16)) * sm_scale

        # 动态列掩码
        col_mask = cols_n[None, :] < N_CTX
        full_mask = row_mask & col_mask

        # 存储临时结果
        qk_masked = tl.where(full_mask, qk, 0.0)
        temp_ptrs = temp_qk + qk_offset + cols_n
        tl.store(temp_ptrs, tl.sum(qk_masked, axis=0), mask=cols_n < N_CTX)

        # 加载padding掩码
        pm_ptrs = padding_mask + mask_offset + cols_n
        pm = tl.load(pm_ptrs, mask=cols_n < N_CTX, other=0)
        qk = tl.where(pm[None, :] == 1, qk, -float("inf"))

        # 在线softmax计算
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        # 更新累加器
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # 加载价值块并累加
        v_ptrs = (
            V
            + vk_offset
            + cols_n[:, None] * stride_vk
            + tl.arange(0, HEAD_DIM)[None, :] * stride_vn
        )
        v = tl.load(v_ptrs, mask=cols_n[:, None] < N_CTX, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

    result_mask = tl.arange(0, 16) == 0  # 形状 [16]
    l_i_0 = tl.sum(l_i * result_mask)  # 提取l_i[0]
    final_acc = tl.sum(acc * result_mask[:, None], axis=0) / l_i_0

    # 存储最终结果
    out_ptrs = Out + q_offset + tl.arange(0, HEAD_DIM) * stride_on
    tl.store(out_ptrs, final_acc.to(tl.float16))


@torch.no_grad()
def decode_flash_attention(
    q: torch.Tensor,  # [batch, heads, 1, dim]
    k: torch.Tensor,  # [batch, heads, seq_len, dim]
    v: torch.Tensor,  # [batch, heads, seq_len, dim]
    padding_mask: torch.Tensor,  # [batch, seq_len]
    sm_scale: float = None,
):
    # 参数验证
    assert q.shape[2] == 1, "解码阶段只支持单个查询位置"
    assert k.shape == v.shape, "K/V形状必须相同"
    HEAD_DIM = q.size(-1)
    if sm_scale is None:
        sm_scale = 1.0 / (HEAD_DIM**0.5)

    # 初始化输出
    o = torch.empty_like(q)
    temp_qk = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2], k.shape[2]),
        dtype=torch.float16,
        device=q.device,
    )
    # 新的grid设置：batch_size × num_heads
    grid = (q.shape[0], q.shape[1])  # (batch, heads)

    # 根据硬件特性选择分块大小
    BLOCK_N = 32 if HEAD_DIM <= 64 else 64
    # 启动优化后的内核
    _decode_attn_fwd[grid](
        q,
        k,
        v,
        padding_mask,
        sm_scale,
        None,
        o,
        temp_qk,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        temp_qk.stride(0),
        temp_qk.stride(1),
        temp_qk.stride(2),
        temp_qk.stride(3),
        padding_mask.stride(0),
        padding_mask.stride(1),
        q.shape[0],
        q.shape[1],
        k.shape[2],
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=16,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=1,
    )
    return o


def standard_softmax_attention(Q, K, V, padding_mask, sm_scale):
    """
    修复后的标准PyTorch注意力计算
    """
    # 矩阵乘法 [batch, heads, q_len, k_len]
    p = torch.matmul(Q, K.transpose(2, 3)) * sm_scale

    # 调整掩码维度 [batch, 1, 1, k_len]
    padding_mask = padding_mask[:, None, None, :].to(torch.bool)

    # 应用掩码 (batch, heads, q_len, k_len)
    p = torch.where(padding_mask, p, float("-inf"))

    # 计算注意力概率
    p = torch.nn.functional.softmax(p, dim=-1)

    # 价值加权
    ref_out = torch.matmul(p, V)
    return ref_out


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    B_req_idx,
    B_seqlen,
    Out,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    kv_group_num,
    Q_HEAD_NUM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)

    req_start = tl.load(B_req_idx + cur_batch)
    cur_seq_len = tl.load(B_seqlen + cur_batch)

    cur_q_head_offs = tl.arange(0, Q_HEAD_NUM)

    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_seq_len = tl.load(B_seqlen + cur_batch)

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    cur_q_head_range = cur_kv_head * kv_group_num + cur_q_head_offs

    off_q = (
        cur_batch * stride_qbs + cur_q_head_range[:, None] * stride_qh + offs_d[None, :]
    )
    off_k = cur_kv_head * stride_kh + offs_d[:, None]
    off_v = cur_kv_head * stride_vh + offs_d[None, :]

    q = tl.load(
        Q + off_q,
        mask=cur_q_head_range[:, None] < (cur_kv_head + 1) * kv_group_num,
        other=0.0,
    )

    # initialize pointer to m and l
    m_i = tl.zeros([Q_HEAD_NUM], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([Q_HEAD_NUM], dtype=tl.float32)
    acc = tl.zeros([Q_HEAD_NUM, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        physical_loc = req_start + start_n + offs_n
        k_ptrs = (
            K
            + physical_loc[None, :] * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[:, None]
        )
        k = tl.load(k_ptrs, mask=(start_n + offs_n[None, :]) < cur_seq_len, other=0.0)

        qk = tl.zeros([Q_HEAD_NUM, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(
            cur_batch_seq_len - 1 >= (start_n + offs_n[None, :]), qk, float("-inf")
        )

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v_ptrs = (
            V
            + physical_loc[:, None] * stride_vbs
            + cur_kv_head * stride_vh
            + offs_d[None, :]
        )
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < cur_seq_len, other=0.0)

        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    off_o = (
        cur_batch * stride_obs + cur_q_head_range[:, None] * stride_oh + offs_d[None, :]
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs, acc, mask=cur_q_head_range[:, None] < (cur_kv_head + 1) * kv_group_num
    )
    return


@torch.no_grad()
def gqa_decode_attention_fwd(q, k, v, b_req_idx, b_seq_len, sm_scale=None):

    BLOCK = 32
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    if sm_scale is None:
        sm_scale = 1.0 / (Lq**0.5)  # 计算scale系数
    batch = b_req_idx.shape[0]
    kv_group_num = q.shape[1] // k.shape[1]
    kv_head_num = k.shape[1]

    grid = (batch, kv_head_num)
    o = torch.empty_like(q)
    num_warps = 4  # if Lk <= 64 else 8
    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_req_idx,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        kv_group_num=kv_group_num,
        Q_HEAD_NUM=max(16, triton.next_power_of_2(kv_group_num)),
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=2,
    )
    return o


def gqa_reference_impl(Q, K, V, position_ids, req_lens, sm_scale, kv_group_num):

    num_reqs, head_num, head_dim = Q.shape
    O = torch.zeros_like(Q)
    kv_heads = head_num // kv_group_num  # 计算实际的KV头数

    for req_idx in range(num_reqs):
        req_start = position_ids[req_idx]
        req_len = req_lens[req_idx]

        for group_idx in range(kv_heads):
            q_start = group_idx * kv_group_num
            q_end = q_start + kv_group_num

            q = Q[req_idx, q_start:q_end, :]
            k = K[req_start : req_start + req_len, group_idx, :]
            v = V[req_start : req_start + req_len, group_idx, :]

            scores = q * sm_scale @ k.transpose(-1, -2)  # [kv_group_num, req_len]
            attn_weights = torch.softmax(scores, dim=-1)

            # 加权求和
            O[req_idx, q_start:q_end, :] = attn_weights @ v  # [kv_group_num, head_dim]
    return O


def test_triton_attention():
    # 创建示例数据
    N_CTX, D_MODEL = 16, 64
    B, H = 2, 16
    SM_M = 101376
    dtype = torch.float16
    Q = (
        torch.empty((B, H, 1, D_MODEL), dtype=dtype, device="cuda")
        .normal_(mean=0.1, std=0.2)
        .requires_grad_()
    )
    K = (
        torch.empty((B, H, N_CTX, D_MODEL), dtype=dtype, device="cuda")
        .normal_(mean=0.4, std=0.2)
        .requires_grad_()
    )
    V = (
        torch.empty((B, H, N_CTX, D_MODEL), dtype=dtype, device="cuda")
        .normal_(mean=0.3, std=0.2)
        .requires_grad_()
    )

    sm_scale = 1.0 / (D_MODEL**0.5)

    padding_mask = torch.ones((B, N_CTX), dtype=torch.bool, device="cuda")

    padding_mask[:, 0:1] = False
    padding_mask[:, 0:1] = False
    output = decode_flash_attention(
        Q, K, V, padding_mask=padding_mask, sm_scale=sm_scale
    )
    standard_out = standard_softmax_attention(
        Q, K, V, padding_mask=padding_mask, sm_scale=sm_scale
    )
    assert torch.allclose(
        output[:, :, :, :], standard_out[:, :, :, :], atol=1e-3
    ), "Error: output and standard_out are not close"


def test_gqa_decode_attention():
    torch.manual_seed(0)

    # 配置测试参数
    num_requests = 4  # 测试请求数量
    max_seq_len = 128  # 最大序列长度
    d_model = 64  # 特征维度
    num_q_heads = 8  # Query头数
    num_kv_heads = 2  # Key/Value头数 (GQA)
    kv_group_num = num_q_heads // num_kv_heads  # 每组包含的query头数

    # 生成请求参数
    req_lens = torch.randint(1, max_seq_len, (num_requests,), device="cuda")
    req_starts = torch.cumsum(
        torch.cat([torch.tensor([0], device="cuda"), req_lens[:-1]]), dim=0
    )
    total_tokens = int(req_lens.sum())

    # 生成输入数据 (紧凑存储)
    Q = torch.randn(
        num_requests, num_q_heads, d_model, device="cuda", dtype=torch.float16
    )
    K = torch.randn(
        total_tokens, num_kv_heads, d_model, device="cuda", dtype=torch.float16
    )
    V = torch.randn(
        total_tokens, num_kv_heads, d_model, device="cuda", dtype=torch.float16
    )

    # 生成位置信息参数
    # position_ids = torch.arange(total_tokens, device="cuda")

    # 运行Triton实现
    sm_scale = 1.0 / (d_model**0.5)
    triton_output = gqa_decode_attention_fwd(
        Q, K, V, b_req_idx=req_starts, b_seq_len=req_lens, sm_scale=sm_scale
    )
    print(f"triton_output: {triton_output}")

    # 运行PyTorch参考实现
    torch_output = gqa_reference_impl(
        Q.contiguous(),  # [num_requests, ...] -> [total_tokens, ...]
        K,
        V,
        req_starts,
        req_lens,
        sm_scale,
        kv_group_num,
    )
    print(f"torch_output: {torch_output}")

    # 数值检查 (考虑FP16精度)
    # mask = (position_ids < (req_starts[:, None] + req_lens[:, None])).any(dim=0)
    assert torch.allclose(triton_output, torch_output, atol=1e-3), "数值不匹配"

    print("测试通过！输出匹配")


# 运行测试
if __name__ == "__main__":
    test_gqa_decode_attention()
