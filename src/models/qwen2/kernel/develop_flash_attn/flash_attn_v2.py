
import triton
import triton.language as tl
import torch

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
        tl.static_print("fsdfsdf")
        tl.static_print(q)
        tl.static_print(k)
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
    BLOCK_N = 32
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


@triton.jit
def gqa_context_attention_fwd(
    Q, K, V, sm_scale, Out,
    B_Start_Loc,  # 每个请求的起始位置 [batch]
    B_Seqlen,     # 每个请求的序列长度 [batch]
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    kv_group_num,
    H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.device_print("fk you1")
    # 计算程序ID和元数据
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H
    cur_kv_head = cur_head // kv_group_num

    # 加载当前请求的元数据
    cur_batch_start = tl.load(B_Start_Loc + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    # 初始化块偏移
    block_start_loc = BLOCK_M * start_m
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # 计算Q指针
    off_q = (cur_batch_start + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    # 初始化累加器
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # 处理有效块
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M, cur_batch_seq_len)
    tl.device_print("fk you2")
    for start_n in range(0, block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # 加载K [BLOCK_N, d]
        off_k = (cur_batch_start + start_n + offs_n[None, :]) * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)
        
        # 计算注意力分数
        qk = tl.dot(q, k) * sm_scale
        
        # 应用因果掩码
        mask = (offs_m[:, None] >= (start_n + offs_n[None, :]))
        qk = tl.where(mask, qk, float("-inf"))
        
        # Stable attention计算
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        
        # 更新累加器
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        # 加载V并累加
        off_v = (cur_batch_start + start_n + offs_n[:, None]) * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < block_end_loc, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
    # 写回结果
    tl.device_print("fk you2")
    acc = acc / l_i[:, None]
    off_o = (cur_batch_start + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
    # tl.store(out_ptrs, acc, mask=cur_q_head_range[:, None] < (cur_kv_head + 1) * kv_group_num)

@torch.no_grad()
def gqa_context_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b_start_loc: torch.Tensor,  # [batch]
    b_seq_len: torch.Tensor,    # [batch]
    kv_group_num: int = 1,
    sm_scale: float = None,
):
    # 参数校验
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    batch, q_heads, d_model = q.shape
    _, kv_heads, _ = k.shape
    assert kv_group_num == q_heads // kv_heads
    
    # 自动计算scale
    if sm_scale is None:
        sm_scale = 1.0 / (d_model ** 0.5)
    sm_scale *= 1.44269504  # 适配exp2计算
    
    BLOCK = 32
    BLOCK_M = BLOCK_N = BLOCK
    # 配置计算网格
    max_seq_len = triton.cdiv(int(torch.max(b_seq_len).item()), BLOCK)
    grid = (triton.cdiv(max_seq_len, BLOCK), batch * q_heads)
    
    # 分配输出张量
    o = torch.empty_like(q)
    # 启动内核
    gqa_context_attention_fwd[grid](
        q, k, v, sm_scale, o,
        b_start_loc, b_seq_len,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        kv_group_num,
        H=q_heads,
        BLOCK_DMODEL=d_model,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=1
    )
    return o

def standard_attention_no_pad(Q, K, V, b_start_loc, b_seq_len, kv_group_num, sm_scale):
    batch_mul_seq_len, q_heads, d_model = Q.shape
    num_reqs = b_start_loc.size(0)
    kv_heads = K.shape[1]
    O = torch.zeros_like(Q)
    
    for req_idx in range(num_reqs):
        start = b_start_loc[req_idx]
        seq_len = b_seq_len[req_idx]
        
        q = Q[start : start + seq_len]
        k = K[start : start + seq_len]
        v = V[start : start + seq_len]
        
        for kv_head_idx in range(kv_heads):
            start_q_idx = kv_head_idx * kv_group_num
            end_q_idx = (kv_head_idx + 1) * kv_group_num
            q_group = q[:, start_q_idx : end_q_idx].transpose(0, 1)
            k_group = k[:, kv_head_idx, :].unsqueeze(1).transpose(0, 1)
            v_group = v[:, kv_head_idx, :].unsqueeze(1).transpose(0, 1)
            
            attn_scores = torch.matmul( q_group * sm_scale, k_group.transpose(-1, -2))
            
            # 因果掩码
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device)).unsqueeze(0)
            attn_scores = torch.where(causal_mask.bool(), attn_scores, -float('inf'))
            
            attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(Q.dtype)
            attn_scores = torch.matmul(attn_weights, v_group).transpose(0, 1)
            
            O[start : start + seq_len, start_q_idx : end_q_idx] = attn_scores
    return O


@triton.jit
def _fwd_kernel_with_no_padding_and_kv_cache(
    Q,
    K,
    V,
    sm_scale,
    Out,
    B_Start_Loc,
    B_Seqlen,
    Req_to_tokens,
    B_req_idx,
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
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    kv_group_num,
    H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H

    cur_kv_head = cur_head // kv_group_num

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )

    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M, cur_batch_seq_len)

    # causal mask
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * (start_n + offs_n),
            mask=(start_n + offs_n) < block_end_loc,
            other=0,
        )
        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)
        qk = tl.dot(q, k)

        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        off_v = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < block_end_loc, other=0.0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@torch.no_grad()
def context_attention_fwd_with_no_pad_and_kv_cache(
    q, k, v, o, b_req_idx, b_start_loc, b_seq_len, max_input_len, req_to_token_indexs
):
    BLOCK_M = 64
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}

    # 计算scale系数, 并乘以 1/log(2) = 1.4426950408889634,
    # 算子内部使用 tl.math.exp2 来使计算与标准attention等价。
    sm_scale = 1.0 / (Lq ** 0.5) * 1.4426950408889634
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = lambda meta: (triton.cdiv(max_input_len, meta["BLOCK_M"]), batch * head, 1)

    BLOCK_N = BLOCK_M
    num_warps = 4 if Lk <= 64 else 8
    num_stages = 1

    _fwd_kernel_with_no_padding_and_kv_cache[grid](
        q,
        k,
        v,
        sm_scale,
        o,
        b_start_loc,
        b_seq_len,
        req_to_token_indexs,
        b_req_idx,
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
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        kv_group_num=kv_group_num,
        H=head,
        BLOCK_DMODEL=Lk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    
@triton.jit
def _fwd_kernel_with_no_padding_and_kv_cache_and_prompt_cache(
    Q,
    K,
    V,
    sm_scale,
    Out,
    B_Start_Loc,
    B_Seqlen,
    Req_to_tokens,
    B_req_idx,
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
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    b_shared_seq_len,
    kv_group_num,
    H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H

    cur_kv_head = cur_head // kv_group_num

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_shared_seq_len = tl.load(b_shared_seq_len + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - cur_batch_shared_seq_len
    # tl.device_print("cur_batch_in_all_start_index: %d", cur_batch_in_all_start_index)
    # tl.device_print("cur_batch_shared_seq_len: %d" ,cur_batch_shared_seq_len)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )

    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M + cur_batch_shared_seq_len, cur_batch_seq_len + cur_batch_shared_seq_len)

    # causal mask
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * (start_n + offs_n),
            mask=(start_n + offs_n) < block_end_loc,
            other=0,
        )
        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)
        qk = tl.dot(q, k)

        mask = (offs_m[:, None] + cur_batch_shared_seq_len) >= (start_n + offs_n[None, :])
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        off_v = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < block_end_loc, other=0.0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@torch.no_grad()
def context_attention_fwd_with_no_pad_and_kv_cache_and_prompt_cache(
    q, k, v, o, b_req_idx, b_start_loc, b_seq_len, b_shared_seq_len, max_input_len, req_to_token_indexs
):
    BLOCK_M = 64
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}

    # 计算scale系数, 并乘以 1/log(2) = 1.4426950408889634,
    # 算子内部使用 tl.math.exp2 来使计算与标准attention等价。
    sm_scale = 1.0 / (Lq ** 0.5) * 1.4426950408889634
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = lambda meta: (triton.cdiv(max_input_len, meta["BLOCK_M"]), batch * head, 1)

    BLOCK_N = BLOCK_M
    num_warps = 4 if Lk <= 64 else 8
    num_stages = 1

    _fwd_kernel_with_no_padding_and_kv_cache_and_prompt_cache[grid](
        q,
        k,
        v,
        sm_scale,
        o,
        b_start_loc,
        b_seq_len,
        req_to_token_indexs,
        b_req_idx,
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
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        b_shared_seq_len,
        kv_group_num=kv_group_num,
        H=head,
        BLOCK_DMODEL=Lk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

def test_context_attention_with_kv_cache_and_prompt_cache():
    def generate_test_data(q_heads=8, kv_heads=2, d_model=64):
        """生成带KV缓存的测试数据"""
        assert q_heads % kv_heads == 0, "q_heads 必须是 kv_heads 的整数倍"
        
        batch_size = 2
        dtype = torch.float16
        device = "cuda"
        
        # 生成序列长度和起始位置
        seq_lens = torch.tensor([16, 32], device=device)
        shared_lens = torch.tensor([8, 11], device=device)
        b_start_loc = torch.tensor([0, 16], device=device)
        total_tokens = seq_lens.sum()
        
        # 生成请求索引映射表（模拟KV缓存位置）
        req_to_token = torch.zeros((batch_size, seq_lens.max()), 
                                 dtype=torch.long, device=device)
        for i in range(batch_size):
            req_to_token[i, :seq_lens[i]] = b_start_loc[i] + torch.arange(seq_lens[i], device=device)
        
        # 生成输入张量
        Q = torch.randn(total_tokens, q_heads, d_model, device=device, dtype=dtype)
        K = torch.randn(total_tokens, kv_heads, d_model, device=device, dtype=dtype)
        V = torch.randn_like(K)
        
        b_req_idx = torch.arange(batch_size, device=device)
        b_start_loc_use_prompt = torch.cat([torch.tensor([0, 8], device=device), torch.cumsum(seq_lens, 0)[:-1]])
        return Q, K, V, b_start_loc, seq_lens, req_to_token, b_req_idx, shared_lens, b_start_loc_use_prompt

    # 生成测试数据
    Q, K, V, b_start_loc, seq_lens, req_to_token, b_req_idx, b_shared_lens, b_start_loc_use_prompt = generate_test_data()
    
    slice_q = torch.cat([Q[8:16], Q[27:]])
    output = torch.empty_like(slice_q)
    def standard_attention():
        output_ref = torch.zeros_like(Q)
        kv_group_num = Q.size(1) // K.size(1)
        
        for batch in range(len(seq_lens)):
            start = b_start_loc[batch]
            end = start + seq_lens[batch]
            M = torch.tril(torch.ones(seq_lens[batch], seq_lens[batch], device=Q.device, dtype=Q.dtype))
            # 获取当前batch的token映射
            token_indices = req_to_token[batch][:seq_lens[batch]]
            for q_head in range(Q.size(1)):
                kv_head = q_head // kv_group_num
                
                # 获取对应的K/V数据
                k = K[token_indices, kv_head]
                v = V[token_indices, kv_head]
                # 计算注意力
                q_data = Q[start: end, q_head]
                scores = torch.matmul(q_data, k.transpose(0, 1)) * (1.0 / (Q.size(-1) ** 0.5))
                scores = torch.where(M == 1, scores, -1.0e4)
                scores = torch.softmax(scores, dim=-1)
                output_ref[start: end, q_head] = torch.matmul(scores, v)
        
        return output_ref

    # 计算标准结果
    ref_output = standard_attention()
    
    context_attention_fwd_with_no_pad_and_kv_cache_and_prompt_cache(
        q=slice_q, k=K, v=V,
        o=output,
        b_req_idx=b_req_idx,
        b_start_loc=b_start_loc_use_prompt,
        b_seq_len=seq_lens,
        b_shared_seq_len=b_shared_lens,
        max_input_len=seq_lens.max().item(),
        req_to_token_indexs=req_to_token
    )
    
    
    # 验证精度
    assert torch.allclose(output[-1], ref_output[-1], atol=1e-2), "精度验证失败！"
    max_diff = torch.max(torch.abs(output[-1] - ref_output[-1]))
    print(f"✅ 测试通过，最大差异：{max_diff.item():.6f}")


def test_context_attention_with_kv_cache():
    def generate_test_data(q_heads=8, kv_heads=2, d_model=64):
        """生成带KV缓存的测试数据"""
        assert q_heads % kv_heads == 0, "q_heads 必须是 kv_heads 的整数倍"
        
        batch_size = 2
        dtype = torch.float16
        device = "cuda"
        
        # 生成序列长度和起始位置
        seq_lens = torch.tensor([16, 32], device=device)
        b_start_loc = torch.cat([torch.tensor([0, 16], device=device), torch.cumsum(seq_lens, 0)[:-1]])
        total_tokens = seq_lens.sum()
        
        # 生成请求索引映射表（模拟KV缓存位置）
        req_to_token = torch.zeros((batch_size, seq_lens.max()), 
                                 dtype=torch.long, device=device)
        for i in range(batch_size):
            req_to_token[i, :seq_lens[i]] = b_start_loc[i] + torch.arange(seq_lens[i], device=device)
        
        # 生成输入张量
        Q = torch.randn(total_tokens, q_heads, d_model, device=device, dtype=dtype)
        K = torch.randn(total_tokens, kv_heads, d_model, device=device, dtype=dtype)
        V = torch.randn_like(K)
        
        
        # 其他参数（示例中暂时不需要prompt缓存）

        b_req_idx = torch.arange(batch_size, device=device)
        
        return Q, K, V, b_start_loc, seq_lens, req_to_token, b_req_idx

    # 生成测试数据
    Q, K, V, b_start_loc, seq_lens, req_to_token, b_req_idx = generate_test_data()
    
    # 准备输出张量
    output = torch.empty_like(Q)
    
    # 调用Triton实现
    context_attention_fwd_with_no_pad_and_kv_cache(
        q=Q, k=K, v=V,
        o=output,
        b_req_idx=b_req_idx,
        b_start_loc=b_start_loc,
        b_seq_len=seq_lens,
        max_input_len=seq_lens.max().item(),
        req_to_token_indexs=req_to_token
    )
    
    def standard_attention():
        output_ref = torch.zeros_like(Q)
        kv_group_num = Q.size(1) // K.size(1)
        
        for batch in range(len(seq_lens)):
            start = b_start_loc[batch]
            end = start + seq_lens[batch]
            M = torch.tril(torch.ones(seq_lens[batch], seq_lens[batch], device=Q.device, dtype=Q.dtype))
            # 获取当前batch的token映射
            token_indices = req_to_token[batch][:seq_lens[batch]]
            
            for q_head in range(Q.size(1)):
                kv_head = q_head // kv_group_num
                
                # 获取对应的K/V数据
                k = K[token_indices, kv_head]
                v = V[token_indices, kv_head]
                # 计算注意力
                q_data = Q[start: end, q_head]
                scores = torch.matmul(q_data, k.transpose(0, 1)) * (1.0 / (Q.size(-1) ** 0.5))
                scores = torch.where(M == 1, scores, -1.0e4)
                scores = torch.softmax(scores, dim=-1)
                output_ref[start: end, q_head] = torch.matmul(scores, v)
        
        return output_ref

    # 计算标准结果
    ref_output = standard_attention()
    
    # 验证精度
    assert torch.allclose(output, ref_output, atol=1e-2), "精度验证失败！"
    max_diff = torch.max(torch.abs(output - ref_output))
    print(f"✅ 测试通过，最大差异：{max_diff.item():.6f}")


def triton_attention():
    # 创建示例数据
    N_CTX, D_MODEL = 32, 64
    B, H = 1, 3
    SM_M = 101376
    dtype = torch.float16
    Q = torch.empty((B, H, N_CTX, D_MODEL), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2).requires_grad_()
    K = torch.empty((B, H, N_CTX, D_MODEL), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2).requires_grad_()
    V = torch.empty((B, H, N_CTX, D_MODEL), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2).requires_grad_()

    padding_mask = torch.ones((B, N_CTX), dtype=torch.bool, device="cuda")
    sm_scale = 0.01

    
    padding_mask[:, 0:1] = False
    times = 1
    while times > 0:
        times -= 1
        output = triton_attention(Q, K, V, padding_mask=padding_mask, sm_scale=sm_scale)
        standard_out = standard_softmax_attention(Q, K, V, padding_mask=padding_mask, sm_scale=sm_scale)
        assert torch.allclose(output[:, :, 1:, :], standard_out[:, :, 1:, :], atol=1e-3), "Error: output and standard_out are not close"

def triton_attention_no_pad():
    def generate_gqa_test_data(q_heads=8, kv_heads=2, d_model=64):
        """生成 GQA 测试数据 (Q heads > K/V heads)"""
        assert q_heads % kv_heads == 0, "q_heads 必须是 kv_heads 的整数倍"
        
        # 随机生成序列
        batch = 2
        dtype = torch.float16
        seq_lens = torch.tensor([16, 32], device="cuda")  # 短的测试序列加速计算
        b_start_loc = torch.cat([torch.tensor([0], device="cuda"), torch.cumsum(seq_lens, 0)[:-1]]).cuda()
        total_tokens = seq_lens.sum()
        
        # Q 和 K/V 的head数不同
        Q = torch.randn(total_tokens, q_heads, d_model, device="cuda", dtype=dtype)
        K = torch.randn(total_tokens, kv_heads, d_model, device="cuda", dtype=dtype)
        V = torch.randn_like(K)  # V 与 K 维度一致
        
        return Q, K, V, b_start_loc, seq_lens
    Q, K, V, b_start_loc, seq_len = generate_gqa_test_data(q_heads=8, kv_heads=2)
    
    # Triton 实现
    triton_out = gqa_context_attention(
        Q, K, V, b_start_loc, seq_len,
        kv_group_num=4,  # q_heads / kv_heads = 4
        sm_scale=0.1
    )
    print(f"triton output: {triton_out}")
    # 标准实现
    ref_out = standard_attention_no_pad(
        Q, K, V, b_start_loc, seq_len,
        kv_group_num=4,
        sm_scale=0.1
    )
    print(f"standard output: {ref_out}")
    print(f"max diff { torch.max(ref_out - triton_out)}")
    # 验证精度一致性
    assert torch.allclose(
        triton_out, ref_out,  atol=1e-2
    ), "GQA分组场景精度不匹配！"
    print("✅ GQA分组逻辑测试通过")

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
    test_context_attention_with_kv_cache_and_prompt_cache()