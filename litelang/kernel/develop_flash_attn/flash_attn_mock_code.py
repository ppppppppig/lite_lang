import torch


def MhaAttn(q, k, v):
    qk = q @ k.transpose(-2, -1)
    qk = qk / (q.size(-1) ** 0.5)
    qk = torch.softmax(qk, dim=-1)
    attn_score = qk @ v
    return attn_score


# 假设SRAM大小为4096
SRAM_SIZE = 4096


def flash_attn_v2(q, k, v):
    BLOCK = SRAM_SIZE // q.size(-1) // 4
    seq_len = q.size(-2)
    batch_size = q.size(0)
    head_num = q.size(1)
    head_dim = q.size(-1)
    M = -torch.inf * torch.ones(batch_size, head_num, seq_len, 1)
    L = torch.zeros(batch_size, head_num, seq_len, head_dim)
    O = torch.zeros(batch_size, head_num, seq_len, head_dim)

    for i in range(0, seq_len, BLOCK):
        q_block = q[:, :, i : i + BLOCK, :]
        o_block = O[:, :, i : i + BLOCK, :]
        m_block = M[:, :, i : i + BLOCK, :]
        l_block = L[:, :, i : i + BLOCK, :]

        for j in range(0, seq_len, BLOCK):
            k_block = k[:, :, j : j + BLOCK, :]
            v_block = v[:, :, j : j + BLOCK, :]
            qk_block = q_block @ k_block.transpose(-2, -1) / (head_dim**0.5)
            current_m = torch.max(qk_block, dim=-1, keepdim=True).values
            M_new = torch.maximum(m_block, current_m)

            l_block = l_block * torch.exp(m_block - M_new) + torch.sum(
                torch.exp(qk_block - M_new), dim=-1, keepdim=True
            )
            o_block = (
                o_block * torch.exp(m_block - M_new)
                + torch.exp(qk_block - M_new) @ v_block
            )
            m_block = M_new
        o_block = o_block / l_block
        O[:, :, i : i + BLOCK, :] = o_block

    return O


def test_flash_v2():
    N, d = 256, 128  # 更新N和d的值
    B, h = 2, 16
    Q_mat = torch.rand((B, h, N, d), dtype=torch.float32)
    K_mat = torch.rand((B, h, N, d), dtype=torch.float32)
    V_mat = torch.rand((B, h, N, d), dtype=torch.float32)

    # 执行flash attention计算
    flash_attention_v2_output = flash_attn_v2(Q_mat, K_mat, V_mat)

    # 执行标准的PyTorch softmax和attention计算
    expected_attention = MhaAttn(Q_mat, K_mat, V_mat)
    # 断言flash attention计算的结果与标准计算结果是否接近
    assert torch.allclose(
        flash_attention_v2_output, expected_attention, atol=1e-07
    ), "Error in flash attention calculation"


if __name__ == "__main__":
    test_flash_v2()
