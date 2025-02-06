import torch

def MhaAttn(q, k, v):
    qk = q @ k.transpose(-2, -1)
    qk = qk / (qk.size(-1) ** 2)
    attn_score = qk @ v
    return attn_score

# 假设SRAM大小为4096
SRAM_SIZE = 4096
# q = [b, h, seq_len, d_k]
# 该算子以fp16进行计算
def flash_attn_v2(q, k, v):
    BLOCK = SRAM_SIZE // q.size(-1)
    
    seq_len = q.size(-2)
    batch_size = q.size(0)
    head_num = q.size(1)
    head_dim = q.size(-1)
    M = torch.empty(batch_size, head_num, BLOCK, 1).fill_(float('-inf'))
    L = torch.empty(batch_size, head_num, BLOCK, head_dim)
    O = torch.empty(batch_size, head_num, seq_len, head_dim)
    
    for i in range(0, seq_len, BLOCK):
        q_block = q[:, :, i:i+BLOCK, :]
        o_block = O[:, :, i:i+BLOCK, :]
        
        for j in range(0, seq_len, BLOCK):
            k_block = k[:, :, j:j+BLOCK, :]
            v_block = v[:, :, j:j+BLOCK, :]
            
            qk_block = torch.matmul(q_block, k_block.transpose(-2, -1))
            qk_block = qk_block / (head_dim ** 0.5)
            
            M_new = torch.max(M, torch.max(qk_block, dim=-1))
            L = L * torch.exp(M - M_new) + torch.sum(torch.exp(qk_block - M_new))
            o_block = torch.exp(qk_block - M_new) / L + o_block * torch.exp(M - M_new)
            
            M = M_new
            
    return O
            

def test_flash_v2():
    N, d = 3, 128  # 更新N和d的值

    Q_mat = torch.rand((N, d))
    K_mat = torch.rand((N, d))
    V_mat = torch.rand((N, d))

    # 执行flash attention计算
    flash_attention_v2_output = flash_attn_v2(Q_mat, K_mat, V_mat)

    # 执行标准的PyTorch softmax和attention计算
    _, expected_attention = MhaAttn(Q_mat, K_mat, V_mat)
    print(flash_attention_v2_output)
    print(expected_attention)
    # 断言flash attention计算的结果与标准计算结果是否接近
    assert torch.allclose(flash_attention_v2_output, expected_attention), "Error in flash attention calculation"

if __name__ == '__main__':
    test_flash_v2()