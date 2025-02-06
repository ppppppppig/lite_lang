import torch

def update_softmax_block(test_tensor_block, m_pre, l_pre, all_softmax_tensor, index):
    m_cur = max(torch.max(test_tensor_block), m_pre)
    l_pre *= torch.exp(m_pre - m_cur)
    p = torch.exp(test_tensor_block - m_cur)
    l_cur = l_pre + torch.sum(p)
    for i in range(index):
        all_softmax_tensor[i] *= l_pre / l_cur
    l_pre = l_cur
    p = p / l_cur
    all_softmax_tensor[index] = p
    return m_cur, l_pre

test_tensor = torch.tensor([1,2,3], dtype=torch.float32)
m_pre = float('-inf')  # 前面的最大值
l_pre = 1  # 前面的分母和
block = 1

all_softmax_tensor = torch.empty_like(test_tensor)

test_tensor_block1 = torch.tensor([1], dtype=torch.float32)
m_pre, l_pre = update_softmax_block(test_tensor_block1, m_pre, l_pre, all_softmax_tensor, 0)
print(f"block 1 : {all_softmax_tensor}")

# Block 2
test_tensor_block2 = torch.tensor([2], dtype=torch.float32)
m_pre, l_pre = update_softmax_block(test_tensor_block2, m_pre, l_pre, all_softmax_tensor, 1)
print(f"block 2 : {all_softmax_tensor}")

# Block 3
test_tensor_block3 = torch.tensor([3], dtype=torch.float32)
m_pre, l_pre = update_softmax_block(test_tensor_block3, m_pre, l_pre, all_softmax_tensor, 2)
print(f"block 3 : {all_softmax_tensor}")

print(f"aftter softmax : {torch.softmax(test_tensor, dim=0)}")