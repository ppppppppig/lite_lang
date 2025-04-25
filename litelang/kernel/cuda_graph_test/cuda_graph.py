import triton
import triton.language as tl
import torch
import os
import math


class AttnCudaGraph:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, sm_scale):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.sm_scale = sm_scale

        # 创建计算图的输入和输出张量
        self.Q = torch.empty(
            (batch_size, num_heads, seq_len, head_dim), device="cuda", dtype=torch.half
        )
        self.K = torch.empty(
            (batch_size, num_heads, seq_len, head_dim), device="cuda", dtype=torch.half
        )
        self.V = torch.empty(
            (batch_size, num_heads, seq_len, head_dim), device="cuda", dtype=torch.half
        )
        self.output = torch.empty(
            (batch_size, num_heads, seq_len, head_dim), device="cuda", dtype=torch.half
        )
        torch.cuda.init()
        # 构造 CUDA Graph
        self._build_cuda_graph()

    def _build_cuda_graph(self):
        # 预热 CUBLAS：执行一次 dummy matmul 以确保 CUBLAS 初始化
        dummy_A = torch.randn((1, 1), device="cuda", dtype=torch.float16)
        dummy_B = torch.randn((1, 1), device="cuda", dtype=torch.float16)
        _ = torch.matmul(dummy_A, dummy_B)

        # 创建一个 CUDA Graph 实例
        g = torch.cuda.CUDAGraph()

        try:
            # 开始捕获 CUDA Graph
            with torch.cuda.graph(g):
                self.output.copy_(
                    self._compute_attention(self.Q, self.K, self.V, self.sm_scale)
                )
            # 保存捕获到的 CUDA Graph
            self.graph = g
        except Exception as e:
            print(f"Error during CUDA graph capture: {e}")
            torch.cuda.empty_cache()
            raise

    def _compute_attention(self, Q, K, V, sm_scale):
        """计算 softmax attention"""
        # 计算注意力分数
        p = torch.matmul(Q, K.transpose(2, 3)) * sm_scale

        # Mask: 使用下三角 mask
        M = torch.tril(
            torch.ones(self.seq_len, self.seq_len, device="cuda", dtype=torch.bool)
        )
        p.masked_fill_(~M, float("-inf"))

        # 计算 softmax 和 attention
        p = torch.softmax(p.float(), dim=-1).half()
        return torch.matmul(p, V)

    def run(self, Q, K, V):
        """运行 CUDA Graph 版本的 Attention 计算"""
        self.Q.copy_(Q)
        self.K.copy_(K)
        self.V.copy_(V)

        # 执行捕获好的 CUDA Graph
        self.graph.replay()

        return self.output


# 标准的 Attention
def standard_softmax_attention(Q, K, V, sm_scale):
    """
    执行标准的PyTorch softmax和attention计算。
    """

    M = torch.tril(torch.ones(Q.size(-2), K.size(-2), device="cuda", dtype=torch.bool))
    p = torch.matmul(Q, K.transpose(2, 3)) * sm_scale
    p.masked_fill_(~M, float("-inf"))
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, V)
    return ref_out


# 创建示例数据
N_CTX, D_MODEL = 512, 64
B, H = 16, 48
dtype = torch.float16
Q = (
    torch.empty((B, H, N_CTX, D_MODEL), dtype=dtype, device="cuda")
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

for i in range(0, 20):
    torch.cuda.synchronize()
    standard_softmax_attention(Q, K, V, sm_scale=1)

attention_cuda = AttnCudaGraph(B, H, N_CTX, D_MODEL, sm_scale=1)
for i in range(0, 20):
    torch.cuda.synchronize()
    attention_cuda.run(Q, K, V)
