### 1.背景（what&why )

使用[算子融合](../develop_silu_and_mul/develop_sillu_and_mul.md#21-算子融合)显然可以提高rmsnorm的性能，以qwen2举例子，其rmsnorm算子定义如下：

```
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

```

可以推断，该rmsnorm算子，在推理过程中，共用7读5写，分别是：

```
def forward(self, hidden_states):
    input_dtype = hidden_states.dtype                                                 # 一读一写
    hidden_states = hidden_states.to(torch.float32)                                   # 一读一写
    variance = hidden_states.pow(2).mean(-1, keepdim=True)                            # 一读一写
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)     # 两读一写
    return self.weight * hidden_states.to(input_dtype)                                # 两读一写
```

使用算子融合后，防存次数变成了2读1写。

在该场景下分析算子融合的优势有：

1.降低kernel launch的开销，cpu负载

2.降低了防存开销，间接减少了HBM上的防存占用

### 2.几种常见Norm算法

Norm方法一般是为了加强模型的泛化能力（防止过拟合），加强训练稳定性，防止梯度爆炸或消失。

#### 2.1 LayerNorm

该算法公式如下：

$$
y = \frac{x - E(x)} {\sqrt{(variance + variance\_eplison)}} * gamma + beta
$$

其中： $ variance = \frac {1} {N} {\sum^{i}_N{ (x - E(x))^2}}$

该算子计算量稍大，一般用于小模型

#### 2.2 rmsNorm

该算法公式如下，注意<b>rmsnorm一般没有beta</b>：

$$
y = \frac {x} {\sqrt{(variance + variance\_eplison)}} * gamma
$$
其中，$variance = \frac {1} {N} \sum^{i}_N{x^2}$

与LayernNorm的主要区别是：

1.计算方差时不减去均值

2.一般没有beta

该算子计算量比layernorm算子更小, 能够提升大模型的性能
