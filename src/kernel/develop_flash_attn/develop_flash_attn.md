### 1.背景

由于存储分级，高速缓存容量小，IO快，所以应该尽量使用高速缓存。在GPU中，应该尽量使用SRAM存储计算内容。

基于这个思想，出现了flash attn，主要思想就是将数据存到SRAM中，避免直接访问HBM显存。


由于SRAM空间较小，如果需要计算完整张量，空间肯定不够，这限制了我们必须使用局部化技术计算完整张量。


分析attn计算过程，实际上做矩阵乘法时，比较好运用局部性原理，我们加载小q，小k进行计算即可。但主要限制的是softmax，一般必须要求得完整张量后得出，于是flash attn提出了局部softmax。

### 2.局部softmax
目的是需要建立一个公式，通过局部的归一化值得到全局的归一化值。

softmax公式如下：
$$
\text{softmax}(x_i - x_{max}) = \frac{e^{x_i - x_{max}}}{\sum_{j=1}^{N} e^{x_j - x_{max}}}
$$
减去$x_{max}$是担心指数爆炸
设前N个元素的分母为：
$$
l_{pre}: \sum_{j=1}^{N} e^{x_j - x_{pre}} 
$$
前N个元素最大值为：$m_{pre}$, 前N+1个元素最大值为:$m_{cur}$,分母为：
$$
l_{cur} = \sum_{j=1}^{N+1} e^{x_j - x_{last\_max}} = l_{pre} +  e^{x_{N+1} - x_{cur}}
$$
可知，前N个元素中，第i个元素$p_i$：
$$
p_i = e^{x_i - m_{pre}} / l_{pre}
$$
当引入第N+1个元素时，$p_i$需要更新为:
$$
p_i = p_i * l_{pre} / l_{cur}
$$
这样就使局部归一化值更新为全局归一化值。

代码在文件[tile_softmax.py](./tile_softmax.py)中：
