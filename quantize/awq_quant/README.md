[TOC]

### 1.背景

大模型太大，仅权重占用显存就非常大，一般会使用量化技术，在不影响精度的情况下，尽量减少显存占用。

大语言模型量化存在一些难题，常规的量化会导致精度大幅下降。smooth_quant算法提供了比较可用的量化思路，实现了大语言模型的w8a8量化，他的主要观点如下：

1.激活值比权重更难量化，需要把激活值的量化难度迁移到权重上；
2.权重和激活值要使用per-channel/per-token的方式量化，因为激活的离群值总是出现在特定通道。

在awq论文以及后续的其他量化相关论文中，进一步发现，迁移量化难度后，可能会导致权重出现离群值，同时，需要更精确地计算迁移多少量化难度，以下是其思路：

1.找到一个alpha，迁移量化难度进行量化，与真实值计算均方误差，找一个最小的alpha。（此处alpha是$\frac {x_{act}^{\alpha}}{x_{weight} ^ {1 - \alpha}}$)

2.找到一个beta，逐通道降低该通道权重的最大值，与真实值计算逐通道的均方误差。

### 2.awq量化实现

#### 2.1 初始化

首先需要获取模型的各个layer，这个比较容易实现，直接查看named_module即可，同时我们还需要获取第一个layer的输入，可以通过构造一个catcher类，替换模型的第一个layer对象，将输入保存起来：

```
class Catcher(nn.Module):
    def __init__(self, layer):
        super(Catcher, self).__init__()
        self.layer = layer
        self.layer_kwargs = {}
        self.inps = None
        
    def forward(self, *args, **kwargs):
        if len(args) > 0:
            hidden_states = args[0]
            del args
        else:
            first_key = list(kwargs.keys())[0]
            hidden_states = kwargs[first_key]
        self.inps = hidden_states
        self.layer_kwargs.update(kwargs)
        raise ValueError
```

同时，我们一般只量化linear层，需要将这些层的输入给保存起来，方便后续计算。
这里获取到需要量化的linear后，使用register_forward_hook, 将输入保存起来：

```
def _get_should_quant_linears(self, layer, feat_map):
    
    # 不量化out_proj层，对性能影响很小
    linear_message_list = []
    # 得确定下feat_map的key值,还有layer的名字
    first_linears_message = dict(
        inp=feat_map['self_attn.q_proj'],
        linears=[layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        names=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
        module=layer.self_attn,
        prev_ops=layer.input_layernorm,
        
    )
    
    second_linears_message = dict(
        inp=feat_map['mlp.gate_proj'],
        linears=[layer.mlp.gate_proj, layer.mlp.up_proj],
        names=['mlp.gate_proj', 'mlp.up_proj'],
        module=layer.mlp,
        prev_ops=layer.post_attention_layernorm,
    )
    
    third_linears_message = dict(
        inp=feat_map['mlp.down_proj'],
        linears=[layer.mlp.down_proj,],
        names=['mlp.down_proj',],
        module=layer.mlp.down_proj,
        prev_ops=layer.mlp.up_proj,
    )
    linear_message_list.append(first_linears_message)
    linear_message_list.append(second_linears_message)
    linear_message_list.append(third_linears_message)
    return linear_message_list

```

#### 2.2 计算scales

一般根据linear前面的算子将量化分为两种类型，一种是fc-fc量化，一种是ln-fc量化，其实两种没啥区别，但有时候fc-fc量化前面会跟着多个矩阵乘再加上逐元素乘法，所以fc-fc通常只会选一个full connection做除法操作，其代码如下：

```
def _apply_scales(self, linear_message, best_scales):
    if isinstance(linear_message['prev_ops'], nn.Linear):
        # import pdb; pdb.set_trace()
        linear_message['prev_ops'].weight.data = torch.div(linear_message['prev_ops'].weight, best_scales.view(-1, 1))
        if hasattr(linear_message['prev_ops'], 'bias') and linear_message['prev_ops'].bias is not None:
            linear_message['prev_ops'].bias.data = torch.div(linear_message['prev_ops'].bias, best_scales)
        for linear in linear_message['linears']:
            linear.weight.data = torch.mul(linear.weight, best_scales.view(1, -1))
    else: 
        for linear in linear_message['linears']:
            linear.weight.data = torch.mul(linear.weight, best_scales)
        linear_message['prev_ops'].weight.data = torch.div(linear_message['prev_ops'].weight, best_scales)
        if hasattr(linear_message['prev_ops'], 'bias') and linear_message['prev_ops'].bias is not None:
            linear_message['prev_ops'].bias.data = torch.div(linear_message['prev_ops'].bias, best_scales)


```

首先计算权重和激活的均值，这里与smooth_quant算法有些异同，smooth_quant算法这里直接取最大值，而awq算法这里取权重和激活的均值，并且权重是先归一化再求均值。

```
weights = [module.weight for module in linear_message['linears']]
w_gather = torch.cat(weights, dim=0) # [c_o * n, c_i]
w_scale = w_gather.abs() / (w_gather.abs().amax(dim=1, keepdim=True) + 1e-6) # [c_o * n, c_i], 防止出现极端值
w_mean = w_scale.mean(dim=0) # [c_i], per_channel

inp = linear_message['inp']
# 希望inp的shape是[batch_size * seq_len, c_i]
original_type = inp.dtype
inp = inp.abs()
inp = inp.cpu().view(-1, inp.size(-1))
inp = inp.to(torch.float32)
a_sum = inp.sum(dim=0) # [c_i], per_token

a_mean = a_sum / inp.size(0)
a_mean = a_mean.to(original_type)
inp = inp.to(original_type)

```
awq论文中说，存在1%的权重，对精度影响很大，需要使用更仔细的量化方法寻找scales。我们这里对于所有的权重，都是用这种方法。

本质上就是根据划定n_grid=20,设置alpha=0, 0.05, 0.1, ..., 0.95等参数，找一个最小MSE。其代码如下：

```
def _compute_best_scales(self, original_output, linear_message, w_mean, a_mean, module_kwargs):
    
    n_grid = 20
    best_loss = float('inf')
    best_scales = None
    for i in range(n_grid):
        
        ratio = i / n_grid
        # import pdb; pdb.set_trace()
        scales = a_mean.pow(ratio) / w_mean.pow(1 - ratio).clamp(1e-4)
        scales = scales / (scales.max() * scales.min()).sqrt() # 很奇怪的归一化
        scales = scales.view(1, -1).cuda()
        
        scales[torch.isinf(scales)] = 1
        scales[torch.isnan(scales)] = 1
        # import pdb; pdb.set_trace()
        original_weights = [linear.weight.clone().cpu() for linear in linear_message['linears']]
        for linear in linear_message['linears']:
            weight = torch.mul(linear.weight.cuda(), scales.cuda())
            linear.weight.data = (self._mock_quantize(weight)[0].cuda() / scales.cuda())
        inp_dtype = linear_message['inp'].dtype
        linear_message['module'] = linear_message['module'].cuda()
        after_quantize_output = self._module_forward(linear_message['module'], linear_message['inp'], module_kwargs)
        linear_message['module'] = linear_message['module'].cpu()
        after_quantize_output.clip(torch.finfo(inp_dtype).min, torch.finfo(inp_dtype).max)
        loss = self._compute_loss(original_output, after_quantize_output)
        print(f"loss: {loss}")

        if loss < best_loss:
            best_loss = loss
            best_scales = scales.detach().clone().cpu()
        
        # 恢复原始权重
        for idx, linear in enumerate(linear_message['linears']):
            linear.weight.data = original_weights[idx]
    # import pdb; pdb.set_trace()
    return best_scales

```
找到最佳scales后，根据ln-fcs和fc-fcs调整对应权重

#### 2.3 计算clip

由于权重中可能存在某些极大值，尝试缩小最大值，这里同样使用类似于scales的方法，设置n_clip_grid=10，将最大值缩小为0.1*max_val, ..., 0.9*max_val, 1*max_val

注意，实验发现对q和k应用clip，效果较差，所以会跳过q和k的clip过程

```
@torch.no_grad()
def _find_best_clip(self, should_process_linear, feat_map, n_sample_token=512):
    clip_map = {}
    
    for name, linear in should_process_linear.items():

        w = linear.weight
        input_feat = feat_map[name]
        
        group_size = w.size(1) 
        input_feat = input_feat.view(-1, input_feat.size(-1))  # [n_token, ci]
        input_feat = input_feat.reshape(1, input_feat.size(0), 1, group_size)  # [1, n_token, 1, ci]
        
        # 下采样输入特征（与_compute_best_clip相同逻辑）
        step_size = max(1, input_feat.size(1) // n_sample_token)
        input_feat = input_feat[:, ::step_size]  # [1, n_sample, 1, ci]
        
        w = w.view(w.size(0), 1, 1, group_size)  # [co, 1, 1, ci]
        
        device = w.device
        input_feat = input_feat.to(device)
        
        org_out = (input_feat * w).sum(dim=-1)  # [co, n_sample, 1]
        
        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # [co, 1, 1, 1]
        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        
        clip_n_grid = 10 
        for i_s in range(clip_n_grid):
            ratio = (i_s + 1) / clip_n_grid
            max_val = org_max_val * ratio
            min_val = -max_val
            
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = self._mock_quantize(cur_w)[0]
            cur_out = (input_feat * q_w).sum(dim=-1)
            
            err = (cur_out - org_out).pow(2).mean(dim=1, keepdim=True)  # [co, 1, 1]
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        
        clip_map[name] = best_max_val.view(-1) 
        
    return clip_map

```

至此，已经实现了权重的量化，但每个权重仍然表示为fp16的格式。

#### 2.4 量化权重存储

量化为int格式后，通过位运算，将8个int4的数合并为一个int32的数，并存储：

```
def transformer_weight_to_save_format(weight, n_bit):
    assert n_bit == 4, "n_bits shoulde equal 4"
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    pack_num = 32 // n_bit
    quant_weights = []
    for i in range(0, weight.size(1), pack_num):
        quant_weight = torch.zeros(weight.size(0), dtype=torch.int32)
        for j in range(pack_num):
            channel = weight[:, i + j]
            channel = channel << order_map[j]
            quant_weight = quant_weight | channel
        quant_weights.append(quant_weight)
    quant_weights_tensor = torch.cat(quant_weights, dim=0)
    return quant_weights_tensor
```