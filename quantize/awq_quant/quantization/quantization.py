import torch
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
from functools import partial
import torch.nn as nn
from awq_quant.linear.wqlinear import WQLinear
from transformers import AwqConfig

class AutoQuantizer:
    def __init__(self, model, dataset_path, tokenizer, input_len, num_samples, per_forward_batch_num, split, output_path, zero_point, group_size):
        self.model_ = model
        self.dataset_path_ = dataset_path
        self.tokenizer_ = tokenizer
        self.input_len_ = input_len
        self.num_samples_ = num_samples
        self.split_ = split
        self.inps_ = None
        
        self.per_forward_batch_num_ = per_forward_batch_num
        
        self.all_layers_, self.layer_kwargs_, self.inps_ = self.init_quant()
        
        self.w_bits = 4
        self.zero_point = zero_point
        self.output_path_ = output_path
        self.group_size_ = group_size
    
    def get_calib_data(self):
        dataset = load_dataset("json", data_files=self.dataset_path_, split=self.split_)
        dataset = dataset.shuffle(seed=42)
        all_samples = []
        for i in tqdm(range(self.num_samples_)):
            sample = dataset[i]
            input = self.tokenizer_(sample['text'], max_length=self.input_len_, return_tensors='pt')
            all_samples.append(input['input_ids'])
        all_samples_tensor = torch.cat(all_samples, dim=1)
        
        # 将输入等分
        n_split = all_samples_tensor.size(1) // self.input_len_
        all_samples_tensor = all_samples_tensor[:, :n_split * self.input_len_].view(n_split, self.input_len_)
        
        return all_samples_tensor
    
    # 获取模型每层的对象，获取第0层的输入，
    def init_quant(self):
        
        # 逐层量化，先获取到每个层的对象
        all_layers = self.model_.model.layers
        self.model_.model.embed_tokens .cuda()
        # 取所有请求后
        all_samples_tensor = self.get_calib_data()
        # import pdb; pdb.set_trace()
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
    
        all_layers[0] = Catcher(all_layers[0])
        try:
            self.model_(all_samples_tensor.to(next(self.model_.parameters()).device))
        except ValueError:  # work with early exit
            pass
        layer_kwargs = all_layers[0].layer_kwargs
        inps = all_layers[0].inps
        all_layers[0] = all_layers[0].layer
        self.model_.model.embed_tokens .to('cpu')
        # layer_kwargs = self.model_.prepare_inputs_for_generation(all_samples_tensor, **layer_kwargs)
        # import pdb; pdb.set_trace()
        # del layer_kwargs['input_ids']
        # import pdb; pdb.set_trace()
        # 查看layr_kwargs的数据结构
        return all_layers, layer_kwargs, inps
    
    def quantization(self):
        for idx in tqdm(range(len(self.all_layers_))):
            layer = self.all_layers_[idx]
            # 获取该层真实输入
            feat_map = self._get_input_feat(layer)
            # 使用smooth_quant的方法，迁移量化难度，并找到最佳的alpha和scale
            self._search_best_scale(layer, feat_map)
            self._search_best_clip(layer, feat_map)
            # import pdb; pdb.set_trace()
            self._apply_quant(layer, feat_map)
            # import pdb; pdb.set_trace()
            pass
            
    def _apply_quant(self, layer, feat_map):
        linear_message_list = self._get_should_quant_linears(layer, feat_map)
        for name, linear in layer.named_modules():
            if isinstance(linear, nn.Linear):
                quant_weight, scales, zeros = self._mock_quantize(linear.weight)
                # import pdb; pdb.set_trace()
                scales = scales.t().contiguous()
                zeros = zeros.t().contiguous()
                new_linear = WQLinear.create_wqlinear(linear=linear, scales=scales, zeros=zeros, 
                                        in_features=linear.in_features, group_size=self.group_size_,
                                        out_features=linear.out_features, n_bits=4)
                new_linear = new_linear.to(next(linear.parameters()).device)
                self._set_op_by_name(layer, name, new_linear)
            
    def _set_op_by_name(self, layer, name, new_linear):
        levels = name.split('.')
        module = layer
        # 沿着层级往下找，找到倒数第二个层级
        for l_idx in range(len(levels) - 1):
            current_level = levels[l_idx]
            module = getattr(module, current_level)
        setattr(module, levels[-1], new_linear)

    def _get_input_feat(self, layer):
        feat_map = defaultdict(dict)
        def hook_fn(module, input, output, name):
            if isinstance(input, tuple):
                input = input[0]
            if name not in feat_map:
                feat_map[name] = [input,]
            else:
                feat_map[name].append(input)
        
        hooks = []
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                hook_fn_wrapper = partial(hook_fn, name=name)
                hooks.append(module.register_forward_hook(hook_fn_wrapper))
        layer = layer.cuda()
        self.inps_ = self._module_forward(layer, self.inps_, self.layer_kwargs_)
        layer = layer.to('cpu')
        for hook in hooks:
            hook.remove()
            
        for key in feat_map.keys():
            feat_map[key] = torch.cat(feat_map[key], dim=0)
        return feat_map
    
    def _module_forward(self, layer, inps, module_kwargs):
        if self.per_forward_batch_num_ is None:
            return layer(inps, **module_kwargs)[0]
        else:
            output_list = []
            split_tensors = torch.split(inps, self.per_forward_batch_num_, dim=0)
            for tensor in split_tensors:
                output_list.append(layer(tensor, **module_kwargs)[0])
            return torch.cat(output_list, dim=0)
    
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
    
    def _search_best_scale(self, layer, feat_map):
        linear_message_list = self._get_should_quant_linears(layer, feat_map)
        for linear_message in linear_message_list:
            weights = [module.weight for module in linear_message['linears']]
            w_gather = torch.cat(weights, dim=0) # [c_o * n, c_i]
            org_w_shape = w_gather.shape
            w_gather = w_gather.view(-1, self.group_size_)
            w_scale = w_gather.abs() / (w_gather.abs().amax(dim=1, keepdim=True) + 1e-6) # [c_o * n, c_i], 防止出现极端值
            w_scale = w_scale.view(org_w_shape)
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
            # 计算原始输出
            with torch.no_grad():
                linear_message['module'].cuda()
                module_kwargs = self._sanitize_kwargs(self.layer_kwargs_, linear_message['module'])
                original_output = self._module_forward(linear_message['module'], linear_message['inp'], module_kwargs)
                linear_message['module'].cpu()
                original_output = original_output.clip(torch.finfo(original_type).min, torch.finfo(original_type).max)
            best_scales = self._compute_best_scales(original_output, linear_message, w_mean, a_mean, module_kwargs)
            self._apply_scales(linear_message, best_scales, feat_map)
    
    def _apply_scales(self, linear_message, best_scales, feat_map):
        
        # TODO(gaolingxiao)： 可以能还需要调整feat_map中的输入
        
        if isinstance(linear_message['prev_ops'], nn.Linear):
            # 这里的linear_message['prev_ops'].weight是已经经过量化了，还需要将激活量化难度迁移到权重上来
            # import pdb; pdb.set_trace()
            linear_message['prev_ops'].weight.data = torch.div(linear_message['prev_ops'].weight, best_scales.view(-1, 1))
            if hasattr(linear_message['prev_ops'], 'bias') and linear_message['prev_ops'].bias is not None:
                linear_message['prev_ops'].bias.data = torch.div(linear_message['prev_ops'].bias, best_scales)
            for linear in linear_message['linears']:
                linear.weight.data = torch.mul(linear.weight, best_scales.view(1, -1))
        else: 
            
            for linear in linear_message['linears']:
                linear.weight.data = torch.mul(linear.weight, best_scales)
            linear_message['prev_ops'].weight.data = torch.div(linear_message['prev_ops'].weight, best_scales.squeeze(0))
            if hasattr(linear_message['prev_ops'], 'bias') and linear_message['prev_ops'].bias is not None:
                linear_message['prev_ops'].bias.data = torch.div(linear_message['prev_ops'].bias, best_scales.squeeze(0))
        
        # 因为前面的prev_ops_除了best_scales，这里也应该除以，才是真实场景
        for layer_name in linear_message['names']:
            if layer_name in feat_map:
                feat_map[layer_name] = torch.div(feat_map[layer_name].cuda(), best_scales.view(1, -1).cuda())
        
    # 对称量化
    def _mock_quantize(self, weight):
        weight_shape = weight.shape
        
        if self.group_size_ > 0:
            assert weight_shape[-1] % self.group_size_ == 0, f"org_w_shape ({weight_shape[-1]}) must be a multiple of group_size ({self.group_size_})!"
            weight = weight.reshape(-1, self.group_size_)
        
        if self.zero_point:
            max_val = weight.amax(dim=1, keepdim=True)
            min_val = weight.amin(dim=1, keepdim=True)
            max_int = 2 ** self.w_bits - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            weight = (
                torch.clamp(torch.round(weight / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(weight_shape[0], -1)
        else:
            # 后期改动后，这一块没有测试
            max = weight.abs().amax(dim=1, keepdim=True)
            int_max = 2 ** (self.w_bits - 1) - 1
            # int_min = -2 ** self.w_bits
            scales = int_max / max
            weight = torch.round(weight / scales) * scales # 伪量化，需要还原
            zeros = None
        scales = scales.view(weight_shape[0], -1)
        weight = weight.view(weight_shape)
        return weight, scales, zeros
    
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

    def _compute_loss(self, original_output, after_quantize_output):
        original_output = original_output.view(-1)
        after_quantize_output = after_quantize_output.view(-1)
        
        loss = (original_output - after_quantize_output).to(torch.float32).pow(2).sum().item()
        return loss

    # 需要过滤一遍输入，否则有时会报错
    def _sanitize_kwargs(self, inputs_kwargs, module):
        import inspect
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs

    def _search_best_clip(self, layer, feat_map):
        
        avoid_match = ["q_", "k_"]
        should_process_linear = {}
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                flag = False
                for avoid_str in avoid_match:
                    if avoid_str in name:
                        flag = True
                if not flag:
                    should_process_linear[name] = (module)
        # import pdb; pdb.set_trace()
        clip_map = self._find_best_clip(should_process_linear, feat_map)
        self._apply_clip(should_process_linear, clip_map)
    
    @torch.no_grad()
    def _apply_clip(self, should_process_linear, clip_map):
        # import pdb; pdb.set_trace()
        for name, linear in should_process_linear.items():
            assert name in clip_map, f"{name} not in clip_map"
            org_shape = linear.weight.shape
            linear.weight.data = linear.weight.data.reshape(*clip_map[name].shape[:2], -1)
            linear.weight.data = torch.clamp(linear.weight.data, -clip_map[name], clip_map[name])
            linear.weight.data = linear.weight.data.reshape(org_shape)
        
    @torch.no_grad()
    def _find_best_clip(self, should_process_linear, feat_map, n_sample_token=512):
        clip_map = {}
        # import pdb; pdb.set_trace()
        for name, linear in should_process_linear.items():

            w = linear.weight
            w_shape = w.shape
            input_feat = feat_map[name]
            
            group_size = self.group_size_ if self.group_size_ > 0 else w_shape[1]
            input_feat = input_feat.view(-1, input_feat.size(-1))  # [n_token, ci]
            input_feat = input_feat.reshape(1, input_feat.size(0), -1, group_size)  # [1, n_token, n_group, group_size]
            
            # 下采样输入特征（与_compute_best_clip相同逻辑）
            step_size = max(1, input_feat.size(1) // n_sample_token)
            input_feat = input_feat[:, ::step_size]  # [1, n_sample, n_group, group_size]
            
            w = w.view(w.size(0), 1, -1, group_size)  # [co, 1, n_group, group_size]
            
            device = w.device
            input_feat = input_feat.to(device)
            
            org_out = (input_feat * w).sum(dim=-1)  # [co, n_sample, n_group]
            
            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # [co, 1, n_group, 1]
            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            
            n_grid = 20
            max_shrink = 0.5
            for i_s in range(int(max_shrink * n_grid)):
                ratio = (1 - i_s / n_grid)
                max_val = org_max_val * ratio
                min_val = -max_val
                
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self._mock_quantize(cur_w)[0]
                cur_out = (input_feat * q_w).sum(dim=-1)
                
                err = (cur_out - org_out).pow(2).mean(dim=1, keepdim=True).view(min_errs.shape)  # [co, 1, n_group] -> [co, 1, n_group, 1]
                
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
                print(f"min_errs: {min_errs.shape}")
            # import pdb; pdb.set_trace()
            print(f"best_max_val: {best_max_val.shape}")
            clip_map[name] = best_max_val.squeeze(1)
            print(f"clip_map[name]: {clip_map[name].shape}")    
        # import pdb; pdb.set_trace()
        return clip_map

    def save_models(self):
        self.model_.config.quantization_config = AwqConfig(
            bits=self.w_bits,
            group_size=self.group_size_,
            modules_to_not_convert=None,
            zero_point=self.zero_point,
            version="gemm"
        )
        self.model_.save_pretrained(self.output_path_)


# 冒似必须要用group_size
