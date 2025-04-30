import torch
import torch.nn as nn
from functools import partial
from transformers import AutoModel, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from smooth_quant.llama import QuantizedLlamaForCausalLM
from smooth_quant.qwen2 import QuantizedQwen2ForCausalLM
import argparse
from collections import defaultdict

       
    
def get_scale_max(model, tokenizer, dataset_path, num_samples, max_length):
    
    hook_list = []
    scale_map = {}
    
    def update_scale_max(module, input, output, name):
        if isinstance(input, tuple):
            input = input[0]
            
        hidden_dim = input.shape[-1]
        input = input.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(input, dim=0)[0].float().cpu()
        if name in scale_map:
            scale_map[name] = torch.max(scale_map[name], comming_max)
        else:
            scale_map[name] = comming_max    
    
    def add_hook(module, name):
        hook_wrapper = partial(update_scale_max, name=name)
        hook_list.append(module.register_forward_hook(hook_wrapper))
    
    # import pdb; pdb.set_trace()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            add_hook(module, name)
    
    def remove_hooks():
        for hook in hook_list:
            hook.remove()
    # import pdb; pdb.set_trace()
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=100)
    for i in tqdm(range(num_samples)):
        sample = dataset[i]
        inputs = tokenizer(sample['text'], max_length=max_length, return_tensors='pt')
        model(**inputs)
    # import pdb; pdb.set_trace()
    remove_hooks()
    
    return scale_map

def smooth_ln_fcs(ln, fcs, input_scales, head_dim, alpha=0.5, do_downsample=False):
    assert isinstance(ln, (Qwen2RMSNorm))
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.size(0) == fc.weight.size(-1) == input_scales.size(0) # input_scales是per token量化
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    device, dtype = fc.weight.device, fc.weight.dtype
    input_scales = input_scales.to(device).to(dtype)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    # import pdb; pdb.set_trace()
    scales = (
        (input_scales.pow(alpha)) / weight_scales.pow(1 - alpha)
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    
    ln.weight.div_(scales)
    if hasattr(ln, 'bias'):
        ln.bias.div_(scales)
    
    
    # def downsample_scale(scales, hidden_size, kv_hidden_size, head_dim):
    #     if hidden_size == kv_hidden_size:
    #         return scales
    #     assert hidden_size % kv_hidden_size == 0
    #     assert scales.size(0) == hidden_size
    #     scales = scales.view(kv_hidden_size //head_dim, hidden_size // kv_hidden_size, head_dim)[:, 0, :].reshape(-1)
    #     return scales
    
    # gqa_scale = downsample_scale(scales, fcs[0].weight.size(0), fcs[1].weight.size(0), head_dim)
    for idx, fc in enumerate(fcs):
        true_scale = scales
        # if idx > 0 and do_downsample :
        #     true_scale = gqa_scale
        print(f"fc weight shape: {fc.weight.shape}, true_scale shape: {true_scale.shape}")
        fc.weight.mul_(true_scale.view(1, -1))

    

@torch.no_grad()
def smooth_qwen2(model, scale_map, head_dim):
    # import pdb;pdb.set_trace()
    for name, module in model.named_modules():
        if isinstance(module, Qwen2DecoderLayer):
            attn_ln = module.input_layernorm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scale_map[name + ".self_attn.q_proj"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, head_dim, do_downsample=True)
            
            ffn_ln = module.post_attention_layernorm
            fcs = [
                module.mlp.gate_proj, 
                module.mlp.up_proj
            ]
            fcs_input_scales = scale_map[name + ".mlp.gate_proj"]
            smooth_ln_fcs(ffn_ln, fcs, fcs_input_scales, head_dim)

def collect_llama_layer_scales(model, act_dict):
    decoder_layer_scales = []
    for idx in range(model.config.num_hidden_layers):
        scale_dict = {}
        scale_dict["attn_input_scale"] = act_dict[
            f"model.layers.{idx}.self_attn.q_proj"]['input'] / 127
        scale_dict["q_output_scale"] = act_dict[
            f"model.layers.{idx}.self_attn.q_proj"]['output'] / 127
        scale_dict["k_output_scale"] = act_dict[
            f"model.layers.{idx}.self_attn.k_proj"]['output'] / 127
        scale_dict["v_output_scale"] = act_dict[
            f"model.layers.{idx}.self_attn.v_proj"]['output'] / 127
        scale_dict["out_input_scale"] = act_dict[
            f"model.layers.{idx}.self_attn.o_proj"]['input'] / 127
        # mlp scales
        scale_dict["gate_input_scale"] = act_dict[
            f"model.layers.{idx}.mlp.gate_proj"]['input'] / 127
        scale_dict["down_input_scale"] = act_dict[
            f"model.layers.{idx}.mlp.down_proj"]["input"] / 127
        decoder_layer_scales.append(scale_dict)
    return decoder_layer_scales

def parse_quant_config(config_path):
    data = {}
    import json
    with open(config_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


@torch.no_grad()
def get_static_decoder_layer_scales(model, tokenizer, dataset_path, num_samples, seq_len, model_type="qwen"):
    # import pdb; pdb.set_trace()
    model.eval()
    device = next(model.parameters()).device
    act_dict = defaultdict(dict)
    
    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x.detach().abs().max().item()
        else:
            act_dict[name]["input"] = max(
                act_dict[name]["input"], x.detach().abs().max().item())
        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y.detach().abs().max().item()
        else:
            act_dict[name]["output"] = max(
                act_dict[name]["output"], y.detach().abs().max().item())
    # import pdb; pdb.set_trace()
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(
                partial(stat_io_hook, name=name)
            ))
            
    pabr = tqdm(range(num_samples))
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=100)
    
    for i in pabr:
        input_ids = tokenizer(dataset[i]["text"], return_tensors="pt",
                              max_length=seq_len, truncation=True).input_ids.to(device)
        model(input_ids)
    
    for hook in hooks:
        hook.remove()
    # import pdb; pdb.set_trace()
    decoder_layer_scales = collect_llama_layer_scales(model, act_dict)
    return decoder_layer_scales, act_dict
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        default='models/llama-13b', help='model path contains weights and config etc')
    parser.add_argument('--quantize-model', action="store_true",
                        help='whether to quant model or not', default=True)
    parser.add_argument('--generate-scale', action="store_true",
                        help='whether to generate scale or not', default=True)
    parser.add_argument('--dataset-path', type=str, default='/home/admin/val.jsonl.zst',
                        help='location of the calibration dataset')
    parser.add_argument('--scale-output', type=str, default='scales/llama-13b',
                        help='where to save the act scales, activate when generating scales')
    parser.add_argument("--scale-input", type=str, default='/home/admin',
                        help='where to save the act scales, activate when quantizing models')
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--type', type=str, default="int8",
                        help='fp8 & fp8_e4m3, fp8_e5m2 or int8, when quant_config.json does not have this '
                             'configuration, this configuration will be used')
    parser.add_argument('--activation-scheme', type=str, default="dynamic", help='dynamic or static, just for fp8')
    parser.add_argument('--ignore-patterns', type=str, default="re:.*lm_head", help='ignore layer, just for fp8')
    parser.add_argument('--seq-len', type=int, default=100)
    parser.add_argument("--model-output", type=str, default='quantized_model/llama-13b',
                        help='where to save the quantized models, activate when quantizing models')
    parser.add_argument("--smooth-strength", type=float, default=0.5,
                        help='migration strength of smoothquant, should be in a range of (0, 1)')
    args = parser.parse_args()
    return args


def qwen_quantize(model_name, dataset_path, num_samples, seq_len, output_path, max_length):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # import pdb; pdb.set_trace()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    scale_map = get_scale_max(model, tokenizer, dataset_path, num_samples, max_length)
    import os
    config_path = os.path.join(model_name, "quant_config.json")
    quant_config = parse_quant_config(config_path)
    smooth_qwen2(model, scale_map, head_dim=128) # 这里强制写死了head_dim
    # import pdb;pdb.set_trace()
    decoder_layer_scales, _ = get_static_decoder_layer_scales(model,
                                                                tokenizer,
                                                                dataset_path,
                                                                num_samples=num_samples,
                                                                seq_len=seq_len,
                                                                model_type="qwen2")
    import pdb;pdb.set_trace()
    quant_model = QuantizedQwen2ForCausalLM.from_float_to_int8(model, decoder_layer_scales, quant_config)
    import pdb;pdb.set_trace()
    quant_model.save_pretrained(output_path, safe_serialization=False)


def main():
    args = parse_args()
    qwen_quantize(args.model_path, args.dataset_path, args.num_samples, args.seq_len, args.model_output, args.seq_len)
    
main()