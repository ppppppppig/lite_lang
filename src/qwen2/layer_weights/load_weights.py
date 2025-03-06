from safetensors import safe_open
import torch

class MMWeight:
    def __init__(self, weights, proj_name, bias_name):
        self.proj = weights[proj_name].cuda()
        self.bias = weights[proj_name].cuda()
        
class LayerNormWeight:
    def __init__(self, weights, proj_name, bias_name):
        self.proj = weights[proj_name].cuda()
        self.bias = weights[proj_name].cuda()

class Qwen2LayerWeight:
    def __init__(self, layer_num, device):
        self.layer_num_ = layer_num
        self.device_ = device
        self.model_type_ = "qwen2.5"
        
    def _init_qkv(self, weights):
        q_proj_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"
        q_bias_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.bias"
        
        k_proj_name = f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"
        k_bias_name = f"model.layers.{self.layer_num_}.self_attn.k_proj.bias"
        
        v_proj_name = f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"
        v_bias_name = f"model.layers.{self.layer_num_}.self_attn.v_proj.bias"
        
        o_proj_name = f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"
        o_bias_name = f"model.layers.{self.layer_num_}.self_attn.o_proj.bias"
        
        qkv_keys = []
        qkv_keys.append(q_proj_name, q_bias_name, k_proj_name, k_bias_name, v_proj_name, v_bias_name)
        
        self.q_proj = MMWeight(weights, q_proj_name, q_bias_name, device=self.device_)
        self.k_proj = MMWeight(weights, k_proj_name, k_bias_name, device=self.device_)
        self.v_proj = MMWeight(weights, v_proj_name, v_bias_name, device=self.device_)
        self.v_proj = MMWeight(weights, o_proj_name, o_bias_name, device=self.device_)
        
    def _init_input_layernorm(self, weights):
        layernorm_weight = f"model.layers.{self.layer_num_}.input_layernorm.weight"
        layernorm_bias = None
        
        self.layernorm = LayerNormWeight(weights, layernorm_weight, layernorm_bias)
        
    def _init_post_attn_layernorm(self, weights):
        layernorm_weight = f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"
        layernorm_bias = None
        
        self.post_attn_layernorm = LayerNormWeight(weights, layernorm_weight, layernorm_bias)
        
    def _init_ffn(self, weights):
        gate_weight = f"model.layers.{self.layer_num_}.mlp.gate_proj.weight"
        down_weight = f"model.layers.{self.layer_num_}.mlp.dodwn_proj.weight"
        up_weight = f"model.layers.{self.layer_num_}.mlp.up_proj.weight"
        
        self.down_proj = weights[down_weight].cuda()
        self.gate_proj = weights[gate_weight].cuda()
        self.up_proj = weights[up_weight].cuda()
    
    def init(self, weights):
        self._init_ffn(weights)
        self._init_post_attn_layernorm(weights)
        self._init_input_layernorm(weights)
        self._init_qkv(weights)
        