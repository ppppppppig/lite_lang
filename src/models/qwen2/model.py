import torch
from .layer import Qwen2TransformerLayer, Qwen2PreLayer, Qwen2PostLayer
from .cache import Cache, NormalCache
from typing import Optional

class Qwen2Config:
    
    def __init__(self, config_path):
        import json
        
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file) 
        
        self.layer_nums = config['num_hidden_layers']
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.hidden_size = config['hidden_size']
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config['num_key_value_heads']
        self.tie_word_embeddings = config['tie_word_embeddings']

class Qwen2Model:
    
    def __init__(self, layer_nums, max_req_length, config: Qwen2Config):
        self.layers = []
        for i in range(layer_nums):
            self.layers.append(Qwen2TransformerLayer(i, config.num_heads, config.head_dim, config.num_key_value_heads))
        self.layer_nums_ = layer_nums
        self.pre_layer = Qwen2PreLayer(max_req_length, config.hidden_size, config.num_heads, config.head_dim)
        
        self.post_layer = Qwen2PostLayer(config.tie_word_embeddings)
        self.position_embeddings = self.pre_layer.position_embeddings

    def forward(self, model_inputs):
        if model_inputs.past_key_values is None:
            model_inputs.past_key_values = NormalCache(self.layer_nums_)
        
        if model_inputs.is_prefill:
            input_tokens = model_inputs.input_tokens
        else:
            input_tokens = model_inputs.output_tokens[:, -1][:, None]
        hidden_states = self.pre_layer.Forward(input_tokens)
        for layer in self.layers:
            hidden_states = layer.Forward(hidden_states, self.position_embeddings, model_inputs.padding_mask, not model_inputs.is_prefill, model_inputs.past_key_values)
        output_tokens = self.post_layer.Forward(hidden_states, model_inputs)
        return output_tokens
    
    def load_weight(self, model_path):
        from safetensors.torch import load_file

        # data_dict = load_file(model_path, device="cpu")
        # 这个读取权重的写的不是很好，后面要细化一下
        import os
        weight_path = os.path.join(model_path, "model.safetensors")
        
        data_dict = load_file(weight_path)
        self.pre_layer.pre_layer_weight.init(data_dict)
        for layer in self.layers:
            layer.layer_weight.init(data_dict)
        self.post_layer.post_layer_weight.init(data_dict)
