import torch
from .layer import Qwen2TransformerLayer, Qwen2PreLayer, Qwen2PostLayer
from .cache import Cache, PageCache
from typing import Optional
from safetensors import safe_open

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
    
    def __init__(self, layer_nums, max_length, config: Qwen2Config, tp_rank, world_size):
        self.layers = []
        for i in range(layer_nums):
            self.layers.append(Qwen2TransformerLayer(i, config.num_heads, config.head_dim, config.num_key_value_heads, tp_rank, world_size))
        self.layer_nums_ = layer_nums
        self.pre_layer = Qwen2PreLayer(max_length, config.hidden_size, config.num_heads, config.head_dim)
        
        self.post_layer = Qwen2PostLayer(config.tie_word_embeddings)
        self.position_embeddings = self.pre_layer.position_embeddings

    def forward(self, model_inputs, kv_cache):
        if model_inputs.is_prefill:
            input_tokens = model_inputs.input_tokens
        else:
            input_tokens = model_inputs.output_token_ids
        hidden_states = self.pre_layer.Forward(input_tokens)
        for layer in self.layers:
            hidden_states = layer.Forward(hidden_states, self.position_embeddings, model_inputs, kv_cache)
        output_tokens = self.post_layer.Forward(hidden_states, model_inputs)
        return output_tokens
    
    def load_weight(self, model_path):
        # data_dict = load_file(model_path, device="cpu")
        # 这个读取权重的写的不是很好，后面要细化一下
        import os
        import json
        index_json_path = None
        if os.path.exists(os.path.join(model_path, "pytorch_model.bin.index.json")):
            index_json_path = os.path.join(model_path, "pytorch_model.bin.index.json")
        if os.path.exists(os.path.join(model_path, "pytorch_model.safetensors.index.json")):
            index_json_path = os.path.join(model_path, "pytorch_model.safetensors.index.json")
        state_dict = {}
        if index_json_path is not None:       
            with open(index_json_path, "r") as f:
                index = json.load(f)
            
            have_loaded_weight = set()
            for key, value in index['weight_map'].items():
                if value in have_loaded_weight:
                    continue
                with safe_open(value, framework="pt") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
        else:
            model_path = os.path.join(model_path, "model.safetensors")
            with safe_open(model_path, framework="pt") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        self.pre_layer.pre_layer_weight.init(state_dict)
        for layer in self.layers:
            layer.layer_weight.init(state_dict)
        self.post_layer.post_layer_weight.init(state_dict)

# 目前只支持qwen2，暂时只实现qwen2模型的Runner
class Qwen2ModelRunner:
    def __init__(self, layer_nums, max_batch_size, config: Qwen2Config, **runner_kwargs):
        self.tp_rank_ = runner_kwargs['tp_rank']
        self.world_size_ = runner_kwargs['world_size']
        self.model_ins_ = Qwen2Model(layer_nums, runner_kwargs['max_length'], config, self.tp_rank_, self.world_size_)
        self.model_path_ = runner_kwargs['model_path']
        self.max_batch_size_ = max_batch_size
        self.max_length_ = runner_kwargs['max_length']
        self.mem_usage_ = runner_kwargs['mem_usage']
        self.layer_nums_ = layer_nums
        self.config_ = config

    def forward(self, model_inputs):
        if model_inputs.is_prefill:
            for req in model_inputs.reqs:
                req.rid = self._add_new_req(req.input_length)
                if req.rid is None:
                    return False
        return self.model_ins_.forward(model_inputs, self.kv_cache_)
        
    def _add_new_req(self, length):
        if self.kv_cache_.can_allocated(length):
            return self.kv_cache_.alloc_req()
        else:
            return None
        
    def free_all(self, reqs):
        self.kv_cache_.dealloc_reqs(reqs)
    
    def init_model(self):
        self.model_ins_.load_weight(self.model_path_)
        self.kv_cache_ = PageCache(self.max_batch_size_, self.max_length_, self.mem_usage_, self.layer_nums_, self.config_.head_dim, self.config_.num_key_value_heads, torch_dtype=torch.float16, tp_rank=self.tp_rank_, tp_world_size=self.world_size_)
        return self.kv_cache_.kv_max_size_