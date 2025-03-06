import torch
from layer import Qwen2TransformerLayer, Qwen2PreLayer, Qwen2PostLayer

class Qwen2Model:
    
    def __init__(self, layer_nums, max_req_length, hidden_size):
        self.layers = []
        for i in range(layer_nums):
            self.layers.append(Qwen2TransformerLayer(i))
        
        self.pre_layer = Qwen2PreLayer(max_req_length, hidden_size)
        
        self.post_layer = Qwen2PostLayer()
        self.position_embeddings = self.pre_layer.position_embeddings

    def forward(self, input_tokens):
        hidden_states = self.pre_layer.forward(input_tokens)
        for i in self.layers:
            hidden_states = self.layers.forward(hidden_states, self.position_embeddings)
        output_tokens = self.post_layer.forward(input_tokens)
        return output_tokens

    def load_weight(self, model_path):
        from safetensors.torch import load_file

        data_dict = load_file(model_path, device="cpu")
        self.pre_layer.pre_layer_weight.init(data_dict)
        for layer in self.layers:
            layer.layer_weight.init(data_dict)
        self.post_layer.post_layer_weight.init(data_dict)

class Qwen2Config:
    
    def __init__(self, config_path):
        import json
        
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file) 
        
        self.layer_nums = config['num_hidden_layers']
        self.hidden_size = config['hidden_size']