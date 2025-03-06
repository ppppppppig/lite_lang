import torch
from ..layer_weights.pre_layer_weights import Qwen2PreLayerWeight

class Qwen2PreLayer:
    
    def __init__(self, max_req_length, hidden_size):
        self.hidden_size_ = hidden_size
        self.max_req_length_ = max_req_length
        self.pre_layer_weight = Qwen2PreLayerWeight()
        self.position_embeddings = self._ComputePosition()
        
    def forward(self, input_tokens):
        hidden_states = self.pre_layer_weight.input_embds[input_tokens]
        return hidden_states, self.position_embeddings
        
    def _ComputePosition(self):
        theta = 10000
        freqs = 1.0 / theta ** (torch.arange(0, self.hidden_size_, 2) / self.hidden_size_)
        pos = torch.arange(0, self.max_req_length_)
        
        angel = pos[:, None] * freqs[None, :]
        
        cos = torch.cos(angel)
        sin = torch.sin(angel)
        return (cos, sin)