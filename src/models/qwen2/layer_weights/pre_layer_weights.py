from safetensors import safe_open
import torch


class Qwen2PreLayerWeight:
    def __init__(self):
        self.model_type_ = "qwen2.5"
    
    def _init_input_embds(self, weights):
        input_embds_name = f"model.embed_tokens.weight"
        self.input_embds = weights[input_embds_name].cuda().to(torch.float32)
        
    def init(self, weights):
        self._init_input_embds(weights)