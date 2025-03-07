import torch


class Qwen2PostLayerWeight:
    def __init__(self):
        self.model_type_ = "qwen2.5"
    
    def _init_lm_head(self, weights):
        print(f"weights: {weights.keys()}")
        lm_head_name = f"model.norm.weight"
        self.lm_head = weights[lm_head_name].cuda()
        
    def init(self, weights):
        self._init_lm_head(weights)