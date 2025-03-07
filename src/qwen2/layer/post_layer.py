import torch
from ..layer_weights.post_layer_weights import Qwen2PostLayerWeight

class Qwen2PostLayer:
    
    def __init__(self):
        self.post_layer_weight = Qwen2PostLayerWeight()
        
    def forward(self, hidden_states):
        logits = hidden_states @ self.post_layer_weight.lm_head
        return self._PostProcess(logits)
        
    def _PostProcess(self, logits):
        output_tokens = torch.argmax(logits, dim=-1)
        return output_tokens