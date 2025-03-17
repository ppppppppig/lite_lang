import torch
from ..layer_weights.post_layer_weights import Qwen2PostLayerWeight

class Qwen2PostLayer:
    
    def __init__(self, tie_wording_embedding):
        self.post_layer_weight = Qwen2PostLayerWeight(tie_wording_embedding)

        self.eps = 1e-6
        
    def Forward(self, hidden_states):

        will_compute_hidden_states = hidden_states[:, -1, :][:, None, :]
        will_compute_hidden_states = self._InputLayernorm(will_compute_hidden_states)
        logits = will_compute_hidden_states @ self.post_layer_weight.lm_head.transpose(-2, -1)
        return self._PostProcess(logits)

    def _InputLayernorm(self, input):
        denominator = torch.sqrt(torch.sum(input * input, dim=-1) / input.size(-1) + self.eps)
        res = input / denominator[..., None] * self.post_layer_weight.layernorm
        return res

    def _PostProcess(self, logits):
        
        output_tokens = torch.argmax(logits, dim=-1)
        return output_tokens