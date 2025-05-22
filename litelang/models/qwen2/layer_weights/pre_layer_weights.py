from safetensors import safe_open
import torch


class Qwen2PreLayerWeight:
    def __init__(self):
        self.model_type_ = "qwen2.5"
        self.torch_dtype = torch.float16

    def _init_input_embds(self, weights):
        input_embds_name = f"model.embed_tokens.weight"
        self.input_embds = weights[input_embds_name].to(self.torch_dtype).cuda()

    def init(self, weights):
        self._init_input_embds(weights)
