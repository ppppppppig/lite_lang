import torch


class Qwen2PostLayerWeight:
    def __init__(self, tie_wording_embedding):
        self.model_type_ = "qwen2.5"
        self.tie_wording_embedding_ = tie_wording_embedding
        self.torch_dtype = torch.float16

    def _init_lm_head(self, weights):
        if self.tie_wording_embedding_:
            lm_head_name = f"model.embed_tokens.weight"
        else:
            lm_head_name = f"lm_head.weight"
        self.lm_head = weights[lm_head_name].cuda().to(self.torch_dtype)

    def _InitNormWeight(self, weights):
        norm_name = f"model.norm.weight"
        self.layernorm = weights[norm_name].cuda().to(self.torch_dtype)

    def init(self, weights):
        self._InitNormWeight(weights)
        self._init_lm_head(weights)
