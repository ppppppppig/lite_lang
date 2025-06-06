import torch
from ..layer_weights.pre_layer_weights import Qwen2PreLayerWeight


class Qwen2PreLayer:

    def __init__(self, max_length, hidden_size, num_head, head_dim):
        self.hidden_size_ = hidden_size
        self.max_length_ = max_length
        self.num_head_ = num_head
        self.head_dim_ = head_dim
        self.pre_layer_weight = Qwen2PreLayerWeight()
        self.position_embeddings = self._ComputePosition()

    def Forward(self, input_tokens):
        hidden_states = torch.embedding(
            self.pre_layer_weight.input_embds, input_tokens, padding_idx=-1
        )
        return hidden_states

    def _ComputePosition(self):
        theta = 1000000
        freqs = 1.0 / theta ** (torch.arange(0, self.head_dim_, 2) / self.head_dim_)
        pos = torch.arange(0, self.max_length_)

        angel = pos[:, None] * freqs[None, :]

        cos = torch.cos(angel).cuda().to(torch.float16)
        sin = torch.sin(angel).cuda().to(torch.float16)
        return (cos, sin)
