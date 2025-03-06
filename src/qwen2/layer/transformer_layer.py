import torch
from ..layer_weights.load_weights import Qwen2LayerWeight

class Qwen2TransformerLayer:
    
    def __init__(self, layer_num):
        self.layer_weight = Qwen2LayerWeight(layer_num)
        self.eps = 1e-3
        
    def Forward(self, hidden_states: torch.Tensor, position_embeddings: torch.Tensor):
        residual = hidden_states
        hidden_states = self._InputLayernorm(input)
        hidden_states = self._QkvCompute(hidden_states, position_embeddings)
        hidden_states = self._ComputeAttnScore(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self._FfnNorm(hidden_states)
        hidden_states = self._Ffn(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
        
    def _Ffn(self, hidden_states):
        hidden_states = torch.nn.functional.silu(self.layer_weight.gate_proj @ hidden_states * self.layer_weight.up_proj @ hidden_states)
        hidden_states = self.layer_weight.down_proj @ hidden_states
        return hidden_states

    def _InputLayernorm(self, input):
        denominator = torch.sqrt(torch.sum(input * input, dim=-1) / input.size(-1) + self.eps)
        res = input / denominator * self.layer_weight.layernorm.proj
        return res
    
    def _FfnNorm(self, input):
        denominator = torch.sqrt(torch.sum(input * input, dim=-1) / input.size(-1) + self.eps)
        res = input / denominator * self.layer_weight.post_attn_layernorm.proj
        return res
    
    def _QkvCompute(self, input):
        q = input @ self.layer_weight.q_proj.proj + self.layer_weight.q_proj.bias
        k = input @ self.layer_weight.k_proj.proj + self.layer_weight.k_proj.bias
        v = input @ self.layer_weight.v_proj.proj + self.layer_weight.v_proj.bias
        return q, k, v
    
    def _ComputeQK(self, q, k, cos, sin):
        seq_len, h, dim = q.shape

        cos_seqlen = cos[:seq_len, 1, dim // 2]
        sin_seqlen = sin[:seq_len, 1, dim // 2]

        q0 = q[:, :, 0 : dim // 2]
        q1 = q[:, :, dim // 2 : dim]
        o0 = q0 * cos_seqlen - q1 * sin_seqlen
        o1 = q0 * sin_seqlen + q1 * cos_seqlen
        q_after = torch.cat((o0, o1), dim=-1)
        
        k0 = k[:, :, 0 : dim // 2]
        k1 = k[:, :, dim // 2 : dim]
        o0 = k0 * cos_seqlen - k1 * sin_seqlen
        o1 = k0 * sin_seqlen + k1 * cos_seqlen
        k_after = torch.cat((o0, o1), dim=-1)
        return q_after, k_after
    
    def _ComputeAttnScore(self, q, k, v, position_embeddings):
        cos, sin = position_embeddings
        q, k = self._computeQK(q, k, cos, sin)
        qk = q @ k.transpose(-2, -1)
        
        # 需要计算q和k的旋转位置编码，将位置信息编码进来，这里没写
        qk = qk / q.size(-1) ** 0.5
        qk = torch.softmax(qk, dim=-1)
        attn_score = qk @ v
        return attn_score
    