import torch
from ..layer_weights.load_weights import Qwen2LayerWeight

class Qwen2TransformerLayer:
    
    def __init__(self, layer_num, num_heads, head_dim, num_key_value_heads):
        self.layer_weight = Qwen2LayerWeight(layer_num)
        self.num_heads_ = num_heads
        self.head_dim_ = head_dim
        self.num_key_value_heads_ = num_key_value_heads
        self.rep_times_ = self.num_heads_ // self.num_key_value_heads_
        self.eps = 1e-6
        
    def Forward(self, hidden_states: torch.Tensor, position_embeddings: torch.Tensor):
        residual = hidden_states
        hidden_states = self._InputLayernorm(hidden_states)
        q, k, v = self._QkvCompute(hidden_states)
        hidden_states = self._ComputeAttnScore(q, k, v, position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self._FfnNorm(hidden_states)
        hidden_states = self._Ffn(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
        
    def _Ffn(self, hidden_states):
        hidden_states = torch.nn.functional.silu(hidden_states @ self.layer_weight.gate_proj.transpose(-2, -1)) *  (hidden_states @ self.layer_weight.up_proj.transpose(-2, -1))
        hidden_states = hidden_states @ self.layer_weight.down_proj.transpose(-2, -1)
        return hidden_states

    # 要求输入[batch_size, seq_len, hidden_size]
    def _InputLayernorm(self, input):
        denominator = torch.sqrt(torch.sum(input * input, dim=-1) / input.size(-1) + self.eps)
        res = input / denominator[..., None] * self.layer_weight.layernorm.proj
        return res
    
    def _FfnNorm(self, input):
        denominator = torch.sqrt(torch.sum(input * input, dim=-1) / input.size(-1) + self.eps)
        res = input / denominator[..., None] * self.layer_weight.post_attn_layernorm.proj
        return res
    
    def _QkvCompute(self, input):
        q = input @ self.layer_weight.q_proj.proj.transpose(-2, -1).to(torch.float32) + self.layer_weight.q_proj.bias.to(torch.float32)
        k = input @ self.layer_weight.k_proj.proj.transpose(-2, -1).to(torch.float32) + self.layer_weight.k_proj.bias.to(torch.float32)
        v = input @ self.layer_weight.v_proj.proj.transpose(-2, -1).to(torch.float32) + self.layer_weight.v_proj.bias.to(torch.float32)
        return q, k, v
    
    def _ComputeQK(self, q, k, cos, sin):
        print(f"jzm: {cos}")
        batch, h, seq_len, dim = q.shape

        cos_seqlen = cos[:seq_len, :]
        sin_seqlen = sin[:seq_len, :]

        q0 = q[:, :, :, 0 : dim // 2]
        q1 = q[:, :, :, dim // 2 : dim]
        o0 = q0 * cos_seqlen[None, None, :, :] - q1 * sin_seqlen[None, None, :, :]
        o1 = q0 * sin_seqlen[None, None, :, :] + q1 * cos_seqlen[None, None, :, :]
        q_after = torch.cat((o0, o1), dim=-1)
        
        k0 = k[:, :, :, 0 : dim // 2]
        k1 = k[:, :, :, dim // 2 : dim]
        p0 = k0 * cos_seqlen[None, None, :, :] - k1 * sin_seqlen[None, None, :, :]
        p1 = k0 * sin_seqlen[None, None, :, :] + k1 * cos_seqlen[None, None, :, :]
        k_after = torch.cat((p0, p1), dim=-1)
        print(f"hjt: {q_after}")
        return q_after, k_after
    
    def repeat_kv(self, te):
        b, h, s, d = te.size(0), te.size(1), te.size(2), te.size(3)
        te =  te[:, :, None, :, :].expand(b, h, self.rep_times_, s, d)
        return te.reshape(b, h * self.rep_times_, s, d)
        
    def _CreateMockAttnMask(self, seq_len, fill_value=torch.finfo(torch.float32).min):
        attn_mask = torch.full((seq_len, seq_len), 0, dtype=torch.float32)
        ind = torch.triu_indices(seq_len, seq_len, offset=1)  # 修改offset=1
        attn_mask[ind[0], ind[1]] = fill_value
        return attn_mask.cuda()

    def _ComputeAttnScore(self, q, k, v, position_embeddings):
        hidden_states_shape = q.shape
        q = q.reshape(q.size(0), q.size(1), self.num_heads_, self.head_dim_).transpose(1, 2)
        k = k.reshape(k.size(0), k.size(1), self.num_key_value_heads_, self.head_dim_).transpose(1, 2)
        v = v.reshape(v.size(0), v.size(1), self.num_key_value_heads_, self.head_dim_).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = self._ComputeQK(q, k, cos, sin)
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)
        qk = q @ k.transpose(-2, -1)
        
        # 需要计算q和k的旋转位置编码，将位置信息编码进来，这里没写
        qk = qk / q.size(-1) ** 0.5
        attn_mask = self._CreateMockAttnMask(q.size(2))
        qk = qk + attn_mask
        qk = torch.softmax(qk, dim=-1)
        attn_score = qk @ v
        attn_score = attn_score.transpose(1, 2).reshape(*hidden_states_shape)
        attn_score = attn_score @ self.layer_weight.o_proj.proj.transpose(-2, -1)
        if self.layer_weight.o_proj.bias is not None:
            attn_score = attn_score + self.layer_weight.o_proj.bias
        return attn_score
    