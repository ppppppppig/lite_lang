import torch
from ..layer_weights.load_weights import Qwen2LayerWeight
from ..cache import Cache, NormalCache
from typing import Optional
from ..kernel.develop_rope.rope import rotary_emb_fwd
from ..kernel.develop_flash_attn.flash_attn_v2 import flash_attentionv2

class Qwen2TransformerLayer:
    
    def __init__(self, layer_idx, num_heads, head_dim, num_key_value_heads):
        self.layer_weight = Qwen2LayerWeight(layer_idx)
        self.layer_idx_ = layer_idx
        self.num_heads_ = num_heads
        self.head_dim_ = head_dim
        self.num_key_value_heads_ = num_key_value_heads
        self.rep_times_ = self.num_heads_ // self.num_key_value_heads_
        self.layer_idx_ = layer_idx
        self.eps = 1e-6
        
    def Forward(self, hidden_states: torch.Tensor, position_embeddings: torch.Tensor, mask, is_decode, past_key_values: Optional[Cache]):
        residual = hidden_states
        hidden_states = self._InputLayernorm(hidden_states)
        q, k, v = self._QkvCompute(hidden_states)
        hidden_states = self._ComputeAttnScore(q, k, v, position_embeddings, mask, is_decode, past_key_values)
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
    
    def _ComputeQK(self, q, k, cos, sin, past_key_values):
        batch, seq_len, _, dim = q.shape
        

        last_input_length = past_key_values.GetInputLength(self.layer_idx_)
        cos_seqlen = cos[last_input_length:last_input_length + seq_len, :]
        sin_seqlen = sin[last_input_length:last_input_length + seq_len, :]

        q = q.reshape(batch * seq_len, -1, dim)
        k = k.reshape(batch * seq_len, -1, dim) # GQA的头数有时不一致
        
        rotary_emb_fwd(q, k, cos_seqlen, sin_seqlen)
        
        q_after = q.reshape(batch, seq_len, -1, dim).transpose(1, 2)
        k_after = k.reshape(batch, seq_len, -1, dim).transpose(1, 2)
        
        return q_after, k_after
    
    def repeat_kv(self, te):
        b, h, s, d = te.size(0), te.size(1), te.size(2), te.size(3)
        te =  te[:, :, None, :, :].expand(b, h, self.rep_times_, s, d)
        return te.reshape(b, h * self.rep_times_, s, d)
        
    def CreateCausalPaddingMask(self, mask, fill_value=torch.finfo(torch.float32).min):
        batch_size, seq_len = mask.shape
        device = mask.device
        
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=0
        )
        padding_start = (mask == 1).float().argmax(dim=1)
        row_mask = (torch.arange(seq_len, device=device)[None, :] >= padding_start[:, None])
        
        causal_mask = causal_mask[None, None, :, :]  
        row_mask = row_mask[:, None, None, :] 
        combined_mask = row_mask & causal_mask
        attn_mask = torch.where(
            combined_mask,
            0,
            fill_value
        ).cuda()
        return attn_mask

    def CreateDecodePaddingMask(self, mask, fill_value=torch.finfo(torch.float32).min):
        attn_mask = torch.where(
            mask.to(torch.bool),
            0,
            fill_value
        ).to(torch.float32)
        return attn_mask

    def _ComputeAttnScore(self, q, k, v, position_embeddings, mask, is_decode, past_key_values: Optional[Cache]):
        hidden_states_shape = q.shape
        q = q.reshape(q.size(0), q.size(1), self.num_heads_, self.head_dim_)
        k = k.reshape(k.size(0), k.size(1), self.num_key_value_heads_, self.head_dim_)
        v = v.reshape(v.size(0), v.size(1), self.num_key_value_heads_, self.head_dim_).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = self._ComputeQK(q, k, cos, sin, past_key_values)
        k, v = past_key_values.update(k, v, self.layer_idx_)
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)
        qk = q @ k.transpose(-2, -1)
        
        # 需要计算q和k的旋转位置编码，将位置信息编码进来，这里没写
        qk = qk / q.size(-1) ** 0.5
        if not is_decode:
            attn_mask = self.CreateCausalPaddingMask(mask)
            qk = qk + attn_mask
        else:
            attn_mask = self.CreateDecodePaddingMask(mask)
            qk = qk + attn_mask[:, None, None, :]
        qk = torch.softmax(qk, dim=-1)
        attn_score = qk @ v
        attn_score = attn_score.transpose(1, 2).reshape(*hidden_states_shape)
        attn_score = attn_score @ self.layer_weight.o_proj.proj.transpose(-2, -1)
        if self.layer_weight.o_proj.bias is not None:
            attn_score = attn_score + self.layer_weight.o_proj.bias
        return attn_score
