import torch
from ..layer_weights.load_weights import Qwen2LayerWeight
from ..cache import Cache, NormalCache
from typing import Optional
from ..kernel.develop_rope.rope import rotary_emb_fwd
from ..kernel.develop_flash_attn.attn_decode_v2 import decode_flash_attention, gqa_decode_attention_fwd, gqa_reference_impl
from ..kernel.develop_flash_attn.flash_attn_v2 import triton_attention, gqa_context_attention, standard_attention_no_pad
from ..kernel.develop_rmsnorm.rms_norm import rmsnorm
from ..kernel.develop_silu_and_mul.silu_and_mul import silu_and_mul_fwd

class Qwen2TransformerLayer:
    
    def __init__(self, layer_idx, num_heads, head_dim, num_key_value_heads):
        self.layer_weight = Qwen2LayerWeight(layer_idx)
        self.layer_idx_ = layer_idx
        self.num_heads_ = num_heads
        self.head_dim_ = head_dim
        self.num_key_value_heads_ = num_key_value_heads
        assert self.num_heads_ % self.num_key_value_heads_ == 0, f"num_heads: {self.num_heads_}, num_key_value_heads: {self.num_key_value_heads_}, can not be divided"
        self.kv_group_num_ = self.num_heads_ // self.num_key_value_heads_
        self.layer_idx_ = layer_idx
        self.eps = 1e-6
        self.sm_scale = 1.0 / (self.head_dim_ ** 0.5)
        
    def Forward(self, hidden_states: torch.Tensor, position_embeddings: torch.Tensor, model_inputs):
        residual = hidden_states
        hidden_states = self._InputLayernorm(hidden_states)
        q, k, v = self._QkvCompute(hidden_states)
        hidden_states = self._ComputeAttnScore(q, k, v, position_embeddings, model_inputs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self._FfnNorm(hidden_states)
        hidden_states = self._Ffn(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    
    def _Ffn(self, hidden_states):
        gate_and_up_proj = torch.cat([self.layer_weight.gate_proj, self.layer_weight.up_proj], dim=0)
        hidden_states = hidden_states @ gate_and_up_proj.transpose(-2, -1)
        M, N = hidden_states.shape
        output = torch.empty((M, N // 2), dtype=torch.float32).cuda()
        silu_and_mul_fwd(hidden_states, output)
        hidden_states = output @ self.layer_weight.down_proj.transpose(-2, -1)
        return hidden_states

    # 要求输入[batch_size * seq_len, hidden_size]
    def _InputLayernorm(self, input):
        output = rmsnorm(input, self.layer_weight.layernorm.proj, self.eps)
        return output
    
    def _FfnNorm(self, input):
        output = rmsnorm(input, self.layer_weight.post_attn_layernorm.proj, self.eps)
        return output
    
    def _QkvCompute(self, input):
        q = input @ self.layer_weight.q_proj.proj.transpose(-2, -1).to(torch.float32) + self.layer_weight.q_proj.bias.to(torch.float32)
        k = input @ self.layer_weight.k_proj.proj.transpose(-2, -1).to(torch.float32) + self.layer_weight.k_proj.bias.to(torch.float32)
        v = input @ self.layer_weight.v_proj.proj.transpose(-2, -1).to(torch.float32) + self.layer_weight.v_proj.bias.to(torch.float32)
        return q, k, v
    
    def _ComputeQK(self, q, k, cos, sin, model_inputs):

        cos_seqlen = torch.index_select(cos, 0, model_inputs.position_ids)
        sin_seqlen = torch.index_select(sin, 0, model_inputs.position_ids)

        rotary_emb_fwd(q, k, cos_seqlen, sin_seqlen)

        return q, k
        
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

    def _ComputeAttnScore(self, q, k, v, position_embeddings, model_inputs):
        hidden_states_shape = q.shape
        
        q = q.reshape(q.size(0), self.num_heads_, self.head_dim_)
        k = k.reshape(k.size(0), self.num_key_value_heads_, self.head_dim_)
        v = v.reshape(v.size(0), self.num_key_value_heads_, self.head_dim_)
        cos, sin = position_embeddings
        q, k = self._ComputeQK(q, k, cos, sin, model_inputs)
        
        k, v = model_inputs.past_key_values.update(k, v, self.layer_idx_, model_inputs)
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
        if not model_inputs.is_prefill:
            if not q.is_contiguous():
                q = q.contiguous()
            attn_score = gqa_reference_impl(q, k, v, model_inputs.kv_start_idx, model_inputs.b_seq_len, sm_scale=self.sm_scale, kv_group_num=self.kv_group_num_)
        else:
            if not q.is_contiguous():
                q = q.contiguous()
            attn_score = standard_attention_no_pad(q, k, v, model_inputs.b_start_idx, model_inputs.b_seq_len, sm_scale=self.sm_scale, kv_group_num=self.kv_group_num_)
        attn_score = attn_score.to(torch.float32)
        attn_score = attn_score.reshape(*hidden_states_shape)
        attn_score = attn_score @ self.layer_weight.o_proj.proj.transpose(-2, -1)
        if self.layer_weight.o_proj.bias is not None:
            attn_score = attn_score + self.layer_weight.o_proj.bias
        return attn_score
