import torch
from ..layer_weights.load_weights import Qwen2LayerWeight
from litelang.kernel.develop_rope.rope import rotary_emb_fwd
from litelang.kernel.develop_flash_attn.attn_decode_v2 import decode_flash_attention, gqa_decode_attention_fwd, gqa_reference_impl
from litelang.kernel.develop_flash_attn.flash_decoding import token_decode_attention_flash_decoding
from litelang.kernel.develop_flash_attn.flash_attn_v2 import triton_attention, gqa_context_attention, standard_attention_no_pad, context_attention_fwd_with_no_pad_and_kv_cache, context_attention_fwd_with_no_pad_and_kv_cache_and_prompt_cache
from litelang.kernel.develop_rmsnorm.rms_norm import rmsnorm
from litelang.kernel.develop_silu_and_mul.silu_and_mul import silu_and_mul_fwd
import torch.distributed as dist

class Qwen2TransformerLayer:
    
    def __init__(self, layer_idx, num_heads, head_dim, num_key_value_heads, tp_rank, world_size):
        self.layer_weight = Qwen2LayerWeight(layer_idx, tp_rank, world_size)
        self.layer_idx_ = layer_idx
        self.num_heads_ = num_heads // world_size
        self.head_dim_ = head_dim
        self.num_key_value_heads_ = num_key_value_heads // world_size
        assert self.num_heads_ % self.num_key_value_heads_ == 0, f"num_heads: {self.num_heads_}, num_key_value_heads: {self.num_key_value_heads_}, can not be divided"
        self.kv_group_num_ = self.num_heads_ // self.num_key_value_heads_
        self.layer_idx_ = layer_idx
        self.eps = 1e-6
        self.sm_scale = 1.0 / (self.head_dim_ ** 0.5)
        self.tp_rank_ = tp_rank
        
    def Forward(self, hidden_states: torch.Tensor, position_embeddings: torch.Tensor, model_inputs, kv_cache):
        residual = hidden_states
        hidden_states = self._InputLayernorm(hidden_states)
        q, k, v = self._QkvCompute(hidden_states)
        hidden_states = self._ComputeAttnScore(q, k, v, position_embeddings, model_inputs, kv_cache)
        before_hidden_states = hidden_states
        dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self._FfnNorm(hidden_states)
        hidden_states = self._Ffn(hidden_states)
        dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM)
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

    def _ComputeAttnScore(self, q, k, v, position_embeddings, model_inputs, kv_cache):
        hidden_states_shape = q.shape

        q = q.reshape(q.size(0), self.num_heads_, self.head_dim_)
        k = k.reshape(k.size(0), self.num_key_value_heads_, self.head_dim_)
        v = v.reshape(v.size(0), self.num_key_value_heads_, self.head_dim_)

        cos, sin = position_embeddings
        q, k = self._ComputeQK(q, k, cos, sin, model_inputs)

        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
        
        if model_inputs.is_prefill:
            kv_cache.write_prefill_kv_cache(model_inputs.request_mapping.values(), model_inputs.b_start_idx, self.layer_idx_, k, v)
            token_idxs = kv_cache.get_token_index(model_inputs.request_mapping.values())
            after_k, after_v = kv_cache.kv_cache_[self.layer_idx_, token_idxs, :self.num_key_value_heads_], kv_cache.kv_cache_[self.layer_idx_, token_idxs, self.num_key_value_heads_:]
        else:
            token_idxs = kv_cache.write_decode_kv_cache(model_inputs.request_mapping.values(), model_inputs.b_start_idx, self.layer_idx_, k, v)
            k, v = kv_cache.kv_cache_[self.layer_idx_, token_idxs, :self.num_key_value_heads_], kv_cache.kv_cache_[self.layer_idx_, token_idxs, self.num_key_value_heads_:]
        if not model_inputs.is_prefill:
            if not q.is_contiguous():
                q = q.contiguous()
            b_req_idx = [req.rid for req in model_inputs.request_mapping.values()]
            b_req_idx = torch.tensor(b_req_idx, dtype=torch.int32).cuda()
            attn_score = torch.zeros_like(q)
            
            token_decode_attention_flash_decoding(
                q, kv_cache.req_to_tokens_, b_req_idx, model_inputs.b_seq_len, model_inputs.b_seq_len.max().item(), q.size(-2), q.size(-1), kv_cache.kv_cache_[self.layer_idx_, :, :self.num_key_value_heads_], kv_cache.kv_cache_[self.layer_idx_, :, self.num_key_value_heads_:], out=attn_score
            )
        else:
            if not q.is_contiguous():
                q = q.contiguous()
            
            if model_inputs.radix_cache is None:
                b_req_idx = [req.rid for req in model_inputs.request_mapping.values()]
                b_req_idx = torch.tensor(b_req_idx, dtype=torch.int32).cuda()
                attn_score = torch.zeros_like(q)
                context_attention_fwd_with_no_pad_and_kv_cache(q, kv_cache.kv_cache_[self.layer_idx_, :, :self.num_key_value_heads_], kv_cache.kv_cache_[self.layer_idx_, :, self.num_key_value_heads_:], attn_score, b_req_idx, model_inputs.b_start_idx, model_inputs.b_seq_len, max_input_len=model_inputs.b_seq_len.max().item(), req_to_token_indexs=kv_cache.req_to_tokens_)
            else:
                b_req_idx = [req.rid for req in model_inputs.request_mapping.values()]
                b_req_idx = torch.tensor(b_req_idx, dtype=torch.int32).cuda()
                attn_score = torch.zeros_like(q)
                context_attention_fwd_with_no_pad_and_kv_cache_and_prompt_cache(q, kv_cache.kv_cache_[self.layer_idx_, :, :self.num_key_value_heads_], kv_cache.kv_cache_[self.layer_idx_, :, self.num_key_value_heads_:], attn_score, b_req_idx, model_inputs.b_start_idx, model_inputs.b_seq_len, b_shared_seq_len=model_inputs.b_shared_seq_len, max_input_len=model_inputs.b_seq_len.max().item(), req_to_token_indexs=kv_cache.req_to_tokens_)

        attn_score = attn_score.to(torch.float32)
        attn_score = attn_score.reshape(*hidden_states_shape)
        attn_score = attn_score @ self.layer_weight.o_proj.proj.transpose(-2, -1)
        if self.layer_weight.o_proj.bias is not None:
            # 一般这里没有bias
            attn_score = attn_score + self.layer_weight.o_proj.bias
        return attn_score
