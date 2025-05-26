import torch
from ..layer_weights.load_weights import Qwen2LayerWeight
from litelang.kernel.develop_rope.rope import rotary_emb_fwd
from litelang.kernel.develop_flash_attn.attn_decode_v2 import (
    decode_flash_attention,
    gqa_decode_attention_fwd,
    gqa_reference_impl,
)
from litelang.kernel.develop_flash_attn.flash_decoding import (
    token_decode_attention_flash_decoding,
)
from litelang.kernel.develop_flash_attn.flash_attn_v2 import (
    triton_attention,
    gqa_context_attention,
    standard_attention_no_pad,
    context_attention_fwd_with_no_pad_and_kv_cache,
    context_attention_fwd_with_no_pad_and_kv_cache_and_prompt_cache,
)
from litelang.kernel.develop_rmsnorm.rms_norm import rmsnorm
from litelang.kernel.develop_silu_and_mul.silu_and_mul import silu_and_mul_fwd
import torch.distributed as dist
import time
from litelang.tools.profile import measure_function_time


class Qwen2TransformerLayer:

    def __init__(
        self, layer_idx, num_heads, head_dim, num_key_value_heads, tp_rank, world_size
    ):
        self.layer_weight = Qwen2LayerWeight(layer_idx, tp_rank, world_size)
        self.layer_idx_ = layer_idx
        self.num_heads_ = num_heads // world_size
        self.head_dim_ = head_dim
        self.num_key_value_heads_ = num_key_value_heads // world_size
        assert (
            self.num_heads_ % self.num_key_value_heads_ == 0
        ), f"num_heads: {self.num_heads_}, num_key_value_heads: {self.num_key_value_heads_}, can not be divided"
        self.kv_group_num_ = self.num_heads_ // self.num_key_value_heads_
        self.layer_idx_ = layer_idx
        self.eps = 1e-6
        self.sm_scale = 1.0 / (self.head_dim_**0.5)
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size

    def Forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        model_inputs,
        kv_cache,
    ):
        if self.layer_idx_ == 0:
            if model_inputs.is_prefill:
                
                match_len = [len(req.match_token_idxs) if req.match_token_idxs is not None else 0 for req in model_inputs.request_mapping.values()]
                b_match_len = torch.tensor(match_len).to(device='cuda', dtype=torch.int32)
                model_inputs.mem_idxs = kv_cache.alloc_token_idxs(model_inputs.b_rids, b_match_len, model_inputs.b_seq_len - b_match_len)
            else:
                model_inputs.mem_idxs = kv_cache.alloc_token_idxs(model_inputs.b_rids, model_inputs.b_seq_len - 1, torch.ones_like(model_inputs.b_seq_len))
        input1 = self._InputLayernorm(hidden_states)

        attn_output = self._ComputeAttnScore(
            input1, position_embeddings, model_inputs, kv_cache
        )
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(attn_output, op=dist.ReduceOp.SUM)
        hidden_states.add_(attn_output)

        input2 = self._FfnNorm(hidden_states)
        ffn_out = self._Ffn(input2)
        input2 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM)
        hidden_states.add_(ffn_out)
        return hidden_states

    def _Ffn(self, hidden_states):
        hidden_states = torch.mm(hidden_states, self.layer_weight.gate_and_up_proj)
        output = silu_and_mul_fwd(hidden_states)
        hidden_states = torch.mm(output, self.layer_weight.down_proj)
        return hidden_states

    # 要求输入[batch_size * seq_len, hidden_size]
    def _InputLayernorm(self, input):
        output = rmsnorm(input, self.layer_weight.layernorm.proj, self.eps)
        return output

    def _FfnNorm(self, input):
        output = rmsnorm(input, self.layer_weight.post_attn_layernorm.proj, self.eps)
        return output

    def _QkvCompute(self, input, model_inputs, cache_kv, position_embeddings):
        q = torch.addmm(
            self.layer_weight.q_proj.bias,
            input,
            self.layer_weight.q_proj.proj,
            beta=1.0,
            alpha=1.0,
        )
        torch.addmm(
            self.layer_weight.kv_proj_bias,
            input,
            self.layer_weight.kv_proj_weight,
            beta=1.0,
            alpha=1.0,
            out=cache_kv.view(cache_kv.size(0), 2 * self.num_key_value_heads_, -1)
        )

        
        cos, sin = position_embeddings
        cos_seqlen = torch.index_select(cos, 0, model_inputs.position_ids)
        sin_seqlen = torch.index_select(sin, 0, model_inputs.position_ids)
        rotary_emb_fwd(q.view(q.size(0), self.num_heads_, self.head_dim_), cache_kv[:, :self.num_key_value_heads_], cos_seqlen, sin_seqlen)
        return q, cache_kv


    def CreateCausalPaddingMask(self, mask, fill_value=torch.finfo(torch.float16).min):
        batch_size, seq_len = mask.shape
        device = mask.device

        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=0
        )
        padding_start = (mask == 1).float().argmax(dim=1)
        row_mask = (
            torch.arange(seq_len, device=device)[None, :] >= padding_start[:, None]
        )

        causal_mask = causal_mask[None, None, :, :]
        row_mask = row_mask[:, None, None, :]
        combined_mask = row_mask & causal_mask
        attn_mask = torch.where(combined_mask, 0, fill_value).cuda()
        return attn_mask

    def CreateDecodePaddingMask(self, mask, fill_value=torch.finfo(torch.float16).min):
        attn_mask = torch.where(mask.to(torch.bool), 0, fill_value).to(torch.float16)
        return attn_mask

    def _ComputeAttnScore(self, input, position_embeddings, model_inputs, kv_cache):
        cache_kv = kv_cache.kv_cache_[self.layer_idx_, model_inputs.mem_idxs, :]
        q, kv_cache.kv_cache_[self.layer_idx_, model_inputs.mem_idxs, :] = self._QkvCompute(input, model_inputs, cache_kv, position_embeddings)

        if not model_inputs.is_prefill:
            
            attn_score = torch.empty_like(q)

            token_decode_attention_flash_decoding(
                q.view(q.size(0), self.num_heads_, self.head_dim_),
                kv_cache.req_to_tokens_,
                model_inputs.b_rids,
                model_inputs.b_seq_len,
                model_inputs.b_seq_len.max().item(),
                self.num_heads_,
                self.head_dim_,
                kv_cache.kv_cache_[self.layer_idx_, :, : self.num_key_value_heads_],
                kv_cache.kv_cache_[self.layer_idx_, :, self.num_key_value_heads_ :],
                out=attn_score.view(q.size(0), self.num_heads_, self.head_dim_),
            )
        else:

            if model_inputs.radix_cache is None:

                attn_score = torch.empty_like(q)
                context_attention_fwd_with_no_pad_and_kv_cache(
                    q.view(q.size(0), self.num_heads_, self.head_dim_),
                    kv_cache.kv_cache_[self.layer_idx_, :, : self.num_key_value_heads_],
                    kv_cache.kv_cache_[self.layer_idx_, :, self.num_key_value_heads_ :],
                    attn_score.view(q.size(0), self.num_heads_, self.head_dim_),
                    model_inputs.b_rids,
                    model_inputs.b_start_idx,
                    model_inputs.b_seq_len,
                    max_input_len=model_inputs.b_seq_len.max().item(),
                    req_to_token_indexs=kv_cache.req_to_tokens_,
                )
            else:
                b_req_idx = model_inputs.b_rids
                attn_score = torch.empty_like(q)
                context_attention_fwd_with_no_pad_and_kv_cache_and_prompt_cache(
                    q.view(q.size(0), self.num_heads_, self.head_dim_),
                    kv_cache.kv_cache_[self.layer_idx_, :, : self.num_key_value_heads_],
                    kv_cache.kv_cache_[self.layer_idx_, :, self.num_key_value_heads_ :],
                    attn_score.view(q.size(0), self.num_heads_, self.head_dim_),
                    model_inputs.b_rids,
                    model_inputs.b_start_idx,
                    model_inputs.b_seq_len,
                    b_shared_seq_len=model_inputs.b_shared_seq_len,
                    max_input_len=model_inputs.b_seq_len.max().item(),
                    req_to_token_indexs=kv_cache.req_to_tokens_,
                )

        attn_score = torch.mm(attn_score.view(-1, self.num_heads_ * self.head_dim_), self.layer_weight.o_proj.proj)
        return attn_score
