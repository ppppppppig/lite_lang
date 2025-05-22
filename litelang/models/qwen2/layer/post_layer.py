import torch
from ..layer_weights.post_layer_weights import Qwen2PostLayerWeight
from litelang.kernel.develop_rmsnorm.rms_norm import rmsnorm

class Qwen2PostLayer:

    def __init__(self, tie_wording_embedding):
        self.post_layer_weight = Qwen2PostLayerWeight(tie_wording_embedding)
        self.temperature = 0.5
        self.top_p = 1.0
        self.top_k = 1
        self.eps = 1e-5

    def Forward(self, hidden_states, model_inputs):
        will_compute_hidden_states = self._select_hidden_states(
            hidden_states, model_inputs
        )
        
        will_compute_hidden_states = self._InputLayernorm(will_compute_hidden_states)
        logits = will_compute_hidden_states @ self.post_layer_weight.lm_head.transpose(
            -2, -1
        )
        # will_compute_hidden_states = will_compute_hidden_states.to(torch.float32)
        return self._PostProcess(logits, model_inputs)

    def _select_hidden_states(self, hidden_states, model_inputs):
        num_reqs = model_inputs.b_start_idx.size(0)
        after_hidden_states = []
        for req_idx in range(num_reqs):
            start_idx = model_inputs.b_start_idx[req_idx]
            seq_len = model_inputs.b_seq_len[req_idx]
            if model_inputs.is_prefill:
                if model_inputs.radix_cache is not None:

                    seq_len = seq_len - model_inputs.b_shared_seq_len[req_idx]
                after_hidden_states.append(hidden_states[start_idx + seq_len - 1, :])
            else:
                after_hidden_states.append(hidden_states[start_idx, :])
        after_hidden_states = torch.stack(after_hidden_states, dim=0)
        
        return after_hidden_states[:, :]

    def _InputLayernorm(self, input):
        res = rmsnorm(input, self.post_layer_weight.layernorm, self.eps)
        return res

    # def _PostProcess(self, logits, model_inputs):
    #     temperature, top_p, top_k, do_sample = model_inputs.get_post_sample_para()
    #     logits = logits.div_(temperature[:, None, None])
    #     logits = logits.softmax(dim=-1)

    #     logits_sort, logits_idx = logits.sort(dim=-1, descending=True)
    #     logits_sum = logits_sort.cumsum(dim=-1)

    #     top_p_mask = logits_sum <= top_p[:, None, None]
    #     # 确保至少保留一个token（即使累积和超过top_p）
    #     top_p_mask[..., 0] = True
    #     logits_sort = logits_sort * top_p_mask

    #     logits_sort[
    #         torch.arange(0, logits_sort.size(-1)).cuda().view(1, 1, -1)
    #         >= top_k[:, None, None]
    #     ] = 0.0
    #     logits_sort = logits_sort.view(-1, logits_sort.size(-1))
    #     logits_idx = logits_idx.view(-1, logits_idx.size(-1))
    #     sample_ind = torch.multinomial(logits_sort, num_samples=1, replacement=True)
    #     output_tokens_sample = torch.gather(logits_idx, dim=-1, index=sample_ind)

    #     output_tokens_argmax = logits_idx[:, 0][:, None]
    #     output_tokens = torch.where(
    #         do_sample[:, None], output_tokens_sample, output_tokens_argmax
    #     )
    #     return output_tokens
    
    def _PostProcess(self, logits, model_inputs):
        """直接取 logits 的 argmax 作为输出（删除所有后处理）"""
        return logits.argmax(dim=-1, keepdim=True)
