import torch
from abc import ABC, abstractmethod

def get_dtype_size(dtype):
    if dtype == torch.bool:
        return 1
    elif dtype.is_floating_point:
        return torch.finfo(dtype).bits // 8
    else:
        return torch.iinfo(dtype).bits // 8

class Cache(ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        raise NotImplementedError("Make sure to implement `update` in a subclass.")
    
    @abstractmethod
    def GetInputLength(
        self,
        query_states: torch.Tensor,
        layer_idx: int
    ):
        raise NotImplementedError("Make sure to implement `update` in a subclass.")


    
# 传统的cache，没有使用block attention
class NormalCache(Cache):
    def __init__(self, layer_num):
        super().__init__()
        self.key_cache = [None for i in range(layer_num)]
        self.value_cache = [None for i in range(layer_num)]
        
    def _convert_states_to_normal_cache(self, states, model_inputs):
        num_reqs = model_inputs.b_start_idx.size(0)
        after_states = []
        for req_idx in range(num_reqs):
            start_idx = model_inputs.b_start_idx[req_idx]
            seq_len = model_inputs.b_seq_len[req_idx]
            if model_inputs.is_prefill:
                # prefill的时候最后一位是seq_len - 1
                after_states.append(states[start_idx: start_idx + seq_len, ...])
            else:
                # decode的时候只有一维
                after_states.append(states[start_idx, ...].unsqueeze(0))
        return after_states     

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        model_inputs
    ):
        
        covert_key_states_list = self._convert_states_to_normal_cache(key_states, model_inputs)
        covert_value_states_list = self._convert_states_to_normal_cache(value_states, model_inputs)
        if key_states is not None:
            if self.key_cache[layer_idx] is None:
                self.key_cache[layer_idx] = covert_key_states_list
                self.value_cache[layer_idx] = covert_value_states_list
            else:
                for idx, _ in enumerate(covert_key_states_list):
                    self.key_cache[layer_idx][idx] = torch.cat([self.key_cache[layer_idx][idx] , covert_key_states_list[idx]], dim=0)
                    self.value_cache[layer_idx][idx] = torch.cat([self.value_cache[layer_idx][idx], covert_value_states_list[idx]], dim=0)
        ret_key_cache = torch.cat(self.key_cache[layer_idx], dim=0)
        ret_value_cache = torch.cat(self.value_cache[layer_idx], dim=0)
        return ret_key_cache, ret_value_cache

    
    def GetInputLength(
        self,
        layer_idx: int
    ):
        if self.key_cache[layer_idx] is None:
            return 0
        else:
            return self.key_cache[layer_idx].size(2)
        
# class PageCache(Cache): 
#     def __init__(self, layer_num, page_size, ):
#         super().__init__()
#         token_size = self.compute_token_size(how_many_token_should_cache, layer_num, head_dim, num_key_value_heads, torch_dtype)

#         self.key_cache = [None for i in range(layer_num)]
#         self.value_cache = [None for i in range(layer_num)]
#         self.req_to_page: torch.Tensor = None
#         self.req_to_position_id: torch.Tensor = None
#         self.page_is_free: torch.Tensor = None
#         self.page_cache: torch.Tensor = torch.empty((how_many_token_should_cache, page_size, token_size), dtype=torch_dtype).cuda()
        
#     def compute_token_size(self, num_layers, head_dim, num_key_value_heads, torch_dtype):
#         dtype_size = get_dtype_size(torch_dtype)
#         token_size = head_dim * num_key_value_heads * 2 * dtype_size * num_layers
#         return token_size
    
#     def update(
#         self,
#         key_states: torch.Tensor,
#         value_states: torch.Tensor,
#         layer_idx: int,
#     ):
#         # 如果没有任何缓存
#         if key_states is not None:
#             if self.key_cache[layer_idx] is None:
#                 self.key_cache[layer_idx] = key_states
#                 self.value_cache[layer_idx] = value_states
#             else:
#                 self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=0)
#                 self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=0)
        
#         return self.page_cache,