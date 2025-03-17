import torch
from abc import ABC, abstractmethod

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
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        # 如果没有任何缓存
        if key_states is not None:
            if self.key_cache[layer_idx] is None:
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def GetInputLength(
        self,
        layer_idx: int
    ):
        if self.key_cache[layer_idx] is None:
            return 0
        else:
            return self.key_cache[layer_idx].size(2)