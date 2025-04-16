import torch
from abc import ABC, abstractmethod
from models.tools.profile import get_available_gpu_memory
from dataclasses import dataclass
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
        

@dataclass
class Node:
    def __init__(self, rid):
        self.rid_ = rid
        self.next_ = None

# 维护一个链表，分配的时候返回一个rid，释放将该rid放在链表最后
class FreeReq:
    def __init__(self, max_batch_size):
        self.free_rid_list_ = [Node(i) for i in range(max_batch_size)]
        self.root_rid_ = Node(-1)
        self.root_rid_.next_ = self.free_rid_list_[0]
        for i in range(max_batch_size - 1):
            self.free_rid_list_[i].next_ = self.free_rid_list_[i + 1]
        self.free_rid_list_[-1].next_ = None
        
    def alloc_new_req(self):
        if self.root_rid_.next_ is None:
            return None
        else:
            rid = self.root_rid_.next_.rid_
            self.root_rid_.next_ = self.root_rid_.next_.next_
        return rid

    def free_req(self, rid):
        now_head_id = self.root_rid_.next_.rid_
        self.root_rid_.next_ = self.free_rid_list_[rid]
        self.free_rid_list_[rid].next_ = self.free_rid_list_[now_head_id]
        return True


# 暂时只处理page=1的情况
@dataclass
class PageCache:
    def __init__(self, max_batch_size, max_length, mem_usage, num_layers, head_dim, num_key_value_heads, torch_dtype=torch.float16, tp_rank=0, tp_world_size=1):
        self.kv_cache_ = None
        self.req_to_tokens_ = None
        self.free_req_ = None
        self.physical_free_token_ = None
        self.kv_max_size_ = None
        self.physical_free_token_start_ = 0
        self.physical_free_token_end_ = 0
        self.token_size_ = None
        
        # 模型相关参数
        assert num_key_value_heads % tp_world_size == 0, "num_key_value_heads_ should be divisible by tp_world_size_"
        self.num_key_value_heads_ = num_key_value_heads // tp_world_size
        self.torch_dtype_ = torch_dtype
        self.dtype_size_ = get_dtype_size(self.torch_dtype_)
        self.num_layers_ = num_layers
        self.head_dim_ = head_dim
        
        # 配置参数
        self.max_batch_size_ = max_batch_size
        self.max_length_ = max_length
        self.memory_usage_ = mem_usage
        self.tp_rank_ = tp_rank
        self.tp_world_size_ = tp_world_size
        
        self._init_token_size()
        self._init_size()
        self._init_kv_buffer()
        self._init_index()
        
    def _init_token_size(self):
        self.token_size_ = self.head_dim_ * self.num_key_value_heads_ * 2 * self.dtype_size_ * self.num_layers_ / 1024 / self.tp_world_size_
    
    def _init_kv_buffer(self):
        self.kv_cache_ = torch.empty((self.num_layers_, self.kv_max_size_, self.num_key_value_heads_ * 2, self.head_dim_), dtype=self.torch_dtype_).cuda()
        self.physical_free_token_ = torch.arange(0, self.kv_max_size_, dtype=self.torch_dtype_)
        self.physical_free_token_start_ = 0
        self.physical_free_token_end_ = self.kv_max_size_ 
    
    def _init_index(self):
        self.req_to_tokens_ = torch.empty((self.max_batch_size_, self.max_length_), dtype=torch.int32).cuda()
        self.free_req_ = FreeReq(self.max_batch_size_)
        
    def _init_size(self):
        self.kv_max_size_ = int((get_available_gpu_memory() * self.memory_usage_) // 2 * ( 1024 ** 2))
        self.kv_max_size_ = int(self.kv_max_size_ // self.token_size_)
        assert self.kv_max_size_ > 0, "GPU memory is not enough to allocate cache."
    
    def _alloc_token_idxs(self, rid, length, new_token_length):
        
        assert self.physical_free_token_start_ + new_token_length <= self.physical_free_token_end_, "no enough kv cache"
        
        self.req_to_tokens_[rid, length: length + new_token_length] = self.physical_free_token_[self.physical_free_token_start_: self.physical_free_token_start_ + new_token_length].cuda()
        self.physical_free_token_start_ += new_token_length
    
    def _write_kv_cache(self, layer_num, rid, start, length, new_token_length, key_states, value_states):
        if layer_num == 0:
            self._alloc_token_idxs(rid, length, new_token_length)
            
        token_idxs = self.req_to_tokens_[rid, length: length + new_token_length]
        # 这里的start应该是从b_start_loc的张量中取
        self.kv_cache_[layer_num, token_idxs, :self.num_key_value_heads_] = key_states[start: start + new_token_length]
        self.kv_cache_[layer_num, token_idxs, self.num_key_value_heads_:] = value_states[start: start + new_token_length]
    
    def alloc_req(self):
        rid = self.free_req_.alloc_new_req()
        return rid
    
    def free_req(self, req):
        rid = req.rid
        length = req.length - 1
        
        token_idxs = self.req_to_tokens_[rid, :length]
        self.free_token_idxs(token_idxs)
        self.free_req_.free_req(rid)
        return True
    
    def get_token_index(self, reqs):
        no_padding_kv_cache = []
        for req in reqs:
            no_padding_kv_cache.append(self.req_to_tokens_[req.rid, :req.length])
        no_padding_kv_cache = torch.cat(no_padding_kv_cache, dim=0).cuda()
        return no_padding_kv_cache
        
    def free_token_idxs(self, token_idxs):
        assert self.physical_free_token_start_ - token_idxs.size(0) >= 0, "page cache internal error"
        self.physical_free_token_[self.physical_free_token_start_ - token_idxs.size(0): self.physical_free_token_start_] = token_idxs.cpu()
        self.physical_free_token_start_ -= token_idxs.size(0)
    
    def write_prefill_kv_cache(self, reqs, b_start_idx, layer_num, key_states, value_states):
        for idx, req in enumerate(reqs):
            rid = req.rid
            length = 0
            new_token_length = req.length
            self._write_kv_cache(layer_num, rid, b_start_idx[idx].item(), length, new_token_length, key_states, value_states)
            
    def write_decode_kv_cache(self, reqs, b_start_idx, layer_num, key_states, value_states):
        for idx, req in enumerate(reqs):
            rid = req.rid
            length = req.length - 1
            new_token_length = 1
            self._write_kv_cache(layer_num, rid, b_start_idx[idx].item(), length, new_token_length, key_states, value_states)
        return self.get_token_index(reqs)
            
    def dealloc_reqs(self, reqs):
        for req in reqs:
            self.free_req(req)

    def can_allocated(self, length):
        if self.physical_free_token_start_ + length > self.physical_free_token_end_:
            return False
        else:
            return True