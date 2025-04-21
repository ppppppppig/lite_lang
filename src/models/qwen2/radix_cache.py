import torch
from typing import List
import time
from sortedcontainers import SortedSet

class UniqueTimeIdGenerator:
    def __init__(self):
        self.counter = 0

    def generate_time_id(self):
        self.counter += 1
        return self.counter


time_gen = UniqueTimeIdGenerator()

class TreeNode:
    def __init__(self, token_ids, page_cache_token_ixs, ref_count=0):
        self.token_ids = token_ids
        self.page_cache_token_idxs = page_cache_token_ixs
        self.ref_count = ref_count
        self.child = {}
        self.parent_node = None
        self.time = time_gen.generate_time_id()
    
    def get_compare_key(self):
        return self.time
    
    def split_node(self, match_len):
        new_child_key = self.token_ids[match_len].item()
        origin_child = self.child.copy()
        self.child.clear()
        new_child = TreeNode(self.token_ids[match_len:], self.page_cache_token_idxs[match_len:])
        new_child.child = origin_child
        # 将当前节点分成两个节点后，ref_count应该相同
        new_child.ref_count = self.ref_count
        self.child[new_child_key] = new_child
        self.child[new_child_key].parent_node = self
        self.token_ids = self.token_ids[:match_len]
        self.page_cache_token_idxs = self.page_cache_token_idxs[:match_len]
        return self.child[new_child_key]
    
    def can_evict(self):
        return len(self.child) == 0 and self.ref_count == 0
    
    def touch(self):
        self.time = time_gen.generate_time_id()
    

def match(a, b):
    idx = 0
    for l,r in zip(a, b):
        if l != r:
            break
        idx += 1
    return idx

class RadixTree:
    def __init__(self):
        self.root = TreeNode(None, None, 1) # 初始化为1，永远不会释放
        self.evict_node_set_ = SortedSet(key=lambda x: x.get_compare_key())
        self.all_taken_tokens_ = 0
        
    def insert(self, key, value, shared_key=None):
        if shared_key is None:
            shared_key = torch.tensor([], device='cpu', dtype=torch.int64)
        now_node = self.root
        
        parent_node, has_match_len, match_len = self.find_parent_node(now_node, key, 0, shared_key)
        #match_len == 0 代表不需要分割 parent_node
        if match_len > 0:
            seperate_child_node = parent_node.split_node(match_len)
            if seperate_child_node.can_evict():
                self.evict_node_set_.add(seperate_child_node)

        # 如果全部匹配完了，说明不用插入新节点
        if has_match_len >= key.size(0):
            return 
        self.all_taken_tokens_ += key.size(0) - has_match_len
        new_child_key = key[has_match_len].item()
        new_node = TreeNode(key[has_match_len:], value[has_match_len:])
        parent_node.child[new_child_key] = new_node
        new_node.parent_node = parent_node
        if new_node.can_evict():
            self.evict_node_set_.discard(parent_node)
            self.evict_node_set_.add(new_node)
        
    def find_parent_node(self, now_node, key, has_match_len, shared_key):
        
        if key.size(0) <= 0:
            return now_node, has_match_len, 0
        child_key = key[0].item()
        if child_key not in now_node.child:
            return now_node, has_match_len, 0
        
        child_node = now_node.child[child_key]
        match_len = match(key, child_node.token_ids)
        has_match_len += match_len
        if match_len != now_node.child[child_key].token_ids.size(0):
            return now_node.child[child_key], has_match_len, match_len
        else:
            # shared_key使用了之前的node，当使用完后，需要将这些节点的ref_count - 1
            if shared_key.size(0) > 0:
                child_node.ref_count -= 1
            return self.find_parent_node(now_node.child[child_key], key[match_len:], has_match_len, shared_key[has_match_len:])
            
    def match_prefix(self, key):
        now_node = self.root
        parent_node, has_match_len, value_list, match_len = self.find_prefix_token_idxs(now_node, key, [], 0)
        if match_len > 0:
            seperate_child_node = parent_node.split_node(match_len)
            if seperate_child_node.can_evict():
                self.evict_node_set_.add(seperate_child_node)
        print(f"value_list: {value_list}")
        page_cache_token_idxs = torch.cat(value_list, dim=0)
        return page_cache_token_idxs, has_match_len
            
    def find_prefix_token_idxs(self, now_node, key, value_list, has_match_len):

        if key.size(0) <= 0:
            return now_node, has_match_len, value_list, 0
        child_key = key[0].item()
        if child_key not in now_node.child:
            return now_node, has_match_len, value_list, 0
        child_node = now_node.child[child_key]
        match_len = match(key, child_node.token_ids)
        has_match_len += match_len
        value_list.append(child_node.page_cache_token_idxs[:match_len])
        
        # 不要更改顺序
        self.evict_node_set_.discard(child_node)
        child_node.ref_count += 1
        child_node.touch()
        
        if match_len != child_node.token_ids.size(0):
            return child_node, has_match_len, value_list, match_len
        else:
            return self.find_prefix_token_idxs(child_node, key[match_len:], value_list, has_match_len)
        
    def evict_node(self, need_token_nums):
        import pdb; pdb.set_trace()
        if need_token_nums > self.all_taken_tokens_:
            raise ValueError("need_token_nums is larger than all taken tokens")
        dealloc_token_nums = 0
        while self.evict_node_set_:
            node = self.evict_node_set_.pop(0)
            dealloc_token_nums += self.try_del_node(node)
            if dealloc_token_nums > need_token_nums:
                self.all_taken_tokens_ -= dealloc_token_nums
                return True
        return False

    
    def try_del_node(self, node):
        assert node.ref_count == 0, "can not del node while ref_count not equal 0" 
        dealloc_token_nums = node.token_ids.size(0)
        del node.parent_node.child[node.token_ids[0].item()]
        if node.parent_node.can_evict(): 
            self.evict_node_set_.add(node.parent_node)
        del node
        
        return dealloc_token_nums
    

def display_the_tree_layer_order(root: TreeNode):
    print(f"==========================================")
    print(f"token_ids: {root.token_ids}, page_cache_token_ixs: {root.page_cache_token_idxs}")
    print(f"refcount: {root.ref_count}, time: {root.time}")
    print(f"child: {root.child.keys()}")
    for child in root.child.values():
        display_the_tree_layer_order(child)


if __name__ == '__main__':
    radix_tree = RadixTree()
    print(f"time.perf_counter(): {time.perf_counter()}")
    key = torch.tensor([1, 2, 3, 4], device='cpu', dtype=torch.int64)
    value = torch.tensor([1, 2, 3, 4], device='cpu', dtype=torch.int64)
    radix_tree.insert(key, value)
    display_the_tree_layer_order(radix_tree.root)
    
    key = torch.tensor([1, 2, 3], device='cpu', dtype=torch.int64)
    value = torch.tensor([1, 2, 3], device='cpu', dtype=torch.int64)
    radix_tree.insert(key, value)
    display_the_tree_layer_order(radix_tree.root)
    
    key = torch.tensor([1, 2, 3, 10, 11, 12, 13], device='cpu', dtype=torch.int64)
    value = torch.tensor([1, 2, 3, 10, 11, 12, 13], device='cpu', dtype=torch.int64)
    radix_tree.insert(key, value)
    display_the_tree_layer_order(radix_tree.root)
    print(f"time.perf_counter(): {time.perf_counter()}")
    
    
    print(f"********************************")
    key = torch.tensor([1, 2, 20, 21, 22, 23], device='cpu', dtype=torch.int64)
    value = torch.tensor([1, 2, 20, 21, 22, 23], device='cpu', dtype=torch.int64)
    radix_tree.insert(key, value)
    display_the_tree_layer_order(radix_tree.root)
    print(f"time.perf_counter(): {time.perf_counter()}")
    print(f"********************************")
    
    key = torch.tensor([1, 2])
    page_cache_token_ixs, has_match_len = radix_tree.match_prefix(key)
    display_the_tree_layer_order(radix_tree.root)
    print(f"key: {key}, page_cache_token_ixs: {page_cache_token_ixs}")
    
    key = torch.tensor([1,2,3,4])
    page_cache_token_ixs, has_match_len = radix_tree.match_prefix(key)
    display_the_tree_layer_order(radix_tree.root)
    print(f"key: {key}, page_cache_token_ixs: {page_cache_token_ixs}")
    
    key = torch.tensor([1, 2, 3, 10, 11, 12, 13, 15, 16, 17,18])
    page_cache_token_ixs, has_match_len = radix_tree.match_prefix(key)
    display_the_tree_layer_order(radix_tree.root)
    print(f"key: {key}, page_cache_token_ixs: {page_cache_token_ixs}")
    
    key = torch.tensor([1, 2, 3, 10, 11, 12, 13, 15, 16, 17,18])
    value = torch.tensor([1, 2, 3, 10, 11, 12, 13, 15, 16, 17,18])
    radix_tree.insert(key, value)
    display_the_tree_layer_order(radix_tree.root)
    
    
    print(f"================================evict==========================")
    print(f"all take tokens: {radix_tree.all_taken_tokens_}")
    
    
    radix_tree.evict_node(5)
    display_the_tree_layer_order(radix_tree.root)
    print(f"all take tokens: {radix_tree.all_taken_tokens_}")
    
    