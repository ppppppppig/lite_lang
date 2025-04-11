import torch
from dataclasses import dataclass
import queue
from .cache import Cache, NormalCache
from typing import List

def get_no_padding_messages(input_reqs, is_prefill=True):
    position_ids = []
    input_ids = []
    req_lengths = []
    start_idx = []
    kv_start_idx = [0,]
    for idx, req in enumerate(input_reqs):
        if is_prefill:
            position_ids.extend([i for i in range(len(req.input_tokens))])
            start_idx.append(len(input_ids))
            input_ids.extend(req.input_tokens)
        else:
            position_ids.append(req.length - 1)
            kv_start_idx.append(req.length + kv_start_idx[-1])
            start_idx.append(idx)
        req_lengths.append(req.length)
    kv_start_idx.pop()
    return torch.tensor(input_ids).cuda(), torch.tensor(position_ids).cuda(), torch.tensor(start_idx).cuda(), torch.tensor(req_lengths).cuda(), torch.tensor(kv_start_idx).cuda()

@dataclass
class Req:
    id: int
    prompt: str
    input_tokens: List[int]
    max_total_length: int
    max_output_length: int
    output_prompt_que: queue.Queue
    temperature: float
    top_p: float
    top_k: float
    do_sample: bool
    input_length: int
    output_length: int = 0
    is_end: bool = False
    rid: int = None   
    # 一共有两个队列，分别是运行队列和就绪队列，分配了rid的就是运行队列，否则就是就绪队列
    # 实际上应该有三个队列，此版本暂时不考虑kv cache不够的情况
    def Add(self, text: str, is_eos_token):
        self.output_length += 1
        if is_eos_token or (self.output_length + self.input_length >= self.max_total_length and self.output_length >= self.max_output_length):
            self.is_end = True
        self.output_prompt_que.put((text, self.is_end))
    
    @property
    def length(self):
        return self.input_length + self.output_length


@dataclass
class ModelInput:
    reqs: list[Req]
    input_tokens: torch.Tensor
    is_prefill: bool
    is_batch_end: bool
    position_ids: torch.Tensor
    b_start_idx: torch.Tensor
    b_seq_len: torch.Tensor
    kv_start_idx: torch.Tensor
    past_key_values: Cache = None
    output_tokens: torch.Tensor = None
    
    def update(self, output_tokens: torch.Tensor):
        self.is_prefill = False
        self.output_tokens = output_tokens.squeeze(1)

        self.input_tokens, self.position_ids, self.b_start_idx, self.b_seq_len, self.kv_start_idx = get_no_padding_messages(self.reqs, self.is_prefill)
    
    def get_post_sample_para(self):
        temperatures = torch.tensor([req.temperature for req in self.reqs]).cuda()
        top_p = torch.tensor([req.top_p for req in self.reqs]).cuda()
        top_k = torch.tensor([req.top_k for req in self.reqs]).cuda()
        do_sample = torch.tensor([req.do_sample for req in self.reqs]).cuda()
        return temperatures, top_p, top_k, do_sample
    
    # 为了屏蔽bug，修改这里，只要有请求停止，batch停止, 否则由于kv cache长度限制，可能导致程序崩溃，后面开发连续批处理时解决这个问题
    def view_is_end(self):
        for req in self.reqs:
            if req.is_end:
                self.is_batch_end = True
                return
            
        self.is_batch_end = False
    
    # 更新请求信息，目前只更新了input_length
    def update_reqs_message(self, b_seq_len):
        req_lengths = b_seq_len.tolist()
        for idx, length in enumerate(req_lengths):
            self.reqs[idx].input_length = length


class ReqManager:
    def __init__(self, max_batch_size, tokenizer, max_total_length, max_output_length):
        self.reqs = queue.Queue()
        self.padding_mask = None
        self.max_batch_size_ = max_batch_size
        self.tokenizer_ = tokenizer
        self.now_id = 0
        self.max_id = 100000
        self.max_total_length_ = max_total_length
        self.max_output_length_ = max_output_length
        
    def add(self, prompt, top_p, top_k, temperature, do_sample):
        
        input_tokens = self.tokenizer_.encode_single(prompt)[0]
        req = Req(id=self.now_id % self.max_id, prompt=prompt, input_tokens = input_tokens.tolist(), max_total_length=self.max_total_length_,
            max_output_length=self.max_output_length_, output_prompt_que=queue.Queue(), top_p=top_p, top_k=top_k, temperature=temperature, do_sample=do_sample,
            input_length=len(input_tokens.tolist()))
        
        self.now_id += 1
        self.reqs.put(req)
        return req            

    def get_req_batch(self) -> ModelInput:
        input_reqs = []
        while not self.reqs.empty() and len(input_reqs) < self.max_batch_size_:
            input_reqs.append(self.reqs.get())
        if len(input_reqs) == 0:
            return None

        input_tokens, position_ids, b_start_idx, b_seq_len, kv_start_idx = get_no_padding_messages(input_reqs, is_prefill=True)
        
        model_inputs = ModelInput(
            reqs = input_reqs,
            input_tokens = input_tokens,
            is_prefill = True,
            position_ids = position_ids,
            is_batch_end = False,
            b_start_idx = b_start_idx,
            b_seq_len = b_seq_len,
            kv_start_idx=kv_start_idx
        )
        
        model_inputs.update_reqs_message(b_seq_len)
        return model_inputs
        
    def update_req(self, model_outputs, output_token_ids):
        output_prompts = self.tokenizer_.decode(output_token_ids)
        for req_idx in range(len(output_prompts)):
            model_outputs.reqs[req_idx].Add(output_prompts[req_idx], output_token_ids[req_idx, :] == self.tokenizer_.eos_token_id)
        model_outputs.update(output_token_ids)
        model_outputs.view_is_end()
        return model_outputs
    