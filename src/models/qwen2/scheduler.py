import torch
from dataclasses import dataclass
import queue
from .cache import Cache, NormalCache
from typing import List
from enum import Enum
from models.tools.tools import get_unique_id

class ReqStatus(Enum):
    READY = 0
    RUNNER = 1
    SWAPPED = 2  # 暂时不支持这种

@dataclass
class Req:
    id: int
    status: ReqStatus
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
    is_prefill: bool = True
    output_length: int = 0
    is_end: bool = False

    def Add(self, text: str, is_eos_token):
        self.output_length += 1
        if is_eos_token or (self.output_length + self.input_length >= self.max_total_length and self.output_length >= self.max_output_length):
            self.is_end = True
        self.output_prompt_que.put((text, self.is_end))
    
    @property
    def length(self):
        return self.input_length + self.output_length
    
    def next_forward_need_tokens(self):
        if self.is_prefill:
            return self.input_length
        else:
            return 1

class SchedulerRunnerQue:
    def __init__(self):
        self.reqs = []
        self.is_end_ = False
    
    def add_req(self, req: Req):
        self.reqs.append(req)
        
    def update_from_forward(self, output_prompts, output_token_ids, eos_token_id):
        self.is_end_ = True
        stop_req_ids = []
        for req_idx in range(len(output_prompts)):
            self.reqs[req_idx].Add(output_prompts[req_idx], output_token_ids[req_idx, :] == eos_token_id)
            if self.reqs[req_idx].is_end  == False:
                self.is_end_ = False
            else:
                stop_req_ids.append(self.reqs[req_idx].id)
        return stop_req_ids

    def is_end(self):
        return self.is_end_
    
    def is_empty(self):
        return len(self.reqs) == 0
    
    def get_transfer_data(self):
        reqs_data = []
        for req in self.reqs:
            input_tokens = req.input_tokens
            reqs_data.append({
                'id': req.id,
                "input_tokens": input_tokens,
                "temperature": req.temperature,
                "top_p": req.top_p,
                "top_k": req.top_k,
                "do_sample": req.do_sample,
            })
        return reqs_data

class ReadyQue:
    def __init__(self, max_reqs, max_input_length, max_output_length, max_sum_kv_tokens):
        self.max_reqs_ = max_reqs
        self.max_input_length_ = max_input_length
        self.max_output_length_ = max_output_length
        self.max_sum_kv_tokens_ = max_sum_kv_tokens
        self.reqs = queue.Queue()
    
    def add_req(self, req: Req):
        if self.reqs.qsize() >= self.max_reqs_:
            return False
        self.reqs.put(req)
        return True
    
    def get_new_reqs(self):
        runner_reqs = SchedulerRunnerQue()
        while self.reqs.empty() is False:
            if self.reqs.queue[0].status != ReqStatus.READY:
                assert 1 == 0, "请求状态错误"
    
            if self.reqs.queue[0].next_forward_need_tokens() > self.max_sum_kv_tokens_:
                break
            req = self.reqs.get()
            self.max_sum_kv_tokens_ -= req.next_forward_need_tokens()
            runner_reqs.add_req(req)
        return runner_reqs

class Scheduler:
    def __init__(self, max_batch_size, tokenizer, max_total_length, max_output_length, max_reqs, max_sum_kv_tokens):
        self.ready_que_ = ReadyQue(max_reqs, max_total_length - max_output_length, max_output_length, max_sum_kv_tokens)
        self.swapped_que_ = queue.Queue()
        self.running_que_ = SchedulerRunnerQue()
        self.running_batch_id_ = []

        self.max_batch_size_ = max_batch_size
        self.tokenizer_ = tokenizer
        self.now_id = 0
        self.max_id = 100000
        self.max_total_length_ = max_total_length
        self.max_output_length_ = max_output_length

    def add_req(self, prompt, top_p, top_k, temperature, do_sample):
        input_tokens = self.tokenizer_.encode_single(prompt)[0]
        req = Req(id=self.now_id % self.max_id, status=ReqStatus.READY, prompt=prompt, input_tokens=input_tokens.tolist(), max_total_length=self.max_total_length_,
            max_output_length=self.max_output_length_, output_prompt_que=queue.Queue(), top_p=top_p, top_k=top_k, temperature=temperature, do_sample=do_sample,
            input_length=len(input_tokens.tolist()))
        
        self.now_id += 1
        self.ready_que_.add_req(req)
        return req
    
    def get_runner_batch(self):
        self.running_que_ = self.ready_que_.get_new_reqs()
        batch_id = get_unique_id()
        return self.running_que_, batch_id
    
    def update_from_forward(self, output_token_ids):
        output_prompts = self.tokenizer_.decode(output_token_ids)
        stop_req_ids = self.running_que_.update_from_forward(output_prompts, output_token_ids, self.tokenizer_.eos_token_id)
        return stop_req_ids

    def filter_stop_req(self):
        