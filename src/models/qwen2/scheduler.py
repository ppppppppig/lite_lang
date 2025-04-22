import torch
from dataclasses import dataclass
import queue
from .cache import Cache, NormalCache
from typing import List
from enum import Enum
from models.tools.tools import get_unique_id
import numpy as np
import time
from models.qwen2.pause_strategy import select_swapped_reqs, STRATEGY_REGISTRY
from collections import deque

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
    output_tokens: List[int]
    max_total_length: int
    max_output_length: int
    output_prompt_que: queue.Queue
    temperature: float
    top_p: float
    top_k: float
    do_sample: bool
    input_length: int
    eos_token: int
    is_prefill: bool = True
    output_length: int = 0
    is_end: bool = False

    def Add(self, text: str, output_token):
        self.is_prefill = False
        self.output_length += 1
        is_eos_token = (output_token == self.eos_token)
        if is_eos_token or (self.output_length + self.input_length >= self.max_total_length and self.output_length >= self.max_output_length):
            self.is_end = True
        self.output_prompt_que.put((text, self.is_end))
        self.output_tokens.append(output_token.item())

    @property
    def length(self):
        return self.input_length + self.output_length

    def next_forward_need_tokens(self):
        if self.is_prefill:
            return self.input_length
        else:
            return 1

    def need_tokens(self, is_system_busy):
        has_been_take_tokens = self.input_length + self.output_length
        # 这里如果系统请求量不大时，假设所有的请求最大生成长度 乘以 0.6
        system_temperature = 1 if is_system_busy else 0.6
        expect_remaining_tokens = self.max_output_length * system_temperature
        if self.output_length >= expect_remaining_tokens:
            expect_remaining_tokens = self.max_output_length
        return [has_been_take_tokens, expect_remaining_tokens]

class SchedulerRunnerQue:
    def __init__(self, max_batch_size, batch_id, busy_scale, max_sum_kv_tokens, swapped_strategy="LongestFirstStrategy"):
        self.req_mappings = {}
        self.is_end_ = False
        self.is_prefill_ = False
        self.max_batch_size_ = max_batch_size
        self.busy_scale_ = busy_scale
        self.batch_id = batch_id
        self.has_been_consume_tokens = 0
        self.max_sum_kv_tokens_ = max_sum_kv_tokens
        
        # 默认将最长的请求给调度出来
        self.strategy_ = STRATEGY_REGISTRY[swapped_strategy]()
        
    def add_req(self, req: Req):
        self.req_mappings[req.id] = req
    
    def req_count(self):
        return len(self.req_mappings)
    
    def update_from_forward(self, output_prompts, output_token_ids, req_ids, eos_token_id):
        # output 的时候要携带更多信息，否则很容易出错
        self.is_end_ = True
        self.is_prefill_ = False
        stop_req = []
        remain_reqs = {}
        has_been_consume_tokens = 0
        
        for idx, text in enumerate(output_prompts):
            now_req = self.req_mappings[req_ids[idx]]
            now_req.Add(text, output_token_ids[idx, :])
            if now_req.is_end  == False:
                self.is_end_ = False
                remain_reqs[now_req.id] = now_req
                has_been_consume_tokens += now_req.length
            else:
                stop_req.append(now_req)
        
        self.has_been_consume_tokens = has_been_consume_tokens
        self.req_mappings = remain_reqs
        return stop_req

    def is_end(self):
        return self.is_end_
    
    def is_empty(self):
        return len(self.req_mappings) == 0
    
    def get_transfer_data(self):
        reqs_data = []
        for req in self.req_mappings.values():
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
            
    def _next_forward_need_tokens(self):
        return sum(req.next_forward_need_tokens() for req in self.req_mappings.values())
    
    def can_decode(self):
        return self.has_been_consume_tokens + self._next_forward_need_tokens() - self.max_sum_kv_tokens_
    
    def try_swapped_reqs(self):
        swapped_reqs = []
        need_more_tokens = self.has_been_consume_tokens + self._next_forward_need_tokens() - self.max_sum_kv_tokens_
        if need_more_tokens > 0:
            swapped_reqs = select_swapped_reqs(self.req_mappings.values(), self.strategy_, need_more_tokens)
            for swapped_req in swapped_reqs:
                del self.req_mappings[swapped_req.id]
        return swapped_reqs
    
    def update(self, new_runner_batch):
        self.req_mappings.update(new_runner_batch.req_mappings)
        self.is_end_ = new_runner_batch.is_end_ | self.is_end_
        self.is_prefill_ = False
        return True
      
class ReadyQue:
    def __init__(self, max_reqs, max_input_length, max_output_length, max_sum_kv_tokens, max_batch_size, busy_scale):
        self.max_reqs_ = max_reqs
        self.max_input_length_ = max_input_length
        self.max_output_length_ = max_output_length
        self.max_sum_kv_tokens_ = max_sum_kv_tokens
        self.max_batch_size_ = max_batch_size
        self.busy_scale_ = busy_scale
        self.system_busy_threshold_ = self.max_sum_kv_tokens_ * self.busy_scale_
        self.reqs_ = deque()
        self.swapped_que_ = deque()
    
    def add_req(self, req: Req):
        if len(self.reqs_) >= self.max_reqs_:
            return False
        self.reqs_.append(req)
        return True
    
    def add_swapped_req(self, req: Req):
        self.swapped_que_.appendleft(req)
        self.reqs_.appendleft(req)
        return True
    
    def _init_cache_len(self, reqs, is_system_busy):
        self.cache_len_list_ = [req.need_tokens(is_system_busy) for req in reqs]
        return

    def _is_system_busy(self, last_runner_batch):
        all_taken_token = last_runner_batch.has_been_consume_tokens
        for req in self.swapped_que_:
            all_taken_token += req.length
        return all_taken_token > self.system_busy_threshold_

    def _can_add_new_req(self, new_req, is_system_busy):
        self.cache_len_list_.append(new_req.need_tokens(is_system_busy))
        req_need_tokens = np.array(self.cache_len_list_, dtype=np.int64)
        sorted_indices = np.argsort(req_need_tokens[:, -1])[::-1]  # 降序排序
        sorted_tokens = req_need_tokens[sorted_indices]
        take_tokens = sorted_tokens[:, 0]
        remain_tokens = sorted_tokens[:, 1]
        cumulative_take = np.cumsum(take_tokens)
        indices = np.arange(1, len(self.cache_len_list_) + 1)
        total_estimates = cumulative_take + remain_tokens * indices
        max_total_estimates = np.max(total_estimates).item()
        if max_total_estimates > self.max_sum_kv_tokens_:
            return False
        return True
    
    def try_generate_new_batch(self, last_runner_batch):
        # import pdb; pdb.set_trace()
        new_runner_batch = SchedulerRunnerQue(self.max_batch_size_, get_unique_id(), self.busy_scale_, self.max_sum_kv_tokens_)
        is_system_busy = self._is_system_busy(last_runner_batch)
        self._init_cache_len(last_runner_batch.req_mappings.values(), is_system_busy)
        
        while len(self.reqs_) != 0:
            assert self.reqs_[0].status == ReqStatus.READY or self.reqs_[0].status == ReqStatus.SWAPPED, "请求状态错误"
            if not self._can_add_new_req(self.reqs_[0], is_system_busy):
                break
            
            req = self.reqs_.popleft()
            req.status = ReqStatus.RUNNER
            new_runner_batch.add_req(req)
        if len(self.swapped_que_) > 0:
            for _ in range(min(new_runner_batch.req_count(), len(self.swapped_que_))):
                self.swapped_que_.popleft()
        return new_runner_batch

class Scheduler:
    def __init__(self, max_batch_size, tokenizer, max_total_length, max_output_length, max_reqs, model_client, busy_scale):
        self.model_client_ = model_client
        max_sum_kv_tokens = self.model_client_.init_model()
        self.ready_and_swapped_que_ = ReadyQue(max_reqs, max_total_length - max_output_length, max_output_length, max_sum_kv_tokens, max_batch_size, busy_scale)
        self.running_que_ = SchedulerRunnerQue(max_batch_size, get_unique_id(), busy_scale, max_sum_kv_tokens)
        self.new_running_que_ = SchedulerRunnerQue(max_batch_size, get_unique_id(), busy_scale, max_sum_kv_tokens)
        self.running_batch_id_ = []

        self.max_batch_size_ = max_batch_size
        self.tokenizer_ = tokenizer
        self.now_id = 0
        self.max_id = 100000
        self.max_total_length_ = max_total_length
        self.max_output_length_ = max_output_length
        self.now_decode_times_, self.every_decode_times_try_get_new_batch_ = 0, 10

    def add_req(self, prompt, top_p, top_k, temperature, do_sample):
        input_tokens = self.tokenizer_.encode_single(prompt)[0]
        req = Req(id=self.now_id % self.max_id, status=ReqStatus.READY, prompt=prompt, input_tokens=input_tokens.tolist(), output_tokens=[], max_total_length=self.max_total_length_,
            max_output_length=self.max_output_length_, output_prompt_que=queue.Queue(), top_p=top_p, top_k=top_k, temperature=temperature, do_sample=do_sample,
            input_length=len(input_tokens.tolist()), eos_token=self.tokenizer_.eos_token_id)
        
        self.now_id += 1
        self.ready_and_swapped_que_.add_req(req)
        return req
    
    # 之前的batch为空的时候，重新生成batch
    def construct_running_que(self):
        self.running_que_ = self.ready_and_swapped_que_.try_generate_new_batch(self.running_que_)
        return True
    
    # 之前的batch不为空时，尝试生成新的batch
    def try_construct_new_running_que(self):
        self.new_running_que_ = self.ready_and_swapped_que_.try_generate_new_batch(self.running_que_)
    
    def should_forward(self):
        if self.running_que_.is_empty():
            return False
        return True
    
    def merge_runner_batch(self):
        self.model_client_.merge_batch(self.running_que_.batch_id, self.new_running_que_.batch_id)
        self.running_que_.update(self.new_running_que_)
    
    def update_from_forward(self, output_token_ids, req_ids):
        output_prompts = self.tokenizer_.decode(output_token_ids)
        stop_req_ids = self.running_que_.update_from_forward(output_prompts, output_token_ids, req_ids, self.tokenizer_.eos_token_id)
        return stop_req_ids
    
    def model_prefill(self, batch_id):
        self.model_client_.add_batch(self.running_que_.get_transfer_data(), batch_id)
        output_token_ids, req_ids = self.model_client_.prefill(batch_id)
        self.update_from_forward(output_token_ids, req_ids)
        return True
    
    def model_decode(self, batch_id):
        swapped_reqs = self.running_que_.try_swapped_reqs()
        if len(swapped_reqs) > 0:
            # 直接使用类似于暂停的策略即可
            swapped_req_ids = [req.id for req in swapped_reqs]
            self.model_client_.swapped_reqs(batch_id, swapped_req_ids)
            for req in swapped_reqs:
                req.input_tokens.extend(req.output_tokens)
                
                req.output_tokens = []
                req.is_prefill = True
                req.status = ReqStatus.SWAPPED
                self.ready_and_swapped_que_.add_swapped_req(req)
        output_token_ids, req_ids = self.model_client_.decode(batch_id)
        stop_reqs = self.update_from_forward(output_token_ids, req_ids)
        if len(stop_reqs) > 0:
            stop_req_ids = [req.id for req in stop_reqs]
            self.model_client_.filter_batch(batch_id, stop_req_ids)
        return True
    
    def new_update_from_forward(self, output_token_ids, req_ids):
        output_prompts = self.tokenizer_.decode(output_token_ids)
        stop_req_ids = self.new_running_que_.update_from_forward(output_prompts, output_token_ids, req_ids, self.tokenizer_.eos_token_id)
        return stop_req_ids
    
    def new_model_prefill(self, batch_id):
        self.model_client_.add_batch(self.new_running_que_.get_transfer_data(), batch_id)
        output_token_ids, req_ids = self.model_client_.prefill(batch_id)
        self.new_update_from_forward(output_token_ids, req_ids)
        return True
    
    def forward(self):
        has_running_batch = True
        if self.running_que_.is_empty():
            has_running_batch = False
        if not has_running_batch:
            self.construct_running_que()
            
            if self.running_que_.is_empty():
                return

            self.model_prefill(self.running_que_.batch_id)

            while not self.running_que_.is_end():
                self.model_decode(self.running_que_.batch_id)
                self.now_decode_times_ += 1
                if self.now_decode_times_ >= self.every_decode_times_try_get_new_batch_:
                    self.now_decode_times_ = 0
                    break
        else:
            self.try_construct_new_running_que()
            
            if not self.new_running_que_.is_empty():
                self.new_model_prefill(self.new_running_que_.batch_id)
                self.merge_runner_batch()
                
            while not self.running_que_.is_end():
                self.model_decode(self.running_que_.batch_id)
                self.now_decode_times_ += 1
                if self.now_decode_times_ >= self.every_decode_times_try_get_new_batch_:
                    self.now_decode_times_ = 0
                    break
                
        if self.running_que_.is_end():
            self.model_client_.remove_batch(self.running_que_.batch_id)

        