import queue
from litelang.tools.tools import get_unique_id
from litelang.scheduler.req_queue import Req, ReqStatus
from litelang.scheduler.req_queue import SchedulerRunnerQue, ReadyQue
import time

class Scheduler:
    def __init__(
        self,
        max_batch_size,
        tokenizer,
        max_total_length,
        max_output_length,
        max_reqs,
        model_client,
        busy_scale,
    ):
        self.model_client_ = model_client
        max_sum_kv_tokens = self.model_client_.init_model()
        self.ready_and_swapped_que_ = ReadyQue(
            max_reqs,
            max_total_length - max_output_length,
            max_output_length,
            max_sum_kv_tokens,
            max_batch_size,
            busy_scale,
        )
        self.running_que_ = SchedulerRunnerQue(
            max_batch_size, get_unique_id(), busy_scale, max_sum_kv_tokens
        )
        self.new_running_que_ = SchedulerRunnerQue(
            max_batch_size, get_unique_id(), busy_scale, max_sum_kv_tokens
        )
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
        req = Req(
            id=self.now_id % self.max_id,
            status=ReqStatus.READY,
            prompt=prompt,
            input_tokens=input_tokens.tolist(),
            output_tokens=[],
            max_total_length=self.max_total_length_,
            max_output_length=self.max_output_length_,
            output_prompt_que=queue.Queue(),
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            do_sample=do_sample,
            input_length=len(input_tokens.tolist()),
            eos_token=self.tokenizer_.eos_token_id,
        )

        self.now_id += 1
        self.ready_and_swapped_que_.add_req(req)
        return req

    # 之前的batch为空的时候，重新生成batch
    def construct_running_que(self):
        self.running_que_ = self.ready_and_swapped_que_.try_generate_new_batch(
            self.running_que_
        )
        return True

    # 之前的batch不为空时，尝试生成新的batch
    def try_construct_new_running_que(self):
        self.new_running_que_ = self.ready_and_swapped_que_.try_generate_new_batch(
            self.running_que_
        )

    def should_forward(self):
        if self.running_que_.is_empty():
            return False
        return True

    def merge_runner_batch(self):
        self.model_client_.merge_batch(
            self.running_que_.batch_id, self.new_running_que_.batch_id
        )
        self.running_que_.update(self.new_running_que_)

    def update_from_forward(self, output_token_ids, req_ids):
        output_prompts = self.tokenizer_.decode(output_token_ids)
        stop_req_ids = self.running_que_.update_from_forward(
            output_prompts, output_token_ids, req_ids, self.tokenizer_.eos_token_id
        )
        return stop_req_ids

    def model_prefill(self, batch_id):
        st_time = time.perf_counter()
        self.model_client_.add_batch(self.running_que_.get_transfer_data(), batch_id)
        output_token_ids, req_ids = self.model_client_.prefill(batch_id)
        self.update_from_forward(output_token_ids, req_ids)
        ed_time = time.perf_counter()
        print(f"prefill throughput: {output_token_ids.size(0) / (ed_time - st_time)} tokens/s")
        return True

    def model_decode(self, batch_id):
        st_time = time.perf_counter()
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
        ed_time = time.perf_counter()
        print(f"decode throughput: {output_token_ids.size(0) / (ed_time - st_time)} tokens/s")
        return True

    def new_update_from_forward(self, output_token_ids, req_ids):
        output_prompts = self.tokenizer_.decode(output_token_ids)
        stop_req_ids = self.new_running_que_.update_from_forward(
            output_prompts, output_token_ids, req_ids, self.tokenizer_.eos_token_id
        )
        return stop_req_ids

    def new_model_prefill(self, batch_id):
        st_time = time.perf_counter()
        self.model_client_.add_batch(
            self.new_running_que_.get_transfer_data(), batch_id
        )
        output_token_ids, req_ids = self.model_client_.prefill(batch_id)
        self.new_update_from_forward(output_token_ids, req_ids)
        ed_time = time.perf_counter()
        print(f"prefill throughput: {output_token_ids.size(0) / (ed_time - st_time)} tokens/s")
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
