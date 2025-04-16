from .model_rpc import ModelClientRPC
from .scheduler import Scheduler
from .model import Qwen2ModelRunner, Qwen2Config
from models.tokenizer import Tokenizer
import torch
import os
import threading
import time

# 目前只支持单TP，只能加载到同一个GPU上
class HttpServerManager:

    def __init__(self, model_path, max_input_length, max_output_length, max_batch_size, mem_usage, tp, max_reqs):
        max_total_length = max_input_length + max_output_length
        config = Qwen2Config(os.path.join(model_path, 'config.json'))
        self.model_client_ = ModelClientRPC(tp, max_batch_size, max_input_length, config, model_path, max_total_length, mem_usage)
        max_sum_kv_tokens = self.model_client_.init_model()
        self.scheduler_ = Scheduler(max_batch_size=max_batch_size, tokenizer=Tokenizer(model_path),
                                       max_total_length=max_total_length, max_output_length=max_output_length,
                                       max_reqs=max_reqs, max_sum_kv_tokens=max_sum_kv_tokens)
        self._thread = threading.Thread(target=self.handle_loop)
        self._thread.start()
    
    # 流式生成接口,这里只有流式生成接口，后续再合并
    def generate(self, prompt, top_p, top_k, temperature, do_sample):
        req = self.scheduler_.add_req(prompt, top_p, top_k, temperature, do_sample)
        
        while True:
            token, is_end = req.output_prompt_que.get()
            yield token
            if is_end:
                break
        
        return
    
    # 启动一个新线程
    def handle_loop(self):
        while True:
            runner_batch, batch_id = self.scheduler_.get_runner_batch()
            if runner_batch.is_empty():
                continue
            start_time = time.perf_counter()
            all_tokens_length = 0
            self.model_client_.add_batch(runner_batch, batch_id)
            output_token_ids = self.model_client_.prefill(batch_id)
            self.scheduler_.update_from_forward(output_token_ids)
            while not runner_batch.is_end():
                output_token_ids = self.model_client_.decode(batch_id)
                stop_req_ids = self.scheduler_.update_from_forward(output_token_ids)
                if len(stop_req_ids):
                    self.model_client_.filter
            self.model_client_.remove_batch(batch_id)
            for req in runner_batch.reqs:
                all_tokens_length += req.length
            end_time = time.perf_counter()
            tokens_per_second = all_tokens_length / (end_time - start_time)
            print(f"生成速度: {tokens_per_second:.2f} tokens/秒")

    # # 每推理十次，尝试进行continous_batch
    # def continous_batch(self):
    #     times = 0
    #     decode_batch_id = 0
    #     while True:
    #         times += 1
    #         if times % 10 == 0:
    #             times = 0
    #             # 尝试看看是否能获取新的请求
    #             runner_batch, prefill_batch_id = self.scheduler_.get_runner_batch()
    #             if runner_batch.is_empty():
    #                 continue
    #             self.model_client_.add_batch(runner_batch, prefill_batch_id)
    #             output_token_ids = self.model_client_.prefill(prefill_batch_id)
    #             self.scheduler_.update_from_forward(output_token_ids)
                
    #             # 跟以前的请求合并
    #             decode_batch_id = self.model_client_.merge_batch(decode_batch_id, prefill_batch_id)
                
    #         self.model_client_.decode(decode_batch_id)