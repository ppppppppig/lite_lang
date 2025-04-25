from litelang.scheduler.rpc.model_rpc import ModelClientRPC
from litelang.scheduler.scheduler import Scheduler
from litelang.models.qwen2.model import Qwen2ModelRunner, Qwen2Config
from litelang.scheduler.tokenizer.tokenizer import Tokenizer
import os
import threading


# 目前只支持单TP，只能加载到同一个GPU上
class HttpServerManager:

    def __init__(
        self,
        model_path,
        max_input_length,
        max_output_length,
        max_batch_size,
        mem_usage,
        tp,
        max_reqs,
        busy_scale,
        use_radix_cache,
    ):
        max_total_length = max_input_length + max_output_length
        config = Qwen2Config(os.path.join(model_path, "config.json"))
        model_client = ModelClientRPC(
            tp,
            max_batch_size,
            max_input_length,
            config,
            model_path,
            max_total_length,
            mem_usage,
            use_radix_cache,
        )
        self.scheduler_ = Scheduler(
            max_batch_size=max_batch_size,
            tokenizer=Tokenizer(model_path),
            max_total_length=max_total_length,
            max_output_length=max_output_length,
            max_reqs=max_reqs,
            model_client=model_client,
            busy_scale=busy_scale,
        )
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
            self.scheduler_.forward()
