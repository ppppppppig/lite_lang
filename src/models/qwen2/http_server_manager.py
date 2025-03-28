
from .req_manager import Req, ReqManager
from .model import Qwen2Model, Qwen2Config
from models.tokenizer import Tokenizer
import torch
import os
import threading

# 目前只支持单TP，只能加载到同一个GPU上
class HttpServerManager:

    def __init__(self, model_path, max_input_length, max_output_length, max_batch_size, device):
        max_total_length = max_input_length + max_output_length
        self.device_ = device
        torch.cuda.set_device(device)
        self.req_manager_ = ReqManager(max_batch_size=max_batch_size, tokenizer=Tokenizer(model_path),
                                       max_total_length=max_total_length, max_output_length=max_output_length)
        config = Qwen2Config(os.path.join(model_path, 'config.json'))
        self.model = Qwen2Model(config.layer_nums, max_total_length, config)
        self.model.load_weight(model_path)
        self._thread = threading.Thread(target=self.HandleLoop)
        self._thread.start()
    
    # 流式生成接口,这里只有流式生成接口，后续再合并
    def generate(self, prompt, top_p, top_k, temperature, do_sample):
        req = self.req_manager_.Add(prompt, top_p, top_k, temperature, do_sample)
        
        while True:
            token, is_end = req.output_prompt_que.get()
            yield token
            if is_end:
                break
        
        return
    
    # 启动一个新线程
    def HandleLoop(self):
        import time
        torch.cuda.set_device(self.device_)
        while True:
            model_inputs = self.req_manager_.GetReqBatch()
            if model_inputs is None:
                continue
            while not model_inputs.is_batch_end:
                output_token_ids = self.model.forward(model_inputs)

                model_inputs.Update(output_token_ids)
                self.req_manager_.UpdateReq(model_inputs)
