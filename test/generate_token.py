from litelang.models.qwen2.model import Qwen2ModelRunner
from litelang.models.qwen2.model import Qwen2Config
from litelang.models.runner_batch import RunnerBatch, RunnerReq
import time
import torch

def construct_mock_messages(max_batch_size: int, every_prefill_length: int):
    # Mocking the request messages
    req_messages = []
    for i in range(max_batch_size):
        req_messages.append({
            "id": i,
            "input_tokens": [10] * every_prefill_length,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": False,
            "ignore_eos": True
        })
    return req_messages

def statistics_inference(model_path, decode_nums, max_batch_size, every_prefill_length):
    config_path = f"{model_path}/config.json"
    qwen2_config = Qwen2Config(config_path)
    model_runner = Qwen2ModelRunner(layer_nums=qwen2_config.layer_nums, max_batch_size=max_batch_size,
                                    config=qwen2_config, tp_rank=0, world_size=1, model_path=model_path,
                                        max_length=1300, mem_usage=0.5, use_radix_cache=False)
    
    model_runner.init_model()
    batch_id = 'abcd-1234-efdg'
    
    req_messages = construct_mock_messages(max_batch_size, every_prefill_length)

    batch = RunnerBatch.create_batch(
        req_messages,
        batch_id,
        is_prefill=True,
        model_runner=model_runner
    )
    
    def warmup():

        model_runner.forward(batch)
        model_runner.free_reqs(batch.request_mapping.values())
    
    warmup()
    
    
    batch = RunnerBatch.create_batch(
        req_messages,
        batch_id,
        is_prefill=True,
        model_runner=model_runner
    )
    
    torch.cuda.synchronize()
    st_time = time.perf_counter()
    for i in range(0, decode_nums):
        output_token_ids = model_runner.forward(batch)
        batch.update_forward_message(output_token_ids)
    torch.cuda.synchronize()
    ed_time = time.perf_counter()
    print(f"decode throughput: {(decode_nums * max_batch_size) / (ed_time - st_time)} tokens/s")
    
    
model_path = "/root/LiteLang/models/Qwen2-1.5B/"

max_batch_size=100
every_prefill_length=200
max_new_tokens=1

statistics_inference(model_path, max_new_tokens, max_batch_size, every_prefill_length)