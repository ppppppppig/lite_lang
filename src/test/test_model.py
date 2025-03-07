import os
import torch
import time
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from qwen2.model import Qwen2Model, Qwen2Config

def test(model_path, device):
    torch.cuda.set_device(device)
    config_path = os.path.join(model_path, 'config.json')
    config = Qwen2Config(config_path)
    max_req_length = 1024
    model = Qwen2Model(config.layer_nums, max_req_length, config.hidden_size)
    model.load_weight(model_path)

    mock_input_token = [23, 24, 25]
    output_token = model.forward(mock_input_token)
    print(f"output_token: {output_token}")


def perf_test(model_path, device):
    config_path = os.path.join(model_path, 'config.json')
    config = Qwen2Config(config_path)
    
    # 构建输入张量
    max_req_length = 1024
    batch_size = 64
        
    mock_prefill_tensor = torch.ones([batch_size, max_req_length], dtype=torch.float16, device='cuda')
    
    model = Qwen2Model(config.layer_nums, max_req_length, config.hidden_size)

    # warmup
    output_token = model.forward(mock_prefill_tensor)
    
    torch.cuda.synchronize()
    st_time = time.perf_counter()
    output_token = model.forward(mock_prefill_tensor)
    torch.cuda.synchronize()
    ed_time = time.perf_counter()
    print(f"spend time is : {ed_time - st_time}")
    
    
    print(f"output_token: {output_token}")
    
test("/root/LiteLang/models/Qwen2-1.5B", "cuda:3")