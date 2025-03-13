import os
import torch
import time
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.qwen2.model import Qwen2Model, Qwen2Config
from models.tokenizer import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel


def test(model_path, device):
    torch.cuda.set_device(device)
    config_path = os.path.join(model_path, 'config.json')
    config = Qwen2Config(config_path)
    max_req_length = 1024
    model = Qwen2Model(config.layer_nums, max_req_length, config)
    model.load_weight(model_path)
    my_tokenizer = Tokenizer(model_path)
    prompts = ['请写一段故事, 关于爱和和平', '写恐怖故事']
    input_tokens, mask = my_tokenizer.encode(prompts)
    mask = mask.cuda()
    past_key_values = None
    all_tokens = None
    output_token, past_key_values = model.forward(input_tokens, mask, past_key_values)
    add_mask = torch.ones([mask.size(0), 1]).cuda()
    mask = torch.cat([mask, add_mask], dim=-1)
    all_tokens = output_token
    for i in range(100):
        output_token, past_key_values = model.forward(output_token, mask, past_key_values)
        mask = torch.cat([mask, add_mask], dim=-1)
        all_tokens = torch.cat([all_tokens, output_token], dim=-1)
    output = my_tokenizer.decode(all_tokens)
    print(f"output: {output}")


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
    
test("/root/LiteLang/models/Qwen2-1.5B", "cuda:2")