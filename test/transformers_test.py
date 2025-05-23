import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer


def construct_mock_messages(max_batch_size: int, every_prefill_length: int):

    # Mocking the request messages
    input_tokens = []
    for i in range(max_batch_size):
        input_tokens.append([10] * every_prefill_length)
    input_tokens = torch.tensor(input_tokens, dtype=torch.int32).cuda()
    return input_tokens

# 加载模型和分词器
model_name = "/root/LiteLang/models/Qwen2-1.5B/"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

max_batch_size=100
every_prefill_length=200
max_new_tokens=1
input_ids = construct_mock_messages(max_batch_size, every_prefill_length)

def warmup():
    # 预热模型
    with torch.no_grad():
        model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
        )

warmup()

torch.cuda.synchronize()
st_time = time.perf_counter()
# 进行推理
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
torch.cuda.synchronize()
ed_time = time.perf_counter()
print(f"decode throughput: {(max_new_tokens * max_batch_size) / (ed_time - st_time)} tokens/s")
# # 解码生成的输出
# decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(decoded)