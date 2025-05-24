import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import click

def construct_mock_messages(max_batch_size: int, every_prefill_length: int):

    # Mocking the request messages
    input_tokens = []
    for i in range(max_batch_size):
        input_tokens.append([10] * every_prefill_length)
    input_tokens = torch.tensor(input_tokens, dtype=torch.int32).cuda()
    return input_tokens

def statistics_inference(model_name, max_new_tokens, max_batch_size, every_prefill_length):
    # 加载模型和分词器
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    print(f"model_path: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda().eval()

    input_ids = construct_mock_messages(max_batch_size, every_prefill_length)
    torch.cuda.empty_cache()
    def warmup():
        # 预热模型
        with torch.no_grad():
            model.generate(
                input_ids=input_ids,
                max_new_tokens=1,
                temperature=0.7,
                top_p=0.95,
                do_sample=False,
            )

    warmup()
    torch.cuda.empty_cache()
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


@click.command()
@click.option(
    "--model_path", default="/root/LiteLang/models/Qwen2-1.5B/", help="权重路径"
)
@click.option("--max_prefill_length", default=600, type=int, help="最大输入长度")
@click.option("--max_new_tokens", default=1000, type=int, help="最大输出长度")
@click.option("--max_batch_size", default=30, type=int, help="最大batchsize")
def test(model_path, max_prefill_length, max_new_tokens, max_batch_size):
    statistics_inference(model_path, max_new_tokens, max_batch_size, max_prefill_length)
    
if __name__ == "__main__":
    test()