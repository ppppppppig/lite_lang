from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 定义模型路径
model_path = "/root/LiteLang/models/Qwen2-1.5B/"  # 替换为你的模型路径

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 输入字符串
input_text = "请写一段故事, 关于爱和和平"

# 对输入字符串进行编码
inputs = tokenizer(input_text, return_tensors="pt")
inputs.input_ids = inputs.input_ids.cuda()
# 将输入传递给模型，获取输出
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_length=50,  # 设置最大生成长度
        max_new_tokens=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
outputs = outputs.cpu()
# 解码输出
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 输出最后一个字符
print("生成的文本:", decoded_output)
print("最后一个字符:", decoded_output[-1])