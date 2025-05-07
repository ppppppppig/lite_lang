from awq_quant.quantization.quantization import AutoQuantizer
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

if __name__ == "__main__":
    
    model_for_causal_lm = AutoModelForCausalLM.from_pretrained("/root/LiteLang/models/Qwen2-1.5B")
    tokenizer = AutoTokenizer.from_pretrained("/root/LiteLang/models/Qwen2-1.5B")
    dataset_path = "/root/LiteLang/quantize/val.jsonl.zst"
    quantizer = AutoQuantizer(
        model=model_for_causal_lm,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        input_len=128,
        num_samples=4,
        per_forward_batch_num=10,
        split="train",
        output_path="/root/LiteLang/models/quantize_model_qwen2-1.5B",
        zero_point=True,
        group_size=128
    )
    quantizer.quantization()
    quantizer.save_models()