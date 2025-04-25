from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"  # 设置为左填充
        self.eos_token_id = self.tokenizer.eos_token_id

    def encode(self, prompts):
        input_tokens = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )
        return input_tokens["input_ids"], input_tokens["attention_mask"]

    def encode_single(self, prompt):
        input_tokens = self.tokenizer(
            [
                prompt,
            ],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return input_tokens["input_ids"]

    def decode(self, generate_ids):
        output = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output


def test():
    tokenizer = Tokenizer("/root/LiteLang/models/Qwen2-1.5B")
    input_tokens = tokenizer.encode("请写一段故事:")


if __name__ == "__main__":
    test()
