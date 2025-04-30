import torch

import torch.nn as nn

@torch.no_grad()
def quantize_weight_per_channel_absmax(w):
    # w: [out_channel, in_channel]
    scales = w.abs().max(dim=1)[0] / 127
    scales = scales.view(-1, 1)
    if not w.is_cuda:
        # half rounding is not supported on CPU
        w = w.float()
    # use inplace operation to save memory
    w.div_(scales).round_().clamp_(-128, 127)
    w_q = w.to(torch.int8)
    return w_q, scales


class W8A8BFP32OFP32Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        # 仅保存 int8 权重和缩放因子（scale）
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=torch.int8, requires_grad=False),
        )
        self.register_buffer(
            "weight_scale",  # 权重量化的缩放因子
            torch.tensor(1.0, dtype=torch.float16, requires_grad=False)
        )
        if self.use_bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, dtype=torch.float16, requires_grad=False),
            )

    @staticmethod
    def from_float(module: nn.Linear):
        # 创建量化后的模块实例
        quantized = W8A8BFP32OFP32Linear(
            in_features=module.in_features,
            out_features=module.out_features,
            use_bias=module.bias is not None
        )

        # 量化权重：float16 -> int8
        int8_weight, weight_scale = quantize_weight_per_channel_absmax(module.weight)
        quantized.weight.data = int8_weight
        quantized.weight_scale.data = weight_scale

        # 保留原模型的偏置（保持 float16）
        if quantized.use_bias:
            quantized.bias.data = module.bias.to(torch.float16)

        return quantized


class W8A8BFP32OFP32LinearWithQuantScale(W8A8BFP32OFP32Linear):
    def __init__(self, in_features: int, out_features: int, use_bias: bool = False):
        super().__init__(in_features, out_features, use_bias )
        self.register_buffer(
            "quant_scale",
            torch.tensor(1.0, dtype=torch.float32, requires_grad=False),
        )

    @staticmethod
    def from_float(
        module: nn.Linear,
        input_scale: float,  # 输入缩放因子（若有需要）
        save_device=torch.device("cpu")
    ):
        # 创建量化模块实例
        quantized = W8A8BFP32OFP32LinearWithQuantScale(
            in_features=module.in_features,
            out_features=module.out_features,
            use_bias=module.bias is not None
        )

        # 量化权重：float16 -> int8
        int8_weight, weight_scale = quantize_weight_per_channel_absmax(module.weight)
        
        quantized.quant_scale = torch.tensor(input_scale, dtype=torch.float32).to(save_device)
        
        quantized.weight.data = int8_weight.to(save_device)
        quantized.weight_scale.data = weight_scale.to(save_device)

        # 保留原模型的偏置（保持 float16）
        if quantized.use_bias:
            quantized.bias.data = module.bias.to(torch.float16).to(save_device)

        return quantized