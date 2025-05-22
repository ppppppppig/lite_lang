from safetensors import safe_open
import torch


class MMWeight:
    def __init__(self, weights, proj_name, bias_name, torch_dtype):
        self.proj = weights[proj_name].cuda()
        self.bias = (
            weights[bias_name].cuda().to(torch_dtype)
            if bias_name in weights
            else None
        )


class RowMMWeight:
    def __init__(self, weights, proj_name, bias_name, tp_rank, world_size, torch_dtype):
        height, width = weights[proj_name].shape
        assert (
            height % world_size == 0
        ), f"row slice, height: {height} % world_size: {world_size} should equal 0"
        single_gpu_height = height // world_size
        self.proj = (
            weights[proj_name][
                single_gpu_height * tp_rank : single_gpu_height * (tp_rank + 1)
            ]
            .to(torch_dtype)
            .cuda()
        )
        self.bias = (
            weights[bias_name][
                single_gpu_height * tp_rank : single_gpu_height * (tp_rank + 1)
            ]
            .to(torch_dtype)
            .cuda()
            if bias_name in weights
            else None
        )


class ColMMWeight:
    def __init__(self, weights, proj_name, bias_name, tp_rank, world_size, torch_dtype):
        height, width = weights[proj_name].shape
        assert (
            width % world_size == 0
        ), f"col slice, width: {width} % world_size: {world_size} should equal 0"
        single_gpu_width = width // world_size
        self.proj = (
            weights[proj_name][
                :, single_gpu_width * tp_rank : single_gpu_width * (tp_rank + 1)
            ]
            .to(torch_dtype)
            .cuda()
        )
        self.bias = (
            weights[bias_name][
                single_gpu_width * tp_rank : single_gpu_width * (tp_rank + 1)
            ]
            .to(torch_dtype)
            .cuda()
            if bias_name in weights
            else None
        )


# 该算子不做切分
class LayerNormWeight:
    def __init__(self, weights, proj_name, bias_name, tp_rank, world_size, torch_dtype):
        self.proj = weights[proj_name].to(torch_dtype).cuda()

        self.bias = (
            weights[bias_name].to(torch_dtype).cuda()
            if bias_name in weights
            else None
        )


class Qwen2LayerWeight:
    def __init__(self, layer_num, tp_rank, world_size):
        self.layer_num_ = layer_num
        self.model_type_ = "qwen2.5"
        assert world_size in (1, 2, 4, 8), "world size should be one of 1, 2, 4, 8"
        self.world_size_ = world_size
        self.tp_rank_ = tp_rank
        self.torch_dtype = torch.float16

    def _init_qkv(self, weights):
        q_proj_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"
        q_bias_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.bias"

        k_proj_name = f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"
        k_bias_name = f"model.layers.{self.layer_num_}.self_attn.k_proj.bias"

        v_proj_name = f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"
        v_bias_name = f"model.layers.{self.layer_num_}.self_attn.v_proj.bias"

        o_proj_name = f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"
        o_bias_name = f"model.layers.{self.layer_num_}.self_attn.o_proj.bias"

        self.q_proj = RowMMWeight(
            weights, q_proj_name, q_bias_name, self.tp_rank_, self.world_size_, self.torch_dtype
        )
        self.k_proj = RowMMWeight(
            weights, k_proj_name, k_bias_name, self.tp_rank_, self.world_size_, self.torch_dtype
        )
        self.v_proj = RowMMWeight(
            weights, v_proj_name, v_bias_name, self.tp_rank_, self.world_size_, self.torch_dtype
        )
        self.o_proj = ColMMWeight(
            weights, o_proj_name, o_bias_name, self.tp_rank_, self.world_size_, self.torch_dtype
        )

    def _init_input_layernorm(self, weights):
        layernorm_weight = f"model.layers.{self.layer_num_}.input_layernorm.weight"
        layernorm_bias = None

        self.layernorm = LayerNormWeight(
            weights, layernorm_weight, layernorm_bias, self.tp_rank_, self.world_size_, self.torch_dtype
        )

    def _init_post_attn_layernorm(self, weights):
        layernorm_weight = (
            f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"
        )
        layernorm_bias = None

        self.post_attn_layernorm = LayerNormWeight(
            weights, layernorm_weight, layernorm_bias, self.tp_rank_, self.world_size_, self.torch_dtype
        )

    def _init_ffn(self, weights):
        gate_weight = f"model.layers.{self.layer_num_}.mlp.gate_proj.weight"
        down_weight = f"model.layers.{self.layer_num_}.mlp.down_proj.weight"
        up_weight = f"model.layers.{self.layer_num_}.mlp.up_proj.weight"
        hidden_size, intermediate_size = weights[down_weight].shape
        assert (
            intermediate_size % self.world_size_ == 0
        ), "It should be intermediate_size % worldd_size == 0"
        single_gpu_intermediate_size = intermediate_size // self.world_size_
        self.down_proj = (
            weights[down_weight][
                :,
                single_gpu_intermediate_size
                * self.tp_rank_ : single_gpu_intermediate_size
                * (self.tp_rank_ + 1),
            ]
            .cuda()
            .to(self.torch_dtype)
        )
        self.gate_proj = (
            weights[gate_weight][
                single_gpu_intermediate_size
                * self.tp_rank_ : single_gpu_intermediate_size
                * (self.tp_rank_ + 1),
                :,
            ]
            .cuda()
            .to(self.torch_dtype)
        )
        self.up_proj = (
            weights[up_weight][
                single_gpu_intermediate_size
                * self.tp_rank_ : single_gpu_intermediate_size
                * (self.tp_rank_ + 1),
                :,
            ]
            .cuda()
            .to(self.torch_dtype)
        )

    def init(self, weights):
        self._init_ffn(weights)
        self._init_post_attn_layernorm(weights)
        self._init_input_layernorm(weights)
        self._init_qkv(weights)
