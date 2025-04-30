import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
    Qwen2MLP,
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2PreTrainedModel,
    Qwen2Model,
    Qwen2ForCausalLM,
    Qwen2RotaryEmbedding
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.activations import ACT2FN
from typing import Optional
from .layer import (
    W8A8BFP32OFP32LinearWithQuantScale,
    W8A8BFP32OFP32Linear,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


# 这一层不进行量化
class QuantizedQwen2RMSNorm(Qwen2RMSNorm):

    @staticmethod
    def from_float(module: Qwen2RMSNorm,
                   input_scale: float):
        quantized_module = QuantizedQwen2RMSNorm(module.weight.numel(), module.variance_epsilon)

        quantized_module.weight.to(module.weight.dtype)
        quantized_module.weight = nn.Parameter(module.weight / input_scale)

        return quantized_module

# 这里需要量化
class QuantizedQwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
            self,
            config: Qwen2Config,
            quant_config: dict[str, str],
            layer_idx: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.qkv_quant_type = quant_config["qkv"]
        self.o_quant_type = quant_config["out"]

        self.k_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.q_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.o_proj = W8A8BFP32OFP32LinearWithQuantScale(self.num_heads * self.head_dim, self.hidden_size)
        # self._init_rope()

    # _init_rope = Qwen2Attention._init_rope
    # _shape = Qwen2Attention._shape
    forward = Qwen2Attention.forward
    
    @staticmethod
    @torch.no_grad()
    def from_float_to_int8(module: Qwen2Attention,
                           config: Qwen2Config,
                           quant_config: dict[str, str],
                           attn_input_scale: float,
                           q_output_scale: float,
                           k_output_scale: float,
                           v_output_scale: float,
                           out_input_scale: float):
        int8_module = QuantizedQwen2Attention(config, quant_config)
        # we do not impelement attn for now bacuase we want use paged attention
        int8_module.q_proj = W8A8BFP32OFP32Linear.from_float(module.q_proj)
        int8_module.k_proj = W8A8BFP32OFP32Linear.from_float(module.k_proj)
        int8_module.v_proj = W8A8BFP32OFP32Linear.from_float(module.v_proj)
        int8_module.o_proj = W8A8BFP32OFP32LinearWithQuantScale.from_float(
            module.o_proj, out_input_scale)
        return int8_module


# 这里需要量化
class QuantizedQwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config, quant_config: dict[str, str]):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_quant_type = quant_config["fc1"]
        self.down_quant_type = quant_config["fc2"]

        self.gate_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = W8A8BFP32OFP32LinearWithQuantScale(self.intermediate_size, self.hidden_size)

        self.act_fn = ACT2FN[config.hidden_act]

    forward = Qwen2MLP.forward

    @staticmethod
    @torch.no_grad()
    def from_float_to_int8(module: Qwen2MLP,
                           config: Qwen2Config,
                           quant_config: dict[str, str],
                           gate_input_scale: float,
                           down_input_scale: float):
        int8_module = QuantizedQwen2MLP(config, quant_config)
        int8_module.gate_proj = W8A8BFP32OFP32Linear.from_float(module.gate_proj)
        int8_module.up_proj = W8A8BFP32OFP32Linear.from_float(module.up_proj)
        int8_module.down_proj = W8A8BFP32OFP32LinearWithQuantScale.from_float(
            module.down_proj,
            down_input_scale)
        return int8_module



class QuantizedQwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, quant_config: dict[str, str], layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        # only support Qwen2Attention for now. TODO: support Qwen2FlashAttention2 and Qwen2SdpaAttention
        self.self_attn = QuantizedQwen2Attention(config, quant_config, layer_idx)
        self.mlp = QuantizedQwen2MLP(config, quant_config)
        self.input_layernorm = Qwen2RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    forward = Qwen2DecoderLayer.forward

    @staticmethod
    def from_float_to_int8(module: Qwen2DecoderLayer,
                           config: Qwen2Config,
                           quant_config: dict[str, str],
                           attn_input_scale: float,
                           q_output_scale: float,
                           k_output_scale: float,
                           v_output_scale: float,
                           out_input_scale: float,
                           gate_input_scale: float,
                           down_input_scale: float
                           ):

        quantized_module = QuantizedQwen2DecoderLayer(
            config,
            quant_config,
            module.self_attn.layer_idx
        )
        quantized_module.self_attn = QuantizedQwen2Attention.from_float_to_int8(
            module.self_attn,
            config,
            quant_config,
            attn_input_scale,
            q_output_scale,
            k_output_scale,
            v_output_scale,
            out_input_scale
        )
        quantized_module.mlp = QuantizedQwen2MLP.from_float_to_int8(
            module.mlp,
            config,
            quant_config,
            gate_input_scale,
            down_input_scale
        )
        if quant_config["qkv"] == "per-channel":
            quantized_module.input_layernorm = QuantizedQwen2RMSNorm.from_float(
                module.input_layernorm,
                attn_input_scale
            )
        else:
            quantized_module.input_layernorm = module.input_layernorm
        if quant_config["fc1"] == "per-channel":
            quantized_module.post_attention_layernorm = QuantizedQwen2RMSNorm.from_float(
                module.post_attention_layernorm,
                gate_input_scale
            )
        else:
            quantized_module.post_attention_layernorm = module.post_attention_layernorm
        return quantized_module

    

class QuantizedQwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config, quant_config: dict[str, str]):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([QuantizedQwen2DecoderLayer(config, quant_config, layer_idx) for layer_idx in
                                     range(config.num_hidden_layers)])
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = Qwen2Model.get_input_embeddings
    set_input_embeddings = Qwen2Model.set_input_embeddings
    forward = Qwen2Model.forward
    _update_causal_mask = Qwen2Model._update_causal_mask

    @staticmethod
    def from_float_to_int8(module, decoder_layer_scales, quant_config):

        quantized_module = QuantizedQwen2Model(module.config, quant_config)

        quantized_module.embed_tokens = module.embed_tokens
        quantized_module.norm = module.norm

        for i, layer in enumerate(module.layers):
            quantized_module.layers[i] = QuantizedQwen2DecoderLayer.from_float_to_int8(
                layer, module.config, quant_config, **decoder_layer_scales[i])
        return quantized_module



class QuantizedQwen2ForCausalLM(Qwen2PreTrainedModel):
    def __init__(self, config, quant_config):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.model = QuantizedQwen2Model(config, quant_config)
        # no need to quant
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = Qwen2ForCausalLM.get_input_embeddings
    set_input_embeddings = Qwen2ForCausalLM.set_input_embeddings
    get_output_embeddings = Qwen2ForCausalLM.get_output_embeddings
    set_output_embeddings = Qwen2ForCausalLM.set_output_embeddings
    set_decoder = Qwen2ForCausalLM.set_decoder
    get_decoder = Qwen2ForCausalLM.get_decoder
    forward = Qwen2ForCausalLM.forward
    prepare_inputs_for_generation = Qwen2ForCausalLM.prepare_inputs_for_generation
    _reorder_cache = Qwen2ForCausalLM._reorder_cache

    @staticmethod
    def from_float_to_int8(module, decoder_layer_scales, quant_config):

        new_config = module.config.to_dict()
        print(f"new_config: {new_config}")
        new_config["tie_word_embeddings"] = True  # 显式声明共享
        new_config = Qwen2Config(**new_config)  # 重新创建配置对象
        
        # 使用修改后的配置创建量化模型
        quantized_module = QuantizedQwen2ForCausalLM(new_config, quant_config)
        
        # quantized_module = QuantizedQwen2ForCausalLM(module.config, quant_config)
        print("start perform weight quantization, this might take a while")
        quantized_module.model = QuantizedQwen2Model.from_float_to_int8(
            module.model, decoder_layer_scales, quant_config)
        
        quantized_module.lm_head = module.lm_head
        print(f"nihao")
        print(torch.allclose(module.lm_head.weight, module.model.embed_tokens.weight))
        print(module.lm_head.weight.data_ptr() == module.model.embed_tokens.weight.data_ptr())
        quantized_module.config.tie_word_embeddings = True
        # print(f"lm_head type: {module.lm_head.dtype}")
        return quantized_module
