import torch
from third_party.lmdeploy.lmdeploy.turbomind.deploy.parameter import Weight

class WQLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, n_bits, bias=True):
        super(WQLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        assert n_bits == 4, "n_bits shoule equal 4"

        self.register_buffer('weight', torch.empty(out_features // 32 * n_bits, in_features, dtype=torch.int32))
        
        self.register_buffer('zeros', torch.empty(out_features // 32 * n_bits, dtype=torch.int32))
        
        self.register_buffer('scales', torch.empty(out_features, dtype=torch.float16))
        
        if bias:
            self.register_buffer('bias', torch.empty(out_features // 32 * n_bits).half())
            
    @classmethod
    def create_wqlinear(cls, linear, scales, zeros, in_features, out_features, n_bits, bias=True):
        # scales: [out_features]
        # zeros: [out_features]
        assert n_bits == 4, "n_bits shoulde equal 4"
        
        original_weight = linear.weight
        
        wqlinear = cls(in_features, out_features, n_bits, bias)
        
        # 标准非对称量化流程，zeros是已经量化过了
        def quant_weight(weight, scales, zeros):
            assert weight.size(1) == scales.size(0) == zeros.size(0), "shape not equal"
            # linear做矩阵乘时要转置
            weight = weight.transpose(0, 1)
            weight = torch.round(weight.mul_(scales)) + zeros
            weight = weight.to(torch.int32)
            weight = weight.transpose(0, 1)
            return weight
            
        int_weight = quant_weight(original_weight, scales, zeros)
        zeros = zeros.to(torch.int32)
        wqlinear.scales = scales.half()
        
        if hasattr(linear, "bias"):
            wqlinear.bias = linear.bias.half()
        
        def transformer_weight_to_save_format(weight, n_bit):
            assert n_bit == 4, "n_bits shoulde equal 4"
            order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            pack_num = 32 // n_bit
            quant_weights = []
            for i in range(0, weight.size(1), pack_num):
                quant_weight = torch.zeros(weight.size(0), dtype=torch.int32)
                for j in range(pack_num):
                    channel = weight[:, i + j]
                    channel = channel << order_map[j]
                    quant_weight = quant_weight | channel
                quant_weights.append(quant_weight)
            quant_weights_tensor = torch.cat(quant_weights, dim=0)
            return quant_weights_tensor
        
        wqlinear.weight = transformer_weight_to_save_format(int_weight, n_bits)
        wqlinear.zeros = transformer_weight_to_save_format(zeros, n_bits)

        return wqlinear
    
    def forward(self):
        pass