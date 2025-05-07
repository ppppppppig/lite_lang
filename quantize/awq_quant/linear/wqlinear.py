import torch

class WQLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, n_bits, has_bias=True):
        super(WQLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        assert n_bits == 4, "n_bits shoule equal 4"

        self.register_buffer('qweight', torch.empty(out_features // 32 * n_bits, in_features, dtype=torch.int32))
        
        self.register_buffer('qzeros', torch.empty(out_features // 32 * n_bits, dtype=torch.int32))
        
        self.register_buffer('scales', torch.empty(out_features, dtype=torch.float16))
        
        if has_bias:
            self.register_buffer('bias', torch.empty(out_features // 32 * n_bits).half())
            
    @classmethod
    def create_wqlinear(cls, linear, scales, zeros, in_features, out_features, n_bits, group_size):
        # scales: [out_features]
        # zeros: [out_features]
        assert n_bits == 4, "n_bits shoulde equal 4"
        assert zeros is not None and scales is not None
        
        scale_zeros = zeros * scales
        has_bias = False
        if hasattr(linear, "bias") and linear.bias is not None:
            has_bias = True
        wqlinear = cls(in_features, out_features, n_bits, has_bias)
        
        # 标准非对称量化流程，zeros是已经量化过了
        def quant_weight(linear, scales, zeros, group_size):
            intweight = []
            # import pdb; pdb.set_trace()
            org_weight_shape = linear.weight.shape
            # weight = weight.view(-1, group_size)
            for idx in range(linear.weight.size(1)):
                
                # 为什么这里没使用per_channel量化
                intweight.append(
                    torch.round(
                        (linear.weight.data[:, idx] + scale_zeros[idx // group_size])
                        / scales[idx // group_size]
                    ).to(torch.int)[:, None])
            intweight = torch.cat(intweight, dim=1)
            intweight = intweight.t().contiguous()
            intweight = intweight.to(dtype=torch.int32)
            # intweight = intweight.view(**org_weight_shape)
            return intweight
        int_weight = quant_weight(linear, scales, zeros, group_size)
        zeros = zeros.to(torch.int32)
        wqlinear.scales = scales.half()
        
        if hasattr(linear, "bias") and linear.bias is not None:
            wqlinear.bias = linear.bias.half()
        
        def transformer_weight_to_save_format(weight, n_bit):
            assert n_bit == 4, "n_bits shoulde equal 4"
            order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            pack_num = 32 // n_bit
            quant_weights = []
            # import pdb; pdb.set_trace()
            for i in range(0, weight.size(1), pack_num):
                quant_weight = torch.zeros(weight.size(0), dtype=torch.int32)
                for j in range(pack_num):
                    channel = weight[:, i + order_map[j]]
                    channel = channel << (j * n_bits)
                    quant_weight = quant_weight | channel
                quant_weights.append(quant_weight)
            # import pdb; pdb.set_trace()
            quant_weights_tensor = torch.stack(quant_weights, dim=1)
            return quant_weights_tensor
        wqlinear.qweight = transformer_weight_to_save_format(int_weight, n_bits)
        # import pdb; pdb.set_trace()
        wqlinear.qzeros = transformer_weight_to_save_format(zeros, n_bits)

        return wqlinear
    
    def forward(self):
        pass