import json
import torch
from collections import OrderedDict

def generate_shard_index(input_path, output_dir, max_shard_size=2e9):
    # 加载原始权重
    state_dict = torch.load(input_path, map_location="cpu")
    
    # 初始化索引结构
    index = {
        "metadata": {"total_size": 0},
        "weight_map": OrderedDict()
    }
    
    # 计算总大小
    total_size = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
    index["metadata"]["total_size"] = total_size
    
    # 分片参数
    current_shard = 0
    current_size = 0
    shard_files = []
    import pdb; pdb.set_trace()
    for name, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        
        # 需要新建分片的情况
        if current_size + tensor_size > max_shard_size:
            current_shard += 1
            current_size = 0
            
        # 记录到索引
        shard_name = f"pytorch_model-{current_shard:05d}-of-{len(shard_files):05d}.bin"
        index["weight_map"][name] = shard_name
        
        # 保存分片
        if shard_name not in shard_files:
            torch.save({name: tensor}, f"{output_dir}/{shard_name}")
            shard_files.append(shard_name)
            current_size += tensor_size
        else:
            existing_shard = torch.load(f"{output_dir}/{shard_name}")
            existing_shard.update({name: tensor})
            torch.save(existing_shard, f"{output_dir}/{shard_name}")
            current_size += tensor_size
    
    # 保存索引文件
    with open(f"{output_dir}/pytorch_model.bin.index.json", "w") as f:
        json.dump(index, f, indent=2)

# 使用示例
generate_shard_index(
    input_path="../models/quantization_qwen_test/pytorch_model.bin",
    output_dir="../models/quantization_qwen_test/sharded_output",
    max_shard_size=2e9  # 2GB
)