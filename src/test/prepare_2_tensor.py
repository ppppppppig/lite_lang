import torch
import argparse

def max_diff(a_path, b_path):
    # 加载两个张量
    a = torch.load(a_path).cpu()
    b = torch.load(b_path).cpu()
    # 计算最大差异
    a = a.unsqueeze(1)
    print(f"a.shape: {a.shape}")
    print(f"b.shape: {b.shape}")  
    max_difference = torch.max(torch.abs(b - a)) 
    print(f"b - a: {b - a}")
    print(f"max diff: {max_difference}")

def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="计算两个张量文件的最大差异")
    
    # 添加命令行参数
    parser.add_argument("a_path", type=str, help="第一个张量文件的路径")
    parser.add_argument("b_path", type=str, help="第二个张量文件的路径")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用 max_diff 函数
    max_diff(args.a_path, args.b_path)

if __name__ == "__main__":
    main()