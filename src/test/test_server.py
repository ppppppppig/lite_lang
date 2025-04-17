import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from models.qwen2 import api_server
from models.qwen2.http_server_manager import HttpServerManager
import uvicorn
import click


@click.command()
@click.option("--model-path", default="/root/LiteLang/models/Qwen2-1.5B/", help="权重路径")
@click.option("--max-output-length", default=20, type=int, help="最大输出长度")
@click.option("--max-input-length", default=20, type=int, help="最大输入长度")
@click.option("--max-batch-size", default=32, type=int, help="最大batchsize")
@click.option("--tp", default=1, type=int, help="tp并行数")
@click.option("--port", default=8080, type=int, help="监听端口")
@click.option('--mem_usage', default=0.5, type=float, help="显存使用率")
@click.option('--max_reqs', default=1000, type=float, help="就绪队列最大请求数")
@click.option('--busy_scale', default=0.6, type=float, help="系统不繁忙时减小请求的最大生成长度，尝试调度更多请求去推理")
def run_server(model_path, max_output_length, max_batch_size, max_input_length, tp, port, mem_usage, max_reqs, busy_scale):
    http_server_manager = HttpServerManager(model_path, max_input_length, max_output_length, max_batch_size, mem_usage, tp, max_reqs, busy_scale)
    api_server.g_objs.http_server_manager = http_server_manager
    uvicorn.run("test_server:api_server.app", host="0.0.0.0", port=port)


if __name__ == "__main__":
   run_server()