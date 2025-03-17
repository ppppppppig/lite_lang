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
@click.option("--max-output-length", default=1024, type=int, help="最大输出长度")
@click.option("--max-input-length", default=1024, type=int, help="最大输入长度")
@click.option("--max-batch-size", default=32, type=int, help="最大batchsize")
@click.option("--device", default='cuda', type=str, help="设备类型")
@click.option("--device_id", default=3, type=int, help="设备ID")
@click.option("--port", default=8080, type=int, help="监听端口")
def run_server(model_path, max_output_length, max_batch_size, max_input_length, device, device_id, port):
    device = f'{device}:{device_id}'
    http_server_manager = HttpServerManager(model_path, max_input_length, max_output_length, max_batch_size, device)
    api_server.g_objs.http_server_manager = http_server_manager
    uvicorn.run("test_server:api_server.app", host="0.0.0.0", port=port)


if __name__ == "__main__":
   run_server()