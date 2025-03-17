import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from models.qwen2 import api_server
from models.qwen2.http_server_manager import HttpServerManager
import uvicorn

if __name__ == "__main__":
    model_path = "/root/LiteLang/models/Qwen2-1.5B/"
    max_req_length = 256
    max_batch_size = 32
    device = 'cuda:3'
    http_server_manager = HttpServerManager(model_path, max_req_length, max_batch_size, device)
    api_server.g_objs.http_server_manager = http_server_manager
    # import pdb
    # pdb.set_trace()
    uvicorn.run("test_server:api_server.app", host="0.0.0.0", port=8000)