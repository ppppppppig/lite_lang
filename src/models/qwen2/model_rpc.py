import multiprocessing as mp
import rpyc
from rpyc.utils.server import ThreadedServer
import socket
import time
from .model import Qwen2ModelRunner, Qwen2Config
from .runner_batch import RunnerBatch, RunnerReq
import sys
import torch.distributed as dist
import concurrent.futures
import torch

# 如果tp=1, 本地运行，否则开销过大
class ModelServerRpc(rpyc.Service):
    def __init__(self, **kwargs):
        super().__init__()
        self.model_path_ = kwargs['model_path']
        self.tp_rank_ = kwargs['tp_rank']
        self.world_size_ = kwargs['world_size']
        config = kwargs['config']
        torch.cuda.set_device(self.tp_rank_)
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://localhost:29699',
            rank=self.tp_rank_,
            world_size=self.world_size_,
        )
        self.model_runner_ = Qwen2ModelRunner(config.layer_nums, kwargs['max_batch_size'], config, model_path=kwargs['model_path'], 
                                              max_length=kwargs['max_total_length'], mem_usage=kwargs['mem_usage'], 
                                              tp_rank=self.tp_rank_, world_size=self.world_size_, max_input_length=kwargs['max_input_length'])

        self.batchs_ = {}

    def init_model(self):
        torch.cuda.set_device(self.tp_rank_)
        kv_max_size = self.model_runner_.init_model()
        return kv_max_size
    
    def add_batch(self, req_messages, batch_id):
        batch = RunnerBatch.create_batch(req_messages, batch_id, is_prefill=True)
        self.batchs_[batch_id] = batch
        return True
    
    def prefill(self, batch_id):
        output_token_ids = self.model_runner_.forward(self.batchs_[batch_id])
        self.batchs_[batch_id].update_forward_message(output_token_ids)
        return output_token_ids
    
    def decode(self, batch_id):
        output_token_ids = self.model_runner_.forward(self.batchs_[batch_id])
        self.batchs_[batch_id].update_forward_message(output_token_ids)
        return output_token_ids
    
    def remove_batch(self, batch_id):
        if batch_id in self.batchs_:
            del self.batchs_[batch_id]
            return True
        return False

def run_server(tp_rank, world_size, port, ready_queue, max_batch_size, config, model_path, max_total_length, mem_usage, max_input_length):
    sys.stdin = open('/dev/tty', 'r')
    service = ModelServerRpc(port=port, max_batch_size=max_batch_size, max_input_length=max_input_length, config=config, model_path=model_path, max_total_length=max_total_length, mem_usage=mem_usage, tp_rank=tp_rank, world_size=world_size)
    server = ThreadedServer(service, port=port, protocol_config={
            "allow_public_attrs": True,
            "allow_pickle": True,
            "sync_request_timeout": 500,
            "serializer": 'json',
        })
    torch.cuda.set_device(tp_rank)
    ready_queue.put(True)  # 通知主进程服务器已启动
    server.start()

def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    return sock, sock.getsockname()[1]

class ModelClientRPC:
    def __init__(self, world_size, max_batch_size, max_input_length, config, model_path, max_total_length, mem_usage):
        
        self.connections_ = []
        self.world_size_ = world_size
        self.processor_ = []
        self.create_processor(max_batch_size, config, model_path, max_total_length, mem_usage, max_input_length)
    
    def create_processor(self, max_batch_size, config, model_path, max_total_length, mem_usage, max_input_length):
        mp.set_start_method('spawn')
        reserve_sockets = []
        free_ports = []
        for tp_rank in range(self.world_size_):
            s, port = find_free_port()
            free_ports.append(port)
            reserve_sockets.append(s)
        with mp.Manager() as manager:
            ready_queue = manager.Queue()
            for tp_rank in range(self.world_size_):
                reserve_sockets[tp_rank].close()
                p = mp.Process(target=run_server, args=(tp_rank, self.world_size_, free_ports[tp_rank], ready_queue, max_batch_size, config, model_path, max_total_length, mem_usage, max_input_length))
                p.start()
                self.processor_.append(p)
            
            # 同步工作进程是否完成初始化
            for _ in range(self.world_size_):
                ready_queue.get()

        del reserve_sockets
        
        for port in free_ports:
            conn = rpyc.connect("localhost", port=port, config={
                "sync_request_timeout": 5000,  # 客户端等待响应的超时时间（秒）
                "heartbeat": True,           # 启用心跳检测
                "heartbeat_freq": 30,        # 心跳间隔（秒）
            })
            self.connections_.append(conn)

    def close(self):
        for conn in self.connections_:
            conn.close()
        
        for p in self.processor_:
            p.terminate()
            p.join()

    def init_model(self):
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(conn.root.init_model) for conn in self.connections_]
        kv_max_size = futures[-1].result()
        return kv_max_size        
            
    def add_batch(self, runner_batch, batch_id):
        for conn in self.connections_:
            conn.root.add_batch(runner_batch.get_transfer_data(), batch_id)
    
    def prefill(self, batch_id):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(conn.root.prefill, batch_id) for conn in self.connections_]
        output_token_ids = futures[-1].result()
        return output_token_ids        
    
    def decode(self, batch_id):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(conn.root.decode, batch_id) for conn in self.connections_]
        output_token_ids = futures[-1].result()
        return output_token_ids             

    
    def remove_batch(self, batch_id):
        for conn in self.connections_:
            output_token_ids = conn.root.remove_batch(batch_id)
        return output_token_ids
    