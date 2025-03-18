import torch
from dataclasses import dataclass
import queue

@dataclass
class Req:
    prompt: str
    output_prompt: str
    
@dataclass
class ModelInput:
    input_tokens: torch.Tensor
    padding_mask: torch.Tensor


class ReqManager:
    def __init__(self, max_batch_size):
        self.reqs = queue.Queue()
        self.padding_mask = None
        self.max_batch_size_ = max_batch_size
        
    def Add(self, prompt):
        req = Req(prompt=prompt)
        self.reqs.push(req)

    def GetReqBatch(self) -> ModelInput:
        input_reqs = []
        while not self.reqs.empty() and len(input_reqs) < self.max_batch_size_:
            input_reqs.append(self.reqs.pop())
        
        return input_reqs
        