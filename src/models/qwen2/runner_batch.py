import torch
from dataclasses import dataclass
from typing import Dict

def get_no_padding_messages(input_reqs, is_prefill=True):
    position_ids = []
    input_ids = []
    req_lengths = []
    start_idx = []
    for idx, req in enumerate(input_reqs):
        if is_prefill:
            position_ids.extend([i for i in range(len(req.input_tokens))])
            start_idx.append(len(input_ids))
            input_ids.extend(req.input_tokens)
        else:
            position_ids.append(req.length - 1)
            start_idx.append(idx)
        req_lengths.append(req.length)
    return torch.tensor(input_ids).cuda(), torch.tensor(position_ids).cuda(), \
           torch.tensor(start_idx).cuda(), torch.tensor(req_lengths).cuda()

@dataclass
class RunnerReq:
    id: int
    rid: str
    input_length: int
    output_length: int = 0
    input_tokens: list[int] = None
    output_tokens: list[int] = None
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: float = 0.0
    do_sample: bool = False
    
    @staticmethod
    def create_runner_req(id:int, input_tokens: list[int], temperature: float, top_p: float, top_k: float, do_sample: bool) -> 'RunnerReq':
        return RunnerReq(
            id=id,
            rid=None,
            input_length=len(input_tokens),
            input_tokens=input_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample
        )
        
    @property
    def length(self):
        return self.input_length + self.output_length
    
    def update_forward_message(self, token_id):
        self.output_length += 1
        if self.output_tokens is None:
            self.output_tokens = []
        self.output_tokens.append(token_id)


@dataclass
class RunnerBatch:
    id: int
    request_mapping: Dict[int, RunnerReq]
    input_tokens: torch.Tensor  # NoPadding形式的输入，prefill的输入
    position_ids: torch.Tensor  # 输入的token在原始文本中的位置
    b_start_idx: torch.Tensor   # 每个请求的起始位置
    b_seq_len: torch.Tensor     # 每个请求的长度
    output_token_ids: torch.Tensor # NoPadding形式的输入，decode的输入
    is_prefill: bool = True
    
    @staticmethod
    def create_batch(req_messages, batch_id, is_prefill: bool = True) -> 'RunnerBatch':
        request_mapping = {}
        for message in req_messages:
            request_id = message['id']
            input_tokens = message['input_tokens']
            new_req = RunnerReq.create_runner_req(
                id=request_id,
                input_tokens=input_tokens,
                temperature=message['temperature'],
                top_p=message['top_p'],
                top_k=message['top_k'],
                do_sample=message['do_sample']
            )
            request_mapping[request_id] = new_req
        input_tokens, position_ids, b_start_idx, b_seq_len = get_no_padding_messages(request_mapping.values(), is_prefill)
        return RunnerBatch(
            id=batch_id,
            request_mapping=request_mapping,
            is_prefill=is_prefill,
            input_tokens=input_tokens,
            position_ids=position_ids,
            b_start_idx=b_start_idx,
            b_seq_len=b_seq_len,
            output_token_ids=torch.tensor([])  # 假设 output_tokens 需要一个默认值
        )
        
    def get_post_sample_para(self):
        temperatures = torch.tensor([req.temperature for req in self.request_mapping.values()]).cuda()
        top_p = torch.tensor([req.top_p for req in self.request_mapping.values()]).cuda()
        top_k = torch.tensor([req.top_k for req in self.request_mapping.values()]).cuda()
        do_sample = torch.tensor([req.do_sample for req in self.request_mapping.values()]).cuda()
        return temperatures, top_p, top_k, do_sample
    
    def filter(self, req_ids):
        new_output_token_ids = []
        should_free_reqs = []
        for req_id in req_ids:
            should_free_reqs.append(self.request_mapping.pop(req_id))
        for req in self.request_mapping.values():
            new_output_token_ids.append(req.output_tokens[-1])
        is_prefill = False
        # import pdb; pdb.set_trace()
        self.input_tokens, self.position_ids, self.b_start_idx, self.b_seq_len = get_no_padding_messages(self.request_mapping.values(), is_prefill)
        self.output_token_ids = torch.tensor(new_output_token_ids).cuda()
        return should_free_reqs
            
    def update_forward_message(self, output_token_ids):
        self.is_prefill = False
        self.output_token_ids = output_token_ids.squeeze(1)
        req_ids = []
        for idx, req in enumerate(self.request_mapping.values()):
            output_token = output_token_ids[idx].item()
            req_ids.append(req.id)
            req.update_forward_message(output_token_ids[idx].item())
        self.input_tokens, self.position_ids, self.b_start_idx, self.b_seq_len = get_no_padding_messages(self.request_mapping.values(), self.is_prefill)
        return req_ids
        
    def update(self, new_runner_batch):
        assert self.is_prefill == False and new_runner_batch.is_prefill == False, "only decode batch can be merged"
        new_output_token_ids = []
        self.request_mapping.update(new_runner_batch.request_mapping)
        self.input_tokens = torch.cat((self.input_tokens, new_runner_batch.input_tokens), dim=0) # 这个后续应该没啥用，没事，先留着
        for req in self.request_mapping.values():
            new_output_token_ids.append(req.output_tokens[-1])
        self.input_tokens, self.position_ids, self.b_start_idx, self.b_seq_len = get_no_padding_messages(self.request_mapping.values(), is_prefill=False)
        self.output_token_ids = torch.tensor(new_output_token_ids).cuda()
        return True