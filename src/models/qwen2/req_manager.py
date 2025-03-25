import torch
from dataclasses import dataclass
import queue
from .cache import Cache, NormalCache

@dataclass
class Req:
    id: int
    prompt: str
    max_total_length: int
    max_output_length: int
    output_prompt_que: queue.Queue
    temperature: float
    top_p: float
    top_k: float
    do_sample: bool
    input_length: int = 0
    output_length: int = 0
    is_end: bool = False
    
    def Add(self, text: str, is_eos_token):
        self.output_length += 1
        if is_eos_token or (self.output_length + self.input_length >= self.max_total_length and self.output_length >= self.max_output_length):
            self.is_end = True
        self.output_prompt_que.put((text, self.is_end))


@dataclass
class ModelInput:
    reqs: list[Req]                        #与input_tokens有一一对应关系
    input_tokens: torch.Tensor
    is_prefill: bool
    is_batch_end: bool
    padding_mask: torch.Tensor
    output_tokens: torch.Tensor = None
    past_key_values: Cache = None
    
    def Update(self, output_tokens: torch.Tensor):
        self.is_prefill = False
        if self.output_tokens is None:
            self.output_tokens = output_tokens
        else:
            self.output_tokens = torch.cat([self.output_tokens, output_tokens], dim=-1)
        
        
        new_padding = torch.ones((self.padding_mask.size(0), 1), dtype=torch.int).cuda()
        for z in range(self.padding_mask.size(0)):
            pos_num = self.padding_mask[z, -1].item()
            new_padding[z, 0] = pos_num + 1
        self.padding_mask = torch.cat([self.padding_mask, new_padding], dim=-1).cuda()
    
    def GetPostSamplePara(self):
        temperatures = torch.tensor([req.temperature for req in self.reqs]).cuda()
        top_p = torch.tensor([req.top_p for req in self.reqs]).cuda()
        top_k = torch.tensor([req.top_k for req in self.reqs]).cuda()
        do_sample = torch.tensor([req.do_sample for req in self.reqs]).cuda()
        return temperatures, top_p, top_k, do_sample
    
    def ViewIsEnd(self):
        for req in self.reqs:
            if not req.is_end:
                self.is_batch_end = False
                return
            
        self.is_batch_end = True
        
    def UpdateReqsMessage(self):
        length = self.padding_mask.size(-1)
        padding_len = (self.padding_mask == 1).float().argmax(dim=-1)
        input_len = length - padding_len
        for idx in range(input_len.size(0)):
            self.reqs[idx].input_length = input_len[idx].item()


class ReqManager:
    def __init__(self, max_batch_size, tokenizer, max_total_length, max_output_length):
        self.reqs = queue.Queue()
        self.padding_mask = None
        self.max_batch_size_ = max_batch_size
        self.tokenizer_ = tokenizer
        self.now_id = 0
        self.max_id = 100000
        self.max_total_length_ = max_total_length
        self.max_output_length_ = max_output_length
        
    def Add(self, prompt, top_p, top_k, temperature, do_sample):
        req = Req(id=self.now_id % self.max_id, prompt=prompt, max_total_length=self.max_total_length_,
            max_output_length=self.max_output_length_, output_prompt_que=queue.Queue(), top_p=top_p, top_k=top_k, temperature=temperature, do_sample=do_sample)
        self.now_id += 1
        self.reqs.put(req)
        return req

    def GetReqBatch(self) -> ModelInput:
        input_reqs = []
        while not self.reqs.empty() and len(input_reqs) < self.max_batch_size_:
            input_reqs.append(self.reqs.get())
        if len(input_reqs) == 0:
            return None
        prompts = [req.prompt for req in input_reqs]
        input_tokens, mask = self.tokenizer_.encode(prompts)
        
        
        for z in range(mask.size(0)):
            first_one = 0
            for h in range(mask.size(1)):
                if mask[z, h] == 1:
                    first_one = h
                    break
            mask[z, first_one:] = torch.arange(0, mask.size(1) - first_one).cuda()
            mask[z, :first_one] = -1
        
        model_inputs = ModelInput(
            reqs = input_reqs,
            input_tokens = input_tokens,
            is_prefill = True,
            padding_mask = mask.cuda(),
            is_batch_end = False,
        )
        
        model_inputs.UpdateReqsMessage()
        return model_inputs
        
    def UpdateReq(self, model_outputs):
        output_tokens = model_outputs.output_tokens[:, -1]
        print(f"output_tokens:{output_tokens}")
        output_prompts = self.tokenizer_.decode(output_tokens)
        for req_idx in range(len(output_prompts)):
            model_outputs.reqs[req_idx].Add(output_prompts[req_idx], output_tokens[req_idx] == self.tokenizer_.eos_token_id)
            
        model_outputs.ViewIsEnd()
        return model_outputs
    