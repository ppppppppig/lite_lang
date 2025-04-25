import torch
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from litelang.models.cache import PageCache

if __name__ == "__main__":
    max_batch_size = 32
    max_length = 1024
    memory_usage = 0.7

    num_layers = 1
    head_dim = 128
    num_key_value_heads = 2
    torch_dtype = torch.float16

    page_attention = PageCache(
        max_batch_size,
        max_length,
        memory_usage,
        num_layers,
        head_dim,
        num_key_value_heads,
        torch_dtype,
    )
    rid = page_attention.alloc_req()
    page_attention.free_req(rid)

    torch.cuda.set_device(0)
    from dataclasses import dataclass

    @dataclass
    class MockReq:
        rid: int = None
        length: int = None
        start: int = None

    batch_size = 2
    seq_len1, seq_len2 = 2, 5
    head_dim = 128
    reqs = []

    def mock_prefill_test():
        layer_idx = 0
        prefill_key_states = torch.randn(
            (seq_len1 + seq_len2, num_key_value_heads, head_dim), dtype=torch_dtype
        ).cuda()
        prefill_value_states = torch.randn(
            (seq_len1 + seq_len2, num_key_value_heads, head_dim), dtype=torch_dtype
        ).cuda()

        mock_req1 = MockReq()
        mock_req1.rid = page_attention.alloc_req()
        mock_req1.length = seq_len1
        mock_req1.start = 0
        reqs.append(mock_req1)

        mock_req2 = MockReq()
        mock_req2.rid = page_attention.alloc_req()
        mock_req2.length = seq_len2
        mock_req2.start = seq_len1
        reqs.append(mock_req2)

        page_attention.write_prefill_kv_cache(
            reqs, layer_idx, prefill_key_states, prefill_value_states
        )

    def mock_decode_test():
        layer_idx = 0
        decode_key_states = torch.randn(
            (2, num_key_value_heads, head_dim), dtype=torch_dtype
        ).cuda()
        decode_value_states = torch.randn(
            (2, num_key_value_heads, head_dim), dtype=torch_dtype
        ).cuda()

        reqs[0].start = 0
        reqs[1].start = 1

        page_attention.write_decode_kv_cache(
            reqs, layer_idx, decode_key_states, decode_value_states
        )

    def get_token_idx_after_prefill():
        token_idx = page_attention.get_token_index(reqs)
        should_be_token_idx = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6], dtype=torch.int
        ).cuda()
        assert torch.allclose(
            token_idx, should_be_token_idx
        ), f"token_idx: {token_idx}, should_be_token_idx: {should_be_token_idx}"

    def get_token_idx_after_decode():
        token_idx = page_attention.get_token_index(reqs)
        should_be_token_idx = torch.tensor(
            [0, 1, 7, 2, 3, 4, 5, 6, 8], dtype=torch.int
        ).cuda()
        assert torch.allclose(
            token_idx, should_be_token_idx
        ), f"token_idx: {token_idx}, should_be_token_idx: {should_be_token_idx}"

    def free_reqs_test():
        page_attention.dealloc_reqs(reqs)

    mock_prefill_test()
    get_token_idx_after_prefill()
