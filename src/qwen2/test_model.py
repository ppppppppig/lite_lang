from .model import Qwen2Model, Qwen2Config
import os

def test(model_path):
    config_path = os.path.join(model_path, 'config.json')
    config = Qwen2Config(config_path)
    max_req_length = 1024
    model = Qwen2Model(config.layer_nums, max_req_length, config.hidden_size)
    
    mock_input_token = [23, 24, 25]
    output_token = model.forward(mock_input_token)
    print(f"output_token: {output_token}")