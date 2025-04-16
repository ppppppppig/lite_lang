### 1.背景

为了能够更加熟悉大语言模型推理框架的技术细节，从零开始编写大语言模型推理框架。

### 2.支持模型类型

| 模型类型    | 参数量 | 状态   | 备注                       |
| ----------- | ------ | ------ | --------------------     |
| qwen2  | 1.5B   | 已支持 |                           |
| qwq  | 32B   | 未支持 |         预计v0.5版本支持                  |


### 3.版本发布列表

| 版本    | 功能 | 状态   |
| ----------- | ------ | ------ |
| v0.1  | 1.支持流式接口<br>2.支持qwen2-1.5B   | 已发布 |  
| v0.2  | 开发rmsnorm/ffn/flash_attn/rope等算子，大幅提高推理性能  | 已发布 |
| v0.3  | 支持NoPadding输入与page attention，以及flash decoding | 已发布 |
| v0.4  | 支持多TP | 已发布 |
| v0.5  | 支持连续批处理 | 正在开发中 |
| v0.6  | 支持QwQ32B| 未开发|


### 4.运行服务

使用以下命令运行服务
```
python test/test_server.py --model-path [MODEL_PATH] --max-output-length 1024 \
--max-input-length --max-batch-size 32 --device cuda device_id 0 --port 8080
```

使用一下命令测试服务是否正常：
```
bash test/start_multi_req.sh
```

### 5.已知BUG

| buglist    | 场景 | 状态   |
| ----------- | ------ | ------ |
| 1  | 使用flast attn算子时，偶发输出不对  | 已解决（flash attn算子存在越界写错误） |
| 2  | 当组batch推理长度不同的prompt时，最长的prompt必须在batch最前，否则会崩溃  | 更新算子后已解决 |  
| 3  | attn计算时q，k,v必须和  | 应该是flash_attn算子导致的问题，暂时不定位，后续开发kv_cache和no_padding需要更改该算子 |  


### 6.需要填的小坑

1.多进程使用rpc通信会造成较大开销，推理速度大概下降10%