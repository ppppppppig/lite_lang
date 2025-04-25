import requests
import json

# 定义请求参数
url = "http://localhost:8080/generate"
stream_url = "http://localhost:8080/generate_stream"
payload = {
    "prompt": "请写一首关于春天的诗",
    "top_p": 0.8,
    "top_k": 20,
    "temperature": 10,
    "do_sample": False,
}


# 普通请求函数（保持原有）
def normal_request():
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        print("普通接口结果:", result["result"])
    else:
        print("请求失败，状态码:", response.status_code)


# 新增流式请求函数
def stream_request():
    try:
        # 开启流式模式
        with requests.post(stream_url, json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"请求失败，状态码: {response.status_code}")
                return

            # 逐块读取内容
            buffer = ""
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    # 解码字节为字符串（参考内容中的编码处理）
                    decoded_chunk = chunk.decode("utf-8")
                    buffer += decoded_chunk
                    ddict = json.loads(decoded_chunk)
                    print(ddict["generated_text"], end="")  # 不换行

    except requests.exceptions.RequestException as e:
        print(f"请求异常: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # print("--- 普通请求 ---")
    # normal_request()

    print("\n--- 流式请求 ---")
    stream_request()
