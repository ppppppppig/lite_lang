import litelang.apis.api_server as api_server
from litelang.apis.http_server_manager import HttpServerManager
import uvicorn
import click


@click.command()
@click.option(
    "--model_path", default="/root/LiteLang/models/Qwen2-1.5B/", help="权重路径"
)
@click.option("--max_output_length", default=200, type=int, help="最大输出长度")
@click.option("--max_input_length", default=4000, type=int, help="最大输入长度")
@click.option("--max_batch_size", default=32, type=int, help="最大batchsize")
@click.option("--tp", default=1, type=int, help="tp并行数")
@click.option("--port", default=8080, type=int, help="监听端口")
@click.option("--mem_usage", default=0.5, type=float, help="显存使用率")
@click.option("--max_reqs", default=1000, type=float, help="就绪队列最大请求数")
@click.option(
    "--busy_scale",
    default=0.6,
    type=float,
    help="系统不繁忙时减小请求的最大生成长度，尝试调度更多请求去推理",
)
@click.option("--use_radix_cache", default=False, type=bool, help="是否使用radix缓存")
def run_server(
    model_path,
    max_output_length,
    max_batch_size,
    max_input_length,
    tp,
    port,
    mem_usage,
    max_reqs,
    busy_scale,
    use_radix_cache,
):
    http_server_manager = HttpServerManager(
        model_path,
        max_input_length,
        max_output_length,
        max_batch_size,
        mem_usage,
        tp,
        max_reqs,
        busy_scale,
        use_radix_cache,
    )
    api_server.g_objs.http_server_manager = http_server_manager
    uvicorn.run("litelang.apis.test_server:api_server.app", host="0.0.0.0", port=port)


def main():
    run_server()


if __name__ == "__main__":
    main()
