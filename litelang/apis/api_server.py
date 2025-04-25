from fastapi import FastAPI
from pydantic import BaseModel
from typing import Generator
import uvicorn
from dataclasses import dataclass
from .http_server_manager import HttpServerManager
from fastapi.responses import Response, StreamingResponse
import json

@dataclass
class GObjs:
    http_server_manager: HttpServerManager = None
    app: FastAPI = None

    default_temperature = 1.0
    default_top_p = 1.0
    default_top_k = 1
    default_do_sample = False


app = FastAPI()
g_objs = GObjs(app=app)


class GenerateRequest(BaseModel):
    top_p: float = g_objs.default_top_p
    temperature: float = g_objs.default_temperature
    top_k: int = g_objs.default_top_k
    do_sample: bool = g_objs.default_do_sample
    prompt: str


@app.post("/generate")
def generate(request: GenerateRequest):
    temp = request.get()
    generator = g_objs.http_server_manager.generate(
        request.prompt,
        request.top_p,
        request.top_k,
        request.temperature,
        request.do_sample,
    )
    output_prompts = []

    for text in generator:
        output_prompts.append(text)
    return Response(
        content=json.dumps({"result": output_prompts}, ensure_ascii=False).encode(
            "utf-8"
        )
    )


@app.post("/generate_stream")
def generate_stream(request: GenerateRequest) -> Generator[dict, None, None]:
    generator = g_objs.http_server_manager.generate(
        request.prompt,
        request.top_p,
        request.top_k,
        request.temperature,
        request.do_sample,
    )

    def stream_results() -> Generator[bytes, None, None]:
        for text in generator:
            ret = {
                "generated_text": text,
            }
            yield (json.dumps(ret, ensure_ascii=False) + "\n\n").encode("utf-8")

    return StreamingResponse(stream_results(), media_type="text/event-stream")
