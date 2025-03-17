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
    
app = FastAPI()
g_objs = GObjs(app=app)

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate(request: GenerateRequest):
    generator = g_objs.http_server_manager.generate(request.prompt)
    output_prompts = []
    
    for text in generator:
        output_prompts.append(text)
    return Response(content=json.dumps({'result': output_prompts}, ensure_ascii=False).encode("utf-8"))

@app.post("/generate_stream")
def generate_stream(request: GenerateRequest) -> Generator[dict, None, None]: 
    generator = g_objs.http_server_manager.generate(request.prompt)
    
    def stream_results() -> Generator[bytes, None, None]: 
        for text in generator:
            ret = {
                "generated_text": text,
            }
            yield ( json.dumps(ret, ensure_ascii=False) + "\n\n").encode("utf-8")
    
    return StreamingResponse(stream_results(), media_type="text/event-stream")
