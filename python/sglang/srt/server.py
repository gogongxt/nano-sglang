"""SRT: SGLang Runtime"""

import argparse
import asyncio
import dataclasses
import json
import multiprocessing as mp
import sys
import threading
import time
from typing import List, Optional

# Fix a Python bug
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

import psutil
import requests
import uvicorn
import uvloop
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from sglang.srt.managers.detokenizer_manager import start_detokenizer_process
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.openai_protocol import CompletionRequest
from sglang.srt.managers.router.manager import start_router_process
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import alloc_usable_network_port

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


app = FastAPI()
tokenizer_manager = None


@app.get("/get_model_info")
async def get_model_info():
    result = {
        "model_path": tokenizer_manager.model_path,
    }
    return result


@app.post("/generate")
async def generate_request(obj: GenerateReqInput):
    obj.post_init()
    result_generator = tokenizer_manager.generate_request(obj)

    if obj.stream:

        async def stream_results():
            async for out in result_generator:
                yield (json.dumps(out) + "\0").encode("utf-8")

        return StreamingResponse(stream_results(), media_type="text/event-stream")
    else:
        ret = await result_generator.__anext__()
        return ret


@app.post("/v1/completions")
async def v1_completions(obj: CompletionRequest):
    assert obj.n == 1
    obj = GenerateReqInput(
        text=obj.prompt,
        sampling_params={
            "temperature": obj.temperature,
            "max_new_tokens": obj.max_tokens,
            "stop": obj.stop,
        },
    )
    ret = await generate_request(obj)
    return {
        "choices": [{"text": ret["text"]}],
    }


def launch_server(server_args, pipe_finish_writer):
    global tokenizer_manager

    # Allocate ports
    can_use_ports = alloc_usable_network_port(
        num=4 + server_args.tp_size, used_list=(server_args.port,)
    )
    port_args = PortArgs(
        tokenizer_port=can_use_ports[0],
        router_port=can_use_ports[1],
        detokenizer_port=can_use_ports[2],
        nccl_port=can_use_ports[3],
        model_rpc_ports=can_use_ports[4:],
    )

    # Launch processes
    tokenizer_manager = TokenizerManager(server_args, port_args)
    pipe_router_reader, pipe_router_writer = mp.Pipe(duplex=False)
    pipe_detoken_reader, pipe_detoken_writer = mp.Pipe(duplex=False)

    proc_router = mp.Process(
        target=start_router_process,
        args=(
            server_args,
            port_args,
            pipe_router_writer,
        ),
    )
    proc_router.start()
    proc_detoken = mp.Process(
        target=start_detokenizer_process,
        args=(
            server_args,
            port_args,
            pipe_detoken_writer,
        ),
    )
    proc_detoken.start()

    # Wait for the model to finish loading
    router_init_state = pipe_router_reader.recv()
    detoken_init_state = pipe_detoken_reader.recv()

    if router_init_state != "init ok" or detoken_init_state != "init ok":
        proc_router.kill()
        proc_detoken.kill()
        print("router init state:", router_init_state)
        print("detoken init state:", detoken_init_state)
        sys.exit(1)

    assert proc_router.is_alive() and proc_detoken.is_alive()

    def launch_server():
        # Launch api server
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )

    t = threading.Thread(target=launch_server)
    t.start()

    if pipe_finish_writer:
        url = server_args.url()

        success = False
        for i in range(60):
            try:
                res = requests.get(url + "/get_model_info", timeout=5)
                success = True
                break
            except requests.exceptions.RequestException as e:
                time.sleep(1)

        if success:
            pipe_finish_writer.send("init ok")
        else:
            pipe_finish_writer.send(str(e))
