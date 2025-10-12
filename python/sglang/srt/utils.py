import base64
import os
import random
import socket
import sys
import time
import traceback
from io import BytesIO

import numpy as np
import requests
import torch
import torch.distributed as dist

is_show_cost_time = False


def mark_cost_time(func_name):
    def inner_func(func):
        def time_func(*args, **kwargs):
            if dist.get_rank() in [0, 1] and is_show_cost_time:
                torch.cuda.synchronize()
                start_time = time.time()
                ans = func(*args, **kwargs)
                torch.cuda.synchronize()
                print(func_name, "cost time:", (time.time() - start_time) * 1000)
                return ans
            else:
                torch.cuda.synchronize()
                ans = func(*args, **kwargs)
                torch.cuda.synchronize()
                return ans

        return time_func

    return inner_func


time_mark = {}


def mark_start(key):
    torch.cuda.synchronize()
    global time_mark
    time_mark[key] = time.time()
    return


def mark_end(key, print_min_cost=0.0):
    torch.cuda.synchronize()
    global time_mark
    cost_time = (time.time() - time_mark[key]) * 1000
    if cost_time > print_min_cost:
        print(f"cost {key}:", cost_time)


def calculate_time(show=False, min_cost_ms=0.0):
    def wrapper(func):
        def inner_func(*args, **kwargs):
            torch.cuda.synchronize()
            if show:
                start_time = time.time()
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            if show:
                cost_time = (time.time() - start_time) * 1000
                if cost_time > min_cost_ms:
                    print(f"Function {func.__name__} took {cost_time} ms to run.")
            return result

        return inner_func

    return wrapper


def set_random_seed(seed: int) -> None:
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def alloc_usable_network_port(num, used_list=()):
    port_list = []
    for port in range(10000, 65536):
        if port in used_list:
            continue

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                port_list.append(port)
            except socket.error:
                pass

            if len(port_list) == num:
                return port_list
    return None


def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


def get_int_token_logit_bias(tokenizer, vocab_size):
    from transformers import LlamaTokenizer, LlamaTokenizerFast

    logit_bias = np.zeros(vocab_size, dtype=np.float32)
    for t_id in range(vocab_size):
        ss = tokenizer.decode(t_id).strip()
        if not (ss.isdigit() or len(ss) == 0 or t_id == tokenizer.eos_token_id):
            logit_bias[t_id] = -1e5
        # else:
        #    print(ss, t_id)

    return logit_bias


def wrap_kernel_launcher(kernel):
    """A faster launcher for triton kernels compatible with modern Triton versions."""
    import torch.distributed as dist

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Handle different kernel cache structures across Triton versions
    try:
        # Try to get the actual kernel instance
        if hasattr(kernel, "cache") and rank in kernel.cache:
            kernels = list(kernel.cache[rank].values())
            if kernels:
                kernel_instance = kernels[0]
            else:
                kernel_instance = kernel
        else:
            kernel_instance = kernel
    except (AttributeError, KeyError):
        kernel_instance = kernel

    # Modern Triton 3.4.0 compatible wrapper
    def modern_launcher(grid, num_warps, *args, **kwargs):
        # For Triton 3.4.0, we need to use the [grid] indexing syntax
        try:
            # Extract kwargs that are meant for kernel launch parameters
            launch_kwargs = {}
            kernel_kwargs = {}

            # Separate launch parameters from kernel parameters
            for key, value in kwargs.items():
                if key in ["num_warps", "num_stages"]:
                    launch_kwargs[key] = value
                else:
                    kernel_kwargs[key] = value

            # Add num_warps if not in kwargs
            if "num_warps" not in launch_kwargs:
                launch_kwargs["num_warps"] = num_warps

            # Call the kernel with modern syntax
            return kernel_instance[grid](*args, **launch_kwargs, **kernel_kwargs)
        except Exception as e:
            # Fallback for different calling conventions
            try:
                # Try without kwargs
                return kernel_instance[grid](*args, num_warps=num_warps)
            except Exception:
                try:
                    # Try direct call (for cached kernels)
                    if callable(kernel_instance):
                        return kernel_instance(grid, num_warps, *args)
                    else:
                        raise RuntimeError(f"Cannot launch kernel: {kernel_instance}")
                except Exception:
                    raise RuntimeError(f"Failed to launch kernel: {e}")

    return modern_launcher


def is_multimodal_model(model):
    if isinstance(model, str):
        return "llava" in model
    from sglang.srt.model_config import ModelConfig

    if isinstance(model, ModelConfig):
        return "llava" in model.path.lower()
    raise Exception("unrecognized type")


def load_image(image_file):
    from PIL import Image

    image = None

    if image_file.startswith("http://") or image_file.startswith("https://"):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
        response = requests.get(image_file, timeout=timeout)
        image = Image.open(BytesIO(response.content))
    elif image_file.lower().endswith(("png", "jpg", "jpeg", "webp", "gif")):
        image = Image.open(image_file)
    elif image_file.startswith("data:"):
        image_file = image_url.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_file)))
    else:
        image = Image.open(BytesIO(base64.b64decode(image_file)))

    return image
