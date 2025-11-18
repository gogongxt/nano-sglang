import base64
import glob
import json
import os
import random
import socket
import sys
import traceback
from collections import defaultdict
from io import BytesIO
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, safe_open, save_file
from tqdm.auto import tqdm
from transformers import PretrainedConfig


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


"""Utils for model executor."""
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"
        setattr(weight, key, value)


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    assert param.size() == loaded_weight.size()
    param.data.copy_(loaded_weight)


def prepare_hf_model_weights(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    use_safetensors: bool = False,
    fall_back_to_pt: bool = True,
    revision: Optional[str] = None,
) -> Tuple[str, List[str], bool]:
    # Download model weights from huggingface.
    is_local = os.path.isdir(model_name_or_path)
    # Some quantized models use .pt files for storing the weights.
    allow_patterns = ["*.safetensors"] if use_safetensors else ["*.bin", "*.pt"]
    if not is_local:
        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(model_name_or_path, cache_dir):
            hf_folder = snapshot_download(
                model_name_or_path,
                allow_patterns=allow_patterns,
                cache_dir=cache_dir,
                tqdm_class=Disabledtqdm,
                revision=revision,
            )
    else:
        hf_folder = model_name_or_path
    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
    if not use_safetensors:
        # Exclude files that are not needed for inference.
        # https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
        blacklist = [
            "training_args.bin",
            "optimizer.bin",
            "optimizer.pt",
            "scheduler.pt",
            "scaler.pt",
        ]
        hf_weights_files = [
            f for f in hf_weights_files if not any(f.endswith(x) for x in blacklist)
        ]

    if len(hf_weights_files) == 0 and use_safetensors and fall_back_to_pt:
        return prepare_hf_model_weights(
            model_name_or_path,
            cache_dir=cache_dir,
            use_safetensors=False,
            fall_back_to_pt=False,
            revision=revision,
        )

    if len(hf_weights_files) == 0:
        raise RuntimeError(f"Cannot find any model weights with `{model_name_or_path}`")

    return hf_folder, hf_weights_files, use_safetensors


def hf_model_weights_iterator(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    load_format: str = "auto",
    revision: Optional[str] = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    use_safetensors = False
    use_np_cache = False
    fall_back_to_pt = False
    if load_format == "auto":
        use_safetensors = True
        fall_back_to_pt = True
    elif load_format == "safetensors":
        use_safetensors = True
    elif load_format == "pt":
        pass
    elif load_format == "npcache":
        use_np_cache = True
    else:
        raise ValueError(f"Unknown load_format: {load_format}")

    hf_folder, hf_weights_files, use_safetensors = prepare_hf_model_weights(
        model_name_or_path,
        cache_dir=cache_dir,
        use_safetensors=use_safetensors,
        fall_back_to_pt=fall_back_to_pt,
        revision=revision,
    )

    if use_np_cache:
        # Currently np_cache only support *.bin checkpoints
        assert use_safetensors is False

        # Convert the model weights from torch tensors to numpy arrays for
        # faster loading.
        np_folder = os.path.join(hf_folder, "np")
        os.makedirs(np_folder, exist_ok=True)
        weight_names_file = os.path.join(np_folder, "weight_names.json")
        # Use file lock to prevent multiple processes from
        # dumping the same model weights to numpy at the same time.
        with get_lock(model_name_or_path, cache_dir):
            if not os.path.exists(weight_names_file):
                weight_names = []
                for bin_file in hf_weights_files:
                    state = torch.load(bin_file, map_location="cpu")
                    for name, param in state.items():
                        param_path = os.path.join(np_folder, name)
                        with open(param_path, "wb") as f:
                            np.save(f, param.cpu().detach().numpy())
                        weight_names.append(name)
                with open(weight_names_file, "w") as f:
                    json.dump(weight_names, f)

        with open(weight_names_file, "r") as f:
            weight_names = json.load(f)

        for name in weight_names:
            param_path = os.path.join(np_folder, name)
            with open(param_path, "rb") as f:
                param = np.load(f)
            yield name, torch.from_numpy(param)
    elif use_safetensors:
        for st_file in hf_weights_files:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    param = f.get_tensor(name)
                    yield name, param
    else:
        for bin_file in hf_weights_files:
            state = torch.load(bin_file, map_location="cpu")
            for name, param in state.items():
                yield name, param
            del state
            torch.cuda.empty_cache()


import contextlib


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)
