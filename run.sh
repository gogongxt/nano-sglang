#!/bin/bash
set -ex

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1  # Enable device-side assertions for debugging

# python3 -m sglang.launch_server \
# --model-path /tmp-data/models/llama-2-7b \
# --port 30000 \
# --mem-fraction-static 0.8 --tp 2

# python3 -m sglang.launch_server \
# --model-path /tmp-data/models/Qwen3-8B \
# --port 30000 \
# --mem-fraction-static 0.8 --tp 1

# --tp 1 \
# --trust-remote-code --host 0.0.0.0

python3 -m sglang.launch_server \
--model-path /tmp-data/models/Llama-2-7B-AWQ \
--port 30000 \
--mem-fraction-static 0.8 --tp 2
