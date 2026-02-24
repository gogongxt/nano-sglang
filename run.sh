#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1 # Enable device-side assertions for debugging

PORT=58887

counter=0
echo "Monitoring port: $PORT"
while true; do
    if ss -antp 2>/dev/null | grep -q ":$PORT "; then
        ((counter += 1))
        if ((counter >= 5)); then
            echo "$(date '+%F %T') Port $PORT is still in use..."
            counter=0
        fi
        sleep 1
    else
        echo "$(date '+%F %T') Port $PORT is free."
        break
    fi
done

set -ex

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

# python3 -m sglang.launch_server \
# --model-path /tmp-data/models/llama-2-7b \
# --port 30000 \
# --mem-fraction-static 0.8 --tp 2 \
# --model-mode flashinfer

timeout=$((3600 * 24 * 7))
export NCCL_TIMEOUT=${timeout}

DEBUGPY_CMD=(
    -m debugpy
    --listen 0.0.0.0:5678
    # --wait-for-client
)

CMD=(
    python3
    "${DEBUGPY_CMD[@]}"
    -m sglang.launch_server
    --model-path /data/models/llama-2-7b
    --host 0.0.0.0
    --port "$PORT"
    --mem-fraction-static 0.8
)

"${CMD[@]}"
