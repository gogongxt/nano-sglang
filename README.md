# nano-sglang

A lightweight LLM inference framework inspired by SGLang, designed for learning and educational purposes. This project focuses on simplicity and clarity over raw performance, implementing core inference concepts while maintaining a clean, minimal codebase.

## 🎯 Project Goals

nano-sglang is created to help developers understand the internal workings of modern LLM inference frameworks. We've stripped away complexity while preserving the essential components that make SGLang powerful:

- **Educational Focus**: Clean, readable code that demonstrates core inference concepts
- **Core Implementation**: RadixTree, scheduling, and other fundamental mechanisms
- **Minimal Dependencies**: Reduced complexity without sacrificing functionality
- **Modern Stack**: Built with torch and triton operators, updated for latest libraries

## ✨ Features

### Core Capabilities

- **RadixTree**: Efficient attention key-value caching and management
- **Advanced Scheduling**: Multiple scheduling heuristics (LPM, weight, random, FCFS)
- **Tensor Parallelism**: Multi-GPU inference support
- **AWQ Quantization**: Memory-efficient model quantization

### Model Support

- ✅ **Llama2** models (base and chat variants)
- ✅ **Llama2 AWQ** quantized models
- 🚧 **More models coming soon** (see roadmap below)

### Technical Implementation

- **Pure Torch/Triton**: All operators implemented using PyTorch and Triton
- **VLLM-Free**: Completely removed VLLM dependencies for cleaner codebase
- **Modern Dependencies**: Updated to work with latest library versions
- **Bug Fixes**: Resolved issues from early SGLang versions

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/nano-sglang.git
cd nano-sglang

# Install with all dependencies
pip install -e ".[all]"

# Optional: Install flashinfer for acceleration
git submodule update --init --recursive
pip install 3rdparty/flashinfer/python
```

### Basic Usage

```bash
# Basic server launch
python3 -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000

# With tensor parallelism
python3 -m sglang.launch_server --model-path /path/to/llama2-model --port 30000 --tp 2

# With AWQ quantization
python3 -m sglang.launch_server --model-path /path/to/Llama-2-7B-AWQ --port 30000 --mem-fraction-static 0.8

# With flashinfer acceleration
python3 -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 --model-mode flashinfer
```

### API Usage

The server provides OpenAI-compatible endpoints:

#### curl Examples

```bash
# Completions
curl -X POST "http://localhost:30000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "prompt": "What is the capital of France?",
    "max_tokens": 40,
    "temperature": 0
  }'
# {"choices":[{"text":"\nFrance is a country in Western Europe. It is the largest country in the European Union. The capital of France is Paris.\nWhat is the capital of France?\nWhat is the capital of"}]}
```

#### Python Examples

```python
import requests

# Completions
response = requests.post("http://localhost:30000/v1/completions", json={
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "prompt": "What is the capital of France?",
    "max_tokens": 40,
    "temperature": 0
})

print(response.json())
# {"choices":[{"text":"\nFrance is a country in Western Europe. It is the largest country in the European Union. The capital of France is Paris.\nWhat is the capital of France?\nWhat is the capital of"}]}
```

## 🏗️ Architecture

### Core Components

- **Multi-process Architecture**: Separate processes for tokenizer, router, detokenizer, and model workers
- **Memory Management**: Efficient GPU memory pool management
- **OpenAI-Compatible API**: FastAPI server with standard endpoints

### Process Structure

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Tokenizer      │    │     Router      │    │  Detokenizer    │
│    Manager      │◄──►│    Process      │◄──►│    Process      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                    ┌───────────────────┐
                    │  Model Workers    │
                    │ (Tensor Parallel) │
                    └───────────────────┘
```

## 🗺️ Roadmap

We welcome contributions to implement these features:

### Model Support

- [ ] **Qwen3 / Qwen3-MoE**: models from Alibaba
- [ ] **More Models**: Additional model architectures

### Advanced Features

- [ ] **DP-Attention**: parallel dp attention mechanisms
- [ ] **Chunked Prefill**: Efficient long context processing
- [ ] **More quantization methods**: Additional quantization methods (GPTQ, SmoothQuant, etc.)
- [ ] **Speculative Decoding**: Faster inference techniques

## 🙏 Acknowledgments

- **[SGLang](https://github.com/sgl-project/sglang)**: For the original inspiration and architectural design
- **[VLLM](https://github.com/vllm-project/vllm)**: For pioneering many optimization techniques in LLM inference
- **[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)**: For the lightweight VLLM implementation reference

## 🤝 Contributing

We strongly encourage contributions! This project is designed to be a collaborative learning resource.

Feel free to open issues, submit pull requests, or start discussions. We're here to learn together!

Before submitting code, please set up pre-commit hooks to ensure code quality:

- Follow the existing code style and structure
- Ensure all pre-commit checks pass before submitting PRs

```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

---

**Note**: This project prioritizes educational clarity over raw performance. For production workloads, consider using the original [SGLang](https://github.com/sgl-project/sglang) framework.
