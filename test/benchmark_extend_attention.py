"""Simple benchmark script for extend attention kernels"""

import os
import sys
import time

import torch
import torch.nn as nn

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sglang.srt.layers.extend_attention import extend_attention_fwd
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.managers.router.model_runner import ForwardMode

try:
    import flashinfer

    FLASHINFER_AVAILABLE = True
except ImportError:
    print("Warning: flashinfer not available")
    FLASHINFER_AVAILABLE = False


def benchmark_triton_direct(
    prefix_len: int,
    extend_len: int,
    batch_size: int = 1,
    num_heads: int = 32,
    num_kv_heads: int = 32,
    head_dim: int = 128,
):
    """Benchmark the direct triton kernel"""
    print(
        f"\n[TRITON DIRECT] prefix={prefix_len}, extend={extend_len}, batch={batch_size}"
    )

    # Create input tensors
    q_extend = torch.randn(
        batch_size * extend_len, num_heads, head_dim, dtype=torch.float16, device="cuda"
    )
    k_extend = torch.randn(
        batch_size * extend_len,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    v_extend = torch.randn(
        batch_size * extend_len,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    o_extend = torch.empty_like(q_extend)

    # Create buffers (need to fill prefix with random data)
    max_total_tokens = batch_size * (prefix_len + extend_len)
    k_buffer = torch.randn(
        (max_total_tokens, num_kv_heads, head_dim), dtype=torch.float16, device="cuda"
    )
    v_buffer = torch.randn(
        (max_total_tokens, num_kv_heads, head_dim), dtype=torch.float16, device="cuda"
    )

    # Create req_to_tokens mapping
    req_to_tokens = torch.full(
        (batch_size, max_total_tokens), -1, dtype=torch.int32, device="cuda"
    )
    for i in range(batch_size):
        # Prefix positions
        for j in range(prefix_len):
            pos = i * (prefix_len + extend_len) + j
            req_to_tokens[i, j] = pos
        # Extend positions
        for j in range(extend_len):
            pos = i * (prefix_len + extend_len) + prefix_len + j
            req_to_tokens[i, prefix_len + j] = pos

    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.full(
        (batch_size,), prefix_len + extend_len, dtype=torch.int32, device="cuda"
    )
    b_start_loc_extend = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len_extend = torch.full(
        (batch_size,), extend_len, dtype=torch.int32, device="cuda"
    )

    # Warmup
    for _ in range(10):
        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            req_to_tokens,
            b_req_idx,
            b_seq_len,
            b_start_loc_extend,
            b_seq_len_extend,
            extend_len,
        )

    torch.cuda.synchronize()

    # Benchmark
    num_iterations = 100
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            req_to_tokens,
            b_req_idx,
            b_seq_len,
            b_start_loc_extend,
            b_seq_len_extend,
            extend_len,
        )

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = total_time / num_iterations * 1000  # ms
    total_tokens = batch_size * extend_len
    throughput = total_tokens / (avg_time / 1000)  # tokens/s

    print(f"  Time per iteration: {avg_time:.3f} ms")
    print(f"  Throughput: {throughput:,.0f} tokens/s")
    print(f"  Total tokens processed: {total_tokens:,}")

    return avg_time, throughput


class MockReqToTokenPool:
    """Mock ReqToTokenPool for benchmark"""

    def __init__(self, batch_size, max_context_len):
        self.req_to_token = torch.full(
            (batch_size, max_context_len), -1, dtype=torch.int32, device="cuda"
        )


class MockTokenToKVPool:
    """Mock TokenToKVPool for benchmark"""

    def __init__(self, size, dtype, head_num, head_dim, layer_num):
        # Create kv_data in 4D format: [size, 2, head_num, head_dim]
        self.kv_data = torch.empty(
            (size, 2, head_num, head_dim), dtype=dtype, device="cuda"
        )

    def get_key_buffer(self, layer_id):
        return self.kv_data[:, 0]

    def get_value_buffer(self, layer_id):
        return self.kv_data[:, 1]

    def get_kv_data_flashinfer(self, layer_id):
        """Get kv_data in 5D format for flashinfer: [num_pages, 2, 1, num_kv_heads, head_dim]"""
        # Reshape to [size, 2, 1, head_num, head_dim] for NHD layout
        size, _, head_num, head_dim = self.kv_data.shape
        return self.kv_data.view(size, 2, 1, head_num, head_dim)


class SimpleMockInputMetadata:
    """Simple mock input metadata"""

    def __init__(
        self,
        mode=ForwardMode.EXTEND,
        prefix_len=0,
        extend_len=0,
        batch_size=1,
        num_kv_heads=32,
        head_dim=128,
        layer_id=0,
    ):
        self.forward_mode = mode

        # Create mock pools
        max_total_tokens = batch_size * (prefix_len + extend_len)
        self.req_to_token_pool = MockReqToTokenPool(batch_size, max_total_tokens)
        self.token_to_kv_pool = MockTokenToKVPool(
            max_total_tokens, torch.float16, num_kv_heads, head_dim, 1
        )

        # Fill req_to_token mapping
        for i in range(batch_size):
            # Prefix positions
            for j in range(prefix_len):
                pos = i * (prefix_len + extend_len) + j
                self.req_to_token_pool.req_to_token[i, j] = pos
            # Extend positions
            for j in range(extend_len):
                pos = i * (prefix_len + extend_len) + prefix_len + j
                self.req_to_token_pool.req_to_token[i, prefix_len + j] = pos

        self.req_pool_indices = torch.arange(
            batch_size, dtype=torch.int32, device="cuda"
        )
        self.seq_lens = torch.full(
            (batch_size,), prefix_len + extend_len, dtype=torch.int32, device="cuda"
        )
        self.extend_start_loc = torch.zeros(
            batch_size, dtype=torch.int32, device="cuda"
        )
        self.extend_seq_lens = torch.full(
            (batch_size,), extend_len, dtype=torch.int32, device="cuda"
        )
        self.max_extend_len = extend_len
        self.max_seq_len = prefix_len + extend_len

        # For decode mode (not used in extend benchmark)
        self.start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        self.other_kv_index = None
        self.total_num_tokens = batch_size * extend_len

        # Cache location
        self.out_cache_loc = torch.arange(
            batch_size * extend_len, dtype=torch.int64, device="cuda"
        )
        self.out_cache_cont_start = None
        self.out_cache_cont_end = None

        # Flashinfer args (will be initialized separately if needed)
        self.use_flashinfer = False
        self.prefill_wrapper = None
        self.decode_wrapper = None
        self.qo_indptr = None
        self.kv_indptr = None
        self.kv_indices = None
        self.kv_last_page_len = None

    def init_flashinfer_args(self, num_qo_heads, num_kv_heads, head_dim, page_size=1):
        """Initialize flashinfer-specific arguments"""
        if not FLASHINFER_AVAILABLE:
            raise ImportError("flashinfer not available")

        batch_size = self.seq_lens.shape[0]
        total_kv_len = self.seq_lens.sum().item()

        # qo_indptr for extend mode
        self.qo_indptr = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device="cuda"
        )
        self.qo_indptr[1:] = torch.cumsum(self.extend_seq_lens, dim=0)

        # kv_indptr and kv_indices for paged KV cache
        # For simplicity, we use page_size=1 (each token is a separate page)
        self.kv_indptr = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device="cuda"
        )
        self.kv_indptr[1:] = torch.cumsum(self.seq_lens, dim=0)

        # kv_indices: block indices for each request
        self.kv_indices = torch.cat(
            [
                torch.arange(start, start + seq_len, dtype=torch.int32, device="cuda")
                for start, seq_len in zip(
                    [0] + self.kv_indptr[1:-1].tolist(), self.seq_lens.tolist()
                )
            ]
        )

        # kv_last_page_len: all pages are full (page_size=1)
        self.kv_last_page_len = torch.ones(
            (batch_size,), dtype=torch.int32, device="cuda"
        )

        # Create workspace buffer for flashinfer
        workspace_buffer = torch.empty(
            (1024 * 1024 * 4,), dtype=torch.float32, device="cuda"
        )

        # Create prefill wrapper for extend mode
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer
        )

        self.prefill_wrapper.begin_forward(
            self.qo_indptr,
            self.kv_indptr,
            self.kv_indices,
            self.kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,  # Use causal=True for extend mode
        )

        self.use_flashinfer = True


def benchmark_radix_attention(
    prefix_len: int,
    extend_len: int,
    batch_size: int = 1,
    num_heads: int = 32,
    num_kv_heads: int = 32,
    head_dim: int = 128,
    use_flashinfer: bool = False,
):
    """Benchmark RadixAttention with triton or flashinfer backend"""
    if use_flashinfer and not FLASHINFER_AVAILABLE:
        print(f"  Flashinfer not available, skipping")
        return None, None

    backend = "FLASHINFER" if use_flashinfer else "TRITON"
    print(
        f"\n[RADIX ATTENTION - {backend}] prefix={prefix_len}, extend={extend_len}, batch={batch_size}"
    )

    # Create RadixAttention module
    layer_id = 0
    scaling = 1.0 / (head_dim**0.5)

    # Set global model mode for flashinfer
    import sglang.srt.managers.router.model_runner as model_runner_module

    if use_flashinfer:
        model_runner_module.global_model_mode = ["flashinfer"]
    else:
        model_runner_module.global_model_mode = []

    attention = RadixAttention(num_heads, head_dim, scaling, num_kv_heads, layer_id)
    attention.cuda()

    # Create input tensors
    q = torch.randn(
        batch_size * extend_len,
        num_heads * head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    k = torch.randn(
        batch_size * extend_len,
        num_kv_heads * head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    v = torch.randn(
        batch_size * extend_len,
        num_kv_heads * head_dim,
        dtype=torch.float16,
        device="cuda",
    )

    # Create mock input metadata
    input_metadata = SimpleMockInputMetadata(
        mode=ForwardMode.EXTEND,
        prefix_len=prefix_len,
        extend_len=extend_len,
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        layer_id=layer_id,
    )

    # Initialize flashinfer args if needed
    if use_flashinfer:
        try:
            input_metadata.init_flashinfer_args(num_heads, num_kv_heads, head_dim)
        except Exception as e:
            print(f"  Failed to initialize flashinfer: {e}")
            return None, None

    # Warmup
    for _ in range(10):
        output = attention(q, k, v, input_metadata)
        torch.cuda.synchronize()

    # Benchmark
    num_iterations = 100
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        output = attention(q, k, v, input_metadata)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = total_time / num_iterations * 1000  # ms
    total_tokens = batch_size * extend_len
    throughput = total_tokens / (avg_time / 1000)  # tokens/s

    print(f"  Time per iteration: {avg_time:.3f} ms")
    print(f"  Throughput: {throughput:,.0f} tokens/s")
    print(f"  Total tokens processed: {total_tokens:,}")

    return avg_time, throughput


def main():
    print("=" * 80)
    print("EXTEND ATTENTION BENCHMARK")
    print("=" * 80)

    batch_size = 1
    num_heads = 32
    num_kv_heads = 32
    head_dim = 128

    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Query heads: {num_heads}")
    print(f"  KV heads: {num_kv_heads}")
    print(f"  Head dimension: {head_dim}")

    test_cases = [
        (0, 1024),
        (0, 2048),
        (0, 4096),
        (0, 8192),
        (1024, 1024),
        (1024, 2048),
        (1024, 4096),
        (1024, 8192),
        (2048, 1024),
        (2048, 2048),
        (2048, 4096),
        (2048, 8192),
        (4096, 1024),
        (4096, 2048),
        (4096, 4096),
        (4096, 8192),
    ]

    results = []

    for prefix_len, extend_len in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST CASE: prefix={prefix_len}, extend={extend_len}")
        print(f"{'='*60}")

        case_result = {
            "prefix": prefix_len,
            "extend": extend_len,
            "triton_direct": None,
            "radix_triton": None,
            "radix_flashinfer": None,
        }

        # Benchmark triton direct
        try:
            time_ms, throughput = benchmark_triton_direct(
                prefix_len, extend_len, batch_size, num_heads, num_kv_heads, head_dim
            )
            case_result["triton_direct"] = {
                "time_ms": time_ms,
                "throughput": throughput,
            }
        except Exception as e:
            print(f"  Triton direct failed: {e}")
            case_result["triton_direct"] = {"error": str(e)}

        # Benchmark RadixAttention with triton
        try:
            time_ms, throughput = benchmark_radix_attention(
                prefix_len,
                extend_len,
                batch_size,
                num_heads,
                num_kv_heads,
                head_dim,
                use_flashinfer=False,
            )
            case_result["radix_triton"] = {"time_ms": time_ms, "throughput": throughput}
        except Exception as e:
            print(f"  RadixAttention (triton) failed: {e}")
            case_result["radix_triton"] = {"error": str(e)}

        # Benchmark RadixAttention with flashinfer
        try:
            time_ms, throughput = benchmark_radix_attention(
                prefix_len,
                extend_len,
                batch_size,
                num_heads,
                num_kv_heads,
                head_dim,
                use_flashinfer=True,
            )
            if time_ms is not None and throughput is not None:
                case_result["radix_flashinfer"] = {
                    "time_ms": time_ms,
                    "throughput": throughput,
                }
            else:
                case_result["radix_flashinfer"] = {
                    "error": "Flashinfer not available or failed"
                }
                print(f"  RadixAttention (flashinfer) not available or failed")
        except Exception as e:
            print(f"  RadixAttention (flashinfer) failed: {e}")
            case_result["radix_flashinfer"] = {"error": str(e)}

        results.append(case_result)

    # Print summary table
    print(f"\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")
    print(
        f"{'Prefix':>8} {'Extend':>8} {'Triton Direct':>20} {'Radix Triton':>20} {'Radix Flashinfer':>20}"
    )
    print(
        f"{'':>8} {'':>8} {'ms':>8} {'tokens/s':>12} {'ms':>8} {'tokens/s':>12} {'ms':>8} {'tokens/s':>12}"
    )
    print(f"{'-'*100}")

    for result in results:
        prefix = result["prefix"]
        extend = result["extend"]

        # Triton direct
        td = result["triton_direct"]
        td_ms = (
            f"{td['time_ms']:.2f}"
            if isinstance(td, dict) and "time_ms" in td
            else "N/A"
        )
        td_tps = (
            f"{td['throughput']:,.0f}"
            if isinstance(td, dict) and "throughput" in td
            else "N/A"
        )

        # Radix triton
        rt = result["radix_triton"]
        rt_ms = (
            f"{rt['time_ms']:.2f}"
            if isinstance(rt, dict) and "time_ms" in rt
            else "N/A"
        )
        rt_tps = (
            f"{rt['throughput']:,.0f}"
            if isinstance(rt, dict) and "throughput" in rt
            else "N/A"
        )

        # Radix flashinfer
        rf = result["radix_flashinfer"]
        rf_ms = (
            f"{rf['time_ms']:.2f}"
            if isinstance(rf, dict) and "time_ms" in rf
            else "N/A"
        )
        rf_tps = (
            f"{rf['throughput']:,.0f}"
            if isinstance(rf, dict) and "throughput" in rf
            else "N/A"
        )

        print(
            f"{prefix:>8} {extend:>8} {td_ms:>8} {td_tps:>12} {rt_ms:>8} {rt_tps:>12} {rf_ms:>8} {rf_tps:>12}"
        )

    print(f"\n{'='*100}")
    print("PERFORMANCE RATIOS (Radix Triton vs Triton Direct)")
    print(f"{'='*100}")
    print(f"{'Prefix':>8} {'Extend':>8} {'Speedup':>12} {'Overhead %':>12}")
    print(f"{'-'*100}")

    for result in results:
        prefix = result["prefix"]
        extend = result["extend"]

        td = result["triton_direct"]
        rt = result["radix_triton"]

        if (
            isinstance(td, dict)
            and "time_ms" in td
            and isinstance(rt, dict)
            and "time_ms" in rt
        ):
            td_time = td["time_ms"]
            rt_time = rt["time_ms"]

            if td_time > 0:
                speedup = td_time / rt_time
                overhead = ((rt_time - td_time) / td_time) * 100
                print(f"{prefix:>8} {extend:>8} {speedup:>12.2f}x {overhead:>11.1f}%")
            else:
                print(f"{prefix:>8} {extend:>8} {'N/A':>12} {'N/A':>12}")
        else:
            print(f"{prefix:>8} {extend:>8} {'N/A':>12} {'N/A':>12}")

    print(f"\n{'='*100}")
    print("PERFORMANCE RATIOS (Radix Flashinfer vs Triton Direct)")
    print(f"{'='*100}")
    print(f"{'Prefix':>8} {'Extend':>8} {'Speedup':>12} {'Overhead %':>12}")
    print(f"{'-'*100}")

    for result in results:
        prefix = result["prefix"]
        extend = result["extend"]

        td = result["triton_direct"]
        rf = result["radix_flashinfer"]

        if (
            isinstance(td, dict)
            and "time_ms" in td
            and isinstance(rf, dict)
            and "time_ms" in rf
        ):
            td_time = td["time_ms"]
            rf_time = rf["time_ms"]

            if td_time > 0:
                speedup = td_time / rf_time
                overhead = ((rf_time - td_time) / td_time) * 100
                print(f"{prefix:>8} {extend:>8} {speedup:>12.2f}x {overhead:>11.1f}%")
            else:
                print(f"{prefix:>8} {extend:>8} {'N/A':>12} {'N/A':>12}")
        else:
            print(f"{prefix:>8} {extend:>8} {'N/A':>12} {'N/A':>12}")


if __name__ == "__main__":
    main()
