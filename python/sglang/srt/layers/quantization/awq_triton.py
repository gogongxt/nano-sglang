# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/awq_triton.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
AWQ Triton内核实现

该模块实现了AWQ量化的GPU加速内核，使用Triton语言编写。
主要功能是对打包的4位量化权重进行反量化并执行矩阵乘法。
"""

import torch
import triton
import triton.language as tl

# 支持的分组大小（-1表示全通道分组）
AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]


@triton.jit
def awq_gemm_kernel(
    a_ptr,  # 输入矩阵A的指针 [M, K]
    b_ptr,  # 量化权重矩阵B的指针 [K, N//8]，每个int32包含8个4位权重
    c_ptr,  # 输出矩阵C的指针 [split_k_iters, M, N]
    zeros_ptr,  # 量化零点指针 [K//G, N//8]
    scales_ptr,  # 缩放因子指针 [K//G, N]
    M,  # 矩阵A的行数（批量大小 × 序列长度）
    N,  # 矩阵B的列数（输出维度）
    K,  # 矩阵A的列数 / 矩阵B的行数（输入维度）
    group_size,  # 量化分组大小
    BLOCK_SIZE_M: tl.constexpr,  # M维度的块大小
    BLOCK_SIZE_N: tl.constexpr,  # N维度的块大小
    BLOCK_SIZE_K: tl.constexpr,  # K维度的块大小
    SPLIT_K: tl.constexpr,  # K维度的并行迭代次数（必须是2的幂）
):
    """AWQ量化矩阵乘法的Triton内核实现

    该内核执行以下操作：
    1. 加载打包的4位量化权重
    2. 使用缩放因子和零点进行反量化
    3. 与输入矩阵执行矩阵乘法
    4. 将结果写入输出矩阵

    使用分块矩阵乘法和K维度并行来提高性能。
    """
    # 获取程序ID：pid处理(M,N)平铺，pid_z处理K维度分割
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)

    # 计算N维度上的程序块数量
    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
    # num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # 将一维程序ID映射到二维(M,N)网格
    pid_m = pid // num_pid_n  # M维度的程序ID
    pid_n = pid % num_pid_n  # N维度的程序ID

    # 获取累加器的数据类型
    accumulator_dtype = c_ptr.type.element_ty

    # 初始化累加器为零矩阵 [BLOCK_SIZE_M, BLOCK_SIZE_N]
    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
    # accumulator = tl.arange(0, BLOCK_SIZE_N)
    # accumulator = tl.broadcast_to(accumulator[None, :],
    # (BLOCK_SIZE_M, BLOCK_SIZE_N))
    # accumulator = accumulator & 0x0
    # accumulator = accumulator.to(accumulator_dtype)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    # 创建AWQ逆序张量：[0, 4, 1, 5, 2, 6, 3, 7]
    # 用于将打包的权重解包到正确的顺序
    # AWQ打包格式将权重按照特定顺序存储，需要恢复到标准顺序
    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)

    # Create the necessary shifts to use to unpack.
    # 创建解包所需的位移量
    # 每个4位权重需要移位到正确的位置：0, 4, 8, 12, 16, 20, 24, 28
    shifts = reverse_awq_order_tensor * 4  # [0, 16, 4, 20, 8, 24, 12, 28]
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    # Offsets and masks.
    # ========== 计算内存偏移量和掩码 ==========
    # M维度的偏移（输入矩阵的行）
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_am = offsets_am < M  # 边界检查：防止越界

    # N维度的偏移（权重矩阵的列，打包前是N，打包后是N//8）
    offsets_bn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_bn = offsets_bn < N // 8

    # 零点张量的N维度偏移（也是打包的）
    offsets_zn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_zn = offsets_zn < N // 8

    # 缩放因子的N维度偏移（未打包）
    offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    masks_sn = offsets_sn < N

    # K维度的偏移（共享维度，考虑K维度分割）
    offsets_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # 计算全局内存指针
    # A矩阵：按行优先，形状 [M, K]
    offsets_a = K * offsets_am[:, None] + offsets_k[None, :]
    # B矩阵：按行优先，形状 [K, N//8]（打包后）
    offsets_b = (N // 8) * offsets_k[:, None] + offsets_bn[None, :]

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    # ========== 主循环：在K维度上进行分块矩阵乘法 ==========
    # NOTE: Use this in TRITON_INTERPRET=1 mode instead of tl.cdiv
    # block_offset = BLOCK_SIZE_K * SPLIT_K
    # for k in range(0, (K + block_offset - 1) // (block_offset)):
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # 加载输入矩阵A的块 [BLOCK_SIZE_M, BLOCK_SIZE_K]
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a, other=0.0)

        # 加载量化权重矩阵B的块 [BLOCK_SIZE_K, BLOCK_SIZE_N // 8]
        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b, other=0.0)
        # 解交错操作：将打包的权重恢复到正确布局
        # AWQ使用特定的交错模式来优化内存访问
        b = tl.interleave(b, b)  # 第一次交错
        b = tl.interleave(b, b)  # 第二次交错
        b = tl.interleave(b, b)  # 第三次交错

        # Dequantize b.
        # ========== 反量化权重 ==========
        # 计算当前K块对应的缩放因子和零点的索引
        offsets_szk = (
            BLOCK_SIZE_K * SPLIT_K * k + pid_z * BLOCK_SIZE_K
        ) // group_size + tl.arange(0, 1)

        # 加载零点张量 [BLOCK_SIZE_K, BLOCK_SIZE_N // 8]
        offsets_z = (N // 8) * offsets_szk[:, None] + offsets_zn[None, :]
        masks_zk = offsets_szk < K // group_size
        masks_z = masks_zk[:, None] & masks_zn[None, :]
        zeros_ptrs = zeros_ptr + offsets_z
        zeros = tl.load(zeros_ptrs, mask=masks_z, other=0.0)
        # 解交错零点张量
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        # 广播到完整形状 [BLOCK_SIZE_K, BLOCK_SIZE_N]
        zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        # 加载缩放因子张量 [BLOCK_SIZE_K, BLOCK_SIZE_N]
        offsets_s = N * offsets_szk[:, None] + offsets_sn[None, :]
        masks_sk = offsets_szk < K // group_size
        masks_s = masks_sk[:, None] & masks_sn[None, :]
        scales_ptrs = scales_ptr + offsets_s
        scales = tl.load(scales_ptrs, mask=masks_s, other=0.0)
        scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        # ========== 执行反量化 ==========
        # 1. 提取4位权重：右移并掩码
        b = (b >> shifts) & 0xF
        # 2. 提取4位零点：右移并掩码
        zeros = (zeros >> shifts) & 0xF
        # 3. 反量化公式：dequantized_weight = (quantized_weight - zero_point) * scale
        b = (b - zeros) * scales
        # 转换为输出数据类型
        b = b.to(c_ptr.type.element_ty)

        # Accumulate results.
        # ========== 矩阵乘法累加 ==========
        # 执行块矩阵乘法：accumulator += a @ b
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        # 更新指针到下一个K块
        offsets_k += BLOCK_SIZE_K * SPLIT_K
        a_ptrs += BLOCK_SIZE_K * SPLIT_K
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * (N // 8)

    # ========== 写回结果 ==========
    # 转换为输出数据类型
    c = accumulator.to(c_ptr.type.element_ty)
    # 计算输出矩阵的偏移量
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # 输出指针：考虑K维度分割的偏移
    c_ptrs = c_ptr + pid_z * N * M + N * offs_cm[:, None] + offs_cn[None, :]
    # 边界检查并写回结果
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# input   - [M, K]
# qweight - [K, N // 8]
# qzeros  - [K // G, N // 8]
# scales  - [K // G, N]
# split_k_iters - parallelism along K-dimension, int, power of 2.
def awq_gemm_triton(
    input: torch.Tensor,  # 输入张量 [M, K]
    qweight: torch.Tensor,  # 量化权重张量 [K, N // 8]，每个int32包含8个4位权重
    scales: torch.Tensor,  # 缩放因子张量 [K // G, N]
    qzeros: torch.Tensor,  # 量化零点张量 [K // G, N // 8]
    split_k_iters: int,  # K维度并行迭代次数，必须是2的幂，最大32
    block_size_m: int = 32,  # M维度块大小
    block_size_n: int = 32,  # N维度块大小
    block_size_k: int = 32,  # K维度块大小
) -> torch.Tensor:
    """AWQ量化矩阵乘法的包装函数

    该函数为Triton内核设置参数并启动计算。

    张量形状说明：
    - input:   [M, K] - 输入激活值矩阵
    - qweight: [K, N // 8] - 量化权重，打包到int32中
    - qzeros:  [K // G, N // 8] - 量化零点，打包到int32中
    - scales:  [K // G, N] - 缩放因子，FP16类型
    - 输出:    [M, N] - 结果矩阵

    其中：
    - M = batch_size × seq_len（批量大小 × 序列长度）
    - K = input_dim（输入维度）
    - N = output_dim（输出维度）
    - G = group_size（分组大小）

    Args:
        input: 输入张量 [M, K]
        qweight: 量化权重张量 [K, N // 8]
        scales: 缩放因子张量 [K // G, N]
        qzeros: 量化零点张量 [K // G, N // 8]
        split_k_iters: K维度并行迭代次数（必须是2的幂）
        block_size_m: M维度块大小（默认32）
        block_size_n: N维度块大小（默认32）
        block_size_k: K维度块大小（默认32）

    Returns:
        输出张量 [M, N]
    """
    M, K = input.shape
    N = qweight.shape[1] * 8
    group_size = qweight.shape[0] // qzeros.shape[0]

    # ========== 形状验证 ==========
    assert N > 0 and K > 0 and M > 0
    assert qweight.shape[0] == K and qweight.shape[1] == N // 8
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == N // 8
    assert scales.shape[0] == K // group_size and scales.shape[1] == N
    # split_k_iters必须是2的幂且非零
    assert split_k_iters & (split_k_iters - 1) == 0 and split_k_iters != 0
    assert split_k_iters <= 32
    assert group_size <= K
    # 分组大小必须是支持的值之一
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    # 定义网格函数：确定内核启动的块数量
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        split_k_iters,
    )

    # 分配结果张量 [split_k_iters, M, N]
    # 使用K维度分割，需要在最后对split_k_iters维度求和
    result = torch.zeros((split_k_iters, M, N), dtype=scales.dtype, device=input.device)

    # ========== 启动Triton内核 ==========
    # A = input, B = qweight, C = result
    # A = M x K, B = K x N, C = M x N
    awq_gemm_kernel[grid](
        input,
        qweight,
        result,
        qzeros,
        scales,
        M,
        N,
        K,
        group_size,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        SPLIT_K=split_k_iters,
    )

    # 对K维度分割的部分求和，得到最终结果 [M, N]
    result = result.sum(0)

    return result
