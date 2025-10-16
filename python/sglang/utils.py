"""Common utilities."""


def get_available_gpu_memory(gpu_id, distributed=True):
    """
    Get available memory for cuda:gpu_id device.
    When distributed is True, the available memory is the minimum available memory of all GPUs.
    """
    import torch

    num_gpus = torch.cuda.device_count()
    assert gpu_id < num_gpus

    if torch.cuda.current_device() != gpu_id:
        print(
            f"WARN: current device is not {gpu_id}, but {torch.cuda.current_device()}, ",
            "which may cause useless memory allocation for torch CUDA context.",
        )

    free_gpu_memory, _ = torch.cuda.mem_get_info(gpu_id)

    if distributed:
        tensor = torch.tensor(free_gpu_memory, dtype=torch.float32).to(
            torch.device("cuda", gpu_id)
        )
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        free_gpu_memory = tensor.item()

    return free_gpu_memory / (1 << 30)
