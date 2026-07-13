# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Distributed-training helpers (world-size, rank, all_gather, reduce_dict)."""

import pickle
from typing import Any

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    """Return True if torch.distributed is available and has been initialised."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """Return the number of processes in the current distributed group."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Return the rank of the current process in the distributed group."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """Return True if the current process is rank 0."""
    return get_rank() == 0


def save_on_master(obj: Any, f: Any, *args: Any, **kwargs: Any) -> None:
    """Save *obj* to *f* only on the main process (rank 0).

    Args:
        obj: Object to save.
        f: File path or file-like object passed to ``torch.save``.
        *args: Additional positional arguments forwarded to ``torch.save``. **kwargs: Additional keyword arguments
        forwarded to ``torch.save``.
    """
    if is_main_process():
        torch.save(obj, f, *args, **kwargs)


def all_gather(data: Any) -> list[Any]:
    """Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: Any picklable object.

    Returns:
        List of data gathered from each rank.
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # Serialize to a byte tensor on the active device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer = pickle.dumps(data)
    tensor = torch.tensor(bytearray(buffer), dtype=torch.uint8, device=device)

    # obtain Tensor size of each rank
    local_size = tensor.numel()
    local_size_tensor = torch.tensor([local_size], device=device)
    size_list = [torch.tensor([0], device=device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size_tensor)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=device))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device=device)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict: dict[str, torch.Tensor], average: bool = True) -> dict[str, torch.Tensor]:
    """Reduce values in *input_dict* across all processes.

    Args:
        input_dict: Dict whose values will be reduced.
        average: If True, compute the mean across ranks; otherwise compute the sum.

    Returns:
        Dict with the same keys as *input_dict*, with values averaged/summed across ranks.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
