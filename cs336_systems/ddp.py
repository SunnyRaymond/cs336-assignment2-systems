from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def broadcast_module_parameters(module: nn.Module, src: int = 0) -> None:
    for p in module.parameters():
        dist.broadcast(p.data, src=src)


def allreduce_individual_parameter_grads(module: nn.Module, div_world_size: bool = True) -> None:
    world_size = dist.get_world_size()
    for p in module.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        if div_world_size:
            p.grad.div_(world_size)


def allreduce_flattened_parameter_grads(module: nn.Module, div_world_size: bool = True) -> None:
    world_size = dist.get_world_size()
    grads: list[torch.Tensor] = []
    params: list[nn.Parameter] = []
    for p in module.parameters():
        if p.grad is None:
            continue
        grads.append(p.grad)
        params.append(p)

    if not grads:
        return

    flat = _flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    if div_world_size:
        flat.div_(world_size)

    synced = _unflatten_dense_tensors(flat, grads)
    for p, g_synced in zip(params, synced):
        p.grad.copy_(g_synced)


class DistributedDataParallelIndividualParameters(nn.Module):
    """
    DDP wrapper that synchronizes parameters at init and overlaps per-parameter
    gradient all-reduce with backward via parameter hooks.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed process group must be initialized before constructing DDP wrapper.")
        self.module = module
        self.world_size = dist.get_world_size()
        self._grad_sync_handles: list[dist.Work] = []
        self._hooked_params: list[nn.Parameter] = []

        broadcast_module_parameters(self.module, src=0)
        self._register_gradient_hooks()

    def _register_gradient_hooks(self) -> None:
        # Use unique parameter identity to avoid duplicate hooks in tied-weight settings.
        seen: set[int] = set()
        for p in self.module.parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)

            def _hook(grad: torch.Tensor, _self=self) -> torch.Tensor:
                handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
                _self._grad_sync_handles.append(handle)
                return grad

            p.register_hook(_hook)
            self._hooked_params.append(p)

    def finish_gradient_synchronization(self) -> None:
        for handle in self._grad_sync_handles:
            handle.wait()

        # Average gradients after all reductions complete.
        for p in self._hooked_params:
            if p.grad is not None:
                p.grad.div_(self.world_size)

        self._grad_sync_handles.clear()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def parameters(self, recurse: bool = True) -> Iterable[nn.Parameter]:  # type: ignore[override]
        return self.module.parameters(recurse=recurse)
