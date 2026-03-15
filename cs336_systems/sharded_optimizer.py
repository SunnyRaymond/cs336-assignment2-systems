from __future__ import annotations

from typing import Any, Type

import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    """
    Optimizer-state sharding wrapper.

    Each rank owns a shard of parameters and keeps optimizer state only for its
    local shard. After local optimizer.step(), each rank broadcasts its updated
    parameter shard so all model replicas stay synchronized.
    """

    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict(kwargs)

        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        # Global parameter index (deterministic by parameter traversal order).
        self._next_param_index = 0
        self._param_index_by_id: dict[int, int] = {}
        self._param_owner_by_id: dict[int, int] = {}
        self._seen_param_ids: set[int] = set()
        self._all_unique_params_in_order: list[torch.nn.Parameter] = []

        self._local_optimizer: torch.optim.Optimizer | None = None
        self._local_param_groups: list[dict[str, Any]] = []

        # Required by assignment: call Optimizer superclass constructor.
        super().__init__(params, defaults={})
        self._rebuild_local_optimizer()

    def _owner_for_param(self, p: torch.nn.Parameter) -> int:
        pid = id(p)
        if pid not in self._param_index_by_id:
            idx = self._next_param_index
            self._next_param_index += 1
            self._param_index_by_id[pid] = idx
            self._param_owner_by_id[pid] = idx % self.world_size
            self._all_unique_params_in_order.append(p)
        return self._param_owner_by_id[pid]

    def _localize_param_group(self, global_group: dict[str, Any]) -> dict[str, Any]:
        local_group = {k: v for k, v in global_group.items() if k != "params"}
        params = global_group["params"]
        local_params = []
        for p in params:
            if self._owner_for_param(p) == self.rank:
                local_params.append(p)
        local_group["params"] = local_params
        return local_group

    def _rebuild_local_optimizer(self) -> None:
        self._local_param_groups = []
        for group in self.param_groups:
            local_group = self._localize_param_group(group)
            if len(local_group["params"]) > 0:
                self._local_param_groups.append(local_group)

        if len(self._local_param_groups) == 0:
            self._local_optimizer = None
        else:
            self._local_optimizer = self.optimizer_cls(self._local_param_groups, **self.optimizer_kwargs)

    def add_param_group(self, param_group: dict[str, Any]):
        # Ensure params are materialized list (Optimizer expects this).
        new_group = dict(param_group)
        new_group["params"] = list(new_group["params"])
        super().add_param_group(new_group)
        self._rebuild_local_optimizer()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        loss = None
        if self._local_optimizer is not None:
            loss = self._local_optimizer.step(closure=closure, **kwargs)

        # Broadcast each updated parameter from its owner rank.
        for p in self._all_unique_params_in_order:
            owner = self._owner_for_param(p)
            if self.world_size > 1:
                dist.broadcast(p.data, src=owner)

        return loss

