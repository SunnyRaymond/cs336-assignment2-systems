from __future__ import annotations

import argparse
import copy
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from .ddp import allreduce_individual_parameter_grads, broadcast_module_parameters


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 16, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify naive DDP correctness against single-process baseline.")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--local-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", default="29520")
    return parser.parse_args()


def _worker(rank: int, args: argparse.Namespace):
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    if use_cuda:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=args.world_size)
    torch.manual_seed(1337 + rank)

    ddp_model = TinyModel().to(device)
    broadcast_module_parameters(ddp_model, src=0)
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    ref_model = None
    ref_optimizer = None
    if rank == 0:
        ref_model = copy.deepcopy(ddp_model).to(device)
        ref_optimizer = optim.SGD(ref_model.parameters(), lr=args.lr)

    global_batch = args.local_batch_size * args.world_size
    for step in range(args.steps):
        torch.manual_seed(2025 + step)
        x_all = torch.randn(global_batch, 32, device=device)
        y_all = torch.randn(global_batch, 16, device=device)

        start = rank * args.local_batch_size
        end = start + args.local_batch_size
        x_local = x_all[start:end]
        y_local = y_all[start:end]

        ddp_optimizer.zero_grad(set_to_none=True)
        ddp_loss = loss_fn(ddp_model(x_local), y_local)
        ddp_loss.backward()
        allreduce_individual_parameter_grads(ddp_model, div_world_size=True)
        ddp_optimizer.step()

        if rank == 0 and ref_model is not None and ref_optimizer is not None:
            ref_optimizer.zero_grad(set_to_none=True)
            ref_loss = loss_fn(ref_model(x_all), y_all)
            ref_loss.backward()
            ref_optimizer.step()

            for p_ref, p_ddp in zip(ref_model.parameters(), ddp_model.parameters()):
                if not torch.allclose(p_ref, p_ddp, rtol=1e-5, atol=1e-6):
                    raise AssertionError(f"Mismatch at step={step}")

    if rank == 0:
        print("Naive DDP verification passed: rank-0 DDP weights match single-process baseline.")

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    mp.spawn(_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()

