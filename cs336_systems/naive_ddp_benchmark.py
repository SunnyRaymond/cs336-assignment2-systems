from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from timeit import default_timer

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM

from .ddp import allreduce_individual_parameter_grads, broadcast_module_parameters


XL_SPEC = {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark naive DDP communication overhead on XL model.")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--batch-size-global", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", default="29530")
    return parser.parse_args()


def _worker(rank: int, args: argparse.Namespace):
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    if not torch.cuda.is_available():
        raise RuntimeError("naive_ddp_benchmark.py expects CUDA (1 node x 2 GPUs).")
    if args.world_size > torch.cuda.device_count():
        raise RuntimeError(f"Requested world_size={args.world_size}, but only {torch.cuda.device_count()} GPUs available.")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=args.world_size)

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=XL_SPEC["d_model"],
        num_layers=XL_SPEC["num_layers"],
        num_heads=XL_SPEC["num_heads"],
        d_ff=XL_SPEC["d_ff"],
        rope_theta=10_000.0,
    ).to(device)
    model.train()
    broadcast_module_parameters(model, src=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    local_bs = args.batch_size_global // args.world_size
    if local_bs * args.world_size != args.batch_size_global:
        raise ValueError("--batch-size-global must be divisible by --world-size.")

    def one_step() -> tuple[float, float]:
        x = torch.randint(0, args.vocab_size, (local_bs, args.context_length), device=device, dtype=torch.long)
        y = torch.randint(0, args.vocab_size, (local_bs, args.context_length), device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        t0 = default_timer()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1))
        loss.backward()

        torch.cuda.synchronize(device)
        comm_start = default_timer()
        allreduce_individual_parameter_grads(model, div_world_size=True)
        torch.cuda.synchronize(device)
        comm_time = default_timer() - comm_start

        optimizer.step()
        torch.cuda.synchronize(device)
        total_time = default_timer() - t0
        return total_time, comm_time

    for _ in range(args.warmup_steps):
        _ = one_step()

    total_times = []
    comm_times = []
    for _ in range(args.measure_steps):
        t_total, t_comm = one_step()
        total_times.append(t_total)
        comm_times.append(t_comm)

    total_mean = statistics.mean(total_times)
    comm_mean = statistics.mean(comm_times)
    payload = {
        "rank": rank,
        "total_step_ms_mean": 1000.0 * total_mean,
        "total_step_ms_stdev": 1000.0 * (statistics.stdev(total_times) if len(total_times) > 1 else 0.0),
        "comm_ms_mean": 1000.0 * comm_mean,
        "comm_ms_stdev": 1000.0 * (statistics.stdev(comm_times) if len(comm_times) > 1 else 0.0),
        "comm_fraction_percent": 100.0 * (comm_mean / total_mean),
    }

    gathered = [None for _ in range(args.world_size)]
    dist.all_gather_object(gathered, payload)

    if rank == 0:
        total_mean_ms = statistics.mean([g["total_step_ms_mean"] for g in gathered if g is not None])
        comm_mean_ms = statistics.mean([g["comm_ms_mean"] for g in gathered if g is not None])
        comm_frac = statistics.mean([g["comm_fraction_percent"] for g in gathered if g is not None])
        result = {
            "setup": {
                "world_size": args.world_size,
                "backend": "nccl",
                "model_size": "xl",
                "batch_size_global": args.batch_size_global,
                "context_length": args.context_length,
                "vocab_size": args.vocab_size,
                "warmup_steps": args.warmup_steps,
                "measure_steps": args.measure_steps,
                "method": "naive_ddp_individual_gradient_allreduce_after_backward",
            },
            "result": {
                "time_per_step_ms": total_mean_ms,
                "time_in_gradient_communication_ms": comm_mean_ms,
                "communication_fraction_percent": comm_frac,
            },
            "per_rank": gathered,
        }
        print(json.dumps(result, indent=2))

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    mp.spawn(_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()

