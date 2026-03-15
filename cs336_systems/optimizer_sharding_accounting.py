from __future__ import annotations

import argparse
import json
import os
import statistics
from timeit import default_timer

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM

from .ddp import broadcast_module_parameters
from .sharded_optimizer import ShardedOptimizer


MODEL_SPECS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Accounting benchmark for optimizer state sharding.")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--model-size", choices=list(MODEL_SPECS.keys()), default="xl")
    parser.add_argument("--batch-size-global", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", default="29550")
    return parser.parse_args()


def _unique_params(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    out = []
    seen = set()
    for p in module.parameters():
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def _estimate_state_bytes_non_sharded(params: list[torch.nn.Parameter]) -> int:
    # AdamW: exp_avg + exp_avg_sq in fp32.
    return int(sum(p.numel() for p in params) * 2 * 4)


def _estimate_state_bytes_sharded(params: list[torch.nn.Parameter], rank: int, world_size: int) -> int:
    owned_numel = 0
    for idx, p in enumerate(params):
        if (idx % world_size) == rank:
            owned_numel += p.numel()
    return int(owned_numel * 2 * 4)


def _run_mode(
    mode: str,
    args: argparse.Namespace,
    rank: int,
    device: torch.device,
) -> dict:
    spec = MODEL_SPECS[args.model_size]
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=spec["d_model"],
        num_layers=spec["num_layers"],
        num_heads=spec["num_heads"],
        d_ff=spec["d_ff"],
        rope_theta=10_000.0,
    ).to(device)
    model.train()
    broadcast_module_parameters(model, src=0)

    if mode == "non_sharded":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif mode == "sharded":
        optimizer = ShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=args.lr)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    unique_params = _unique_params(model)
    param_bytes = int(sum(p.numel() * p.element_size() for p in unique_params))
    if mode == "non_sharded":
        state_bytes_est = _estimate_state_bytes_non_sharded(unique_params)
    else:
        state_bytes_est = _estimate_state_bytes_sharded(unique_params, rank=rank, world_size=args.world_size)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device=device)
    init_allocated = torch.cuda.memory_allocated(device=device) / (1024 ** 2)
    init_reserved = torch.cuda.memory_reserved(device=device) / (1024 ** 2)

    local_bs = args.batch_size_global // args.world_size
    if local_bs * args.world_size != args.batch_size_global:
        raise ValueError("--batch-size-global must be divisible by --world-size.")

    before_step_peak = 0.0
    after_step_peak = 0.0
    iter_times = []

    for i in range(args.warmup_steps + args.measure_steps):
        x = torch.randint(0, args.vocab_size, (local_bs, args.context_length), device=device, dtype=torch.long)
        y = torch.randint(0, args.vocab_size, (local_bs, args.context_length), device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        t0 = default_timer()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1))
        loss.backward()

        torch.cuda.synchronize(device)
        mem_before_step = torch.cuda.memory_allocated(device=device) / (1024 ** 2)
        before_step_peak = max(before_step_peak, mem_before_step)

        optimizer.step()
        torch.cuda.synchronize(device)
        mem_after_step = torch.cuda.memory_allocated(device=device) / (1024 ** 2)
        after_step_peak = max(after_step_peak, mem_after_step)

        if i >= args.warmup_steps:
            iter_times.append((default_timer() - t0) * 1000.0)

    return {
        "mode": mode,
        "param_bytes": param_bytes,
        "optimizer_state_bytes_estimate": state_bytes_est,
        "memory_after_init_mb": init_allocated,
        "memory_after_init_reserved_mb": init_reserved,
        "memory_before_optimizer_step_peak_mb": before_step_peak,
        "memory_after_optimizer_step_peak_mb": after_step_peak,
        "time_per_iteration_ms_mean": statistics.mean(iter_times),
        "time_per_iteration_ms_stdev": statistics.stdev(iter_times) if len(iter_times) > 1 else 0.0,
    }


def _worker(rank: int, args: argparse.Namespace):
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    if not torch.cuda.is_available():
        raise RuntimeError("optimizer_sharding_accounting.py expects CUDA.")
    if args.world_size > torch.cuda.device_count():
        raise RuntimeError(f"Requested world_size={args.world_size}, but only {torch.cuda.device_count()} GPUs available.")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=args.world_size)
    torch.manual_seed(1234)

    non_sharded = _run_mode("non_sharded", args, rank, device)
    torch.cuda.empty_cache()
    sharded = _run_mode("sharded", args, rank, device)

    payload = {"rank": rank, "non_sharded": non_sharded, "sharded": sharded}
    gathered = [None for _ in range(args.world_size)]
    dist.all_gather_object(gathered, payload)

    if rank == 0:
        def _avg(path: str, mode_key: str) -> float:
            vals = [g[mode_key][path] for g in gathered if g is not None]
            return float(statistics.mean(vals))

        summary = {
            "setup": {
                "world_size": args.world_size,
                "model_size": args.model_size,
                "batch_size_global": args.batch_size_global,
                "context_length": args.context_length,
                "vocab_size": args.vocab_size,
                "warmup_steps": args.warmup_steps,
                "measure_steps": args.measure_steps,
            },
            "non_sharded": {
                "memory_after_init_mb": _avg("memory_after_init_mb", "non_sharded"),
                "memory_before_optimizer_step_peak_mb": _avg("memory_before_optimizer_step_peak_mb", "non_sharded"),
                "memory_after_optimizer_step_peak_mb": _avg("memory_after_optimizer_step_peak_mb", "non_sharded"),
                "time_per_iteration_ms_mean": _avg("time_per_iteration_ms_mean", "non_sharded"),
                "optimizer_state_bytes_estimate": _avg("optimizer_state_bytes_estimate", "non_sharded"),
                "param_bytes": _avg("param_bytes", "non_sharded"),
            },
            "sharded": {
                "memory_after_init_mb": _avg("memory_after_init_mb", "sharded"),
                "memory_before_optimizer_step_peak_mb": _avg("memory_before_optimizer_step_peak_mb", "sharded"),
                "memory_after_optimizer_step_peak_mb": _avg("memory_after_optimizer_step_peak_mb", "sharded"),
                "time_per_iteration_ms_mean": _avg("time_per_iteration_ms_mean", "sharded"),
                "optimizer_state_bytes_estimate": _avg("optimizer_state_bytes_estimate", "sharded"),
                "param_bytes": _avg("param_bytes", "sharded"),
            },
            "per_rank": gathered,
        }
        print(json.dumps(summary, indent=2))

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    mp.spawn(_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()

