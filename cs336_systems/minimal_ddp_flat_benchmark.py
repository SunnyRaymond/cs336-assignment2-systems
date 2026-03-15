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

from .ddp import allreduce_flattened_parameter_grads, allreduce_individual_parameter_grads, broadcast_module_parameters


MODEL_SPECS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark minimal DDP: individual vs flattened all-reduce.")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--batch-size-global", type=int, default=4)
    parser.add_argument("--model-size", choices=list(MODEL_SPECS.keys()), default="xl")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", default="29540")
    return parser.parse_args()


def _bench_one_mode(
    model: BasicsTransformerLM,
    optimizer: torch.optim.Optimizer,
    local_bs: int,
    args: argparse.Namespace,
    device: torch.device,
    mode: str,
) -> dict:
    if mode not in {"individual", "flattened"}:
        raise ValueError(f"Unknown mode: {mode}")

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
        t_comm0 = default_timer()
        if mode == "individual":
            allreduce_individual_parameter_grads(model, div_world_size=True)
        else:
            allreduce_flattened_parameter_grads(model, div_world_size=True)
        torch.cuda.synchronize(device)
        t_comm = default_timer() - t_comm0

        optimizer.step()
        torch.cuda.synchronize(device)
        t_total = default_timer() - t0
        return t_total, t_comm

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
    return {
        "time_per_step_ms": 1000.0 * total_mean,
        "time_per_step_stdev_ms": 1000.0 * (statistics.stdev(total_times) if len(total_times) > 1 else 0.0),
        "time_in_gradient_communication_ms": 1000.0 * comm_mean,
        "time_in_gradient_communication_stdev_ms": 1000.0 * (statistics.stdev(comm_times) if len(comm_times) > 1 else 0.0),
        "communication_fraction_percent": 100.0 * (comm_mean / total_mean),
    }


def _worker(rank: int, args: argparse.Namespace):
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark expects CUDA (1 node x 2 GPUs).")
    if args.world_size > torch.cuda.device_count():
        raise RuntimeError(f"Requested world_size={args.world_size}, but only {torch.cuda.device_count()} GPUs available.")
    if args.batch_size_global % args.world_size != 0:
        raise ValueError("--batch-size-global must be divisible by --world-size.")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=args.world_size)

    spec = MODEL_SPECS[args.model_size]
    torch.manual_seed(42)
    base_model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=spec["d_model"],
        num_layers=spec["num_layers"],
        num_heads=spec["num_heads"],
        d_ff=spec["d_ff"],
        rope_theta=10_000.0,
    ).to(device)
    broadcast_module_parameters(base_model, src=0)
    base_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}

    local_bs = args.batch_size_global // args.world_size

    results_by_mode: dict[str, dict] = {}
    for mode in ("individual", "flattened"):
        model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=spec["d_model"],
            num_layers=spec["num_layers"],
            num_heads=spec["num_heads"],
            d_ff=spec["d_ff"],
            rope_theta=10_000.0,
        ).to(device)
        model.load_state_dict(base_state, strict=True)
        if args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        results_by_mode[mode] = _bench_one_mode(model, optimizer, local_bs, args, device, mode)

    gathered = [None for _ in range(args.world_size)]
    dist.all_gather_object(gathered, {"rank": rank, "results": results_by_mode})

    if rank == 0:
        def _avg(metric: str, mode: str) -> float:
            vals = [g["results"][mode][metric] for g in gathered if g is not None]
            return float(statistics.mean(vals))

        summary = {
            "setup": {
                "world_size": args.world_size,
                "backend": "nccl",
                "model_size": args.model_size,
                "batch_size_global": args.batch_size_global,
                "context_length": args.context_length,
                "vocab_size": args.vocab_size,
                "warmup_steps": args.warmup_steps,
                "measure_steps": args.measure_steps,
                "optimizer": args.optimizer,
            },
            "individual_allreduce": {
                "time_per_step_ms": _avg("time_per_step_ms", "individual"),
                "time_in_gradient_communication_ms": _avg("time_in_gradient_communication_ms", "individual"),
                "communication_fraction_percent": _avg("communication_fraction_percent", "individual"),
            },
            "flattened_allreduce": {
                "time_per_step_ms": _avg("time_per_step_ms", "flattened"),
                "time_in_gradient_communication_ms": _avg("time_in_gradient_communication_ms", "flattened"),
                "communication_fraction_percent": _avg("communication_fraction_percent", "flattened"),
            },
        }
        print(json.dumps(summary, indent=2))

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    mp.spawn(_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
