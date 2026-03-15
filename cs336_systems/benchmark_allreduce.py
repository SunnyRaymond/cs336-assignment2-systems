from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@dataclass
class BenchmarkResult:
    backend: str
    device_type: str
    world_size: int
    size_mb: int
    numel: int
    dtype: str
    warmup_iters: int
    measure_iters: int
    rank0_mean_ms: float
    rank0_stdev_ms: float
    rank0_min_ms: float
    rank0_max_ms: float
    rank0_bandwidth_gbps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-node all-reduce benchmark.")
    parser.add_argument("--process-counts", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--sizes-mb", type=int, nargs="+", default=[1, 10, 100, 1024])
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["gloo_cpu", "nccl_gpu"],
        default=["gloo_cpu", "nccl_gpu"],
    )
    parser.add_argument("--dtype", choices=["float32"], default="float32")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=20)
    parser.add_argument("--adaptive-iters", action="store_true")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port-base", type=int, default=29600)
    parser.add_argument("--csv-out", default="profiles/allreduce/allreduce_benchmark.csv")
    parser.add_argument("--json-out", default="profiles/allreduce/allreduce_benchmark.json")
    return parser.parse_args()


def _iters_for_size(size_mb: int, warmup: int, measure: int, adaptive: bool) -> tuple[int, int]:
    if not adaptive:
        return warmup, measure
    if size_mb >= 1024:
        return 1, 3
    if size_mb >= 100:
        return 2, 5
    return warmup, measure


def _worker(
    rank: int,
    world_size: int,
    backend: str,
    device_type: str,
    size_mb: int,
    warmup_iters: int,
    measure_iters: int,
    master_addr: str,
    master_port: int,
    out_queue,
) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL benchmark requested but CUDA is unavailable.")
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    bytes_total = size_mb * 1024 * 1024
    numel = bytes_total // 4  # float32
    tensor = torch.ones(numel, device=device, dtype=torch.float32)

    for _ in range(warmup_iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if device_type == "cuda":
            torch.cuda.synchronize(device)

    dist.barrier()
    timings = []
    for _ in range(measure_iters):
        if device_type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if device_type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings.append(elapsed_ms)

    gathered: list[list[float] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, timings)

    if rank == 0:
        rank0_times = gathered[0]
        assert rank0_times is not None
        mean_ms = statistics.mean(rank0_times)
        stdev_ms = statistics.stdev(rank0_times) if len(rank0_times) > 1 else 0.0
        min_ms = min(rank0_times)
        max_ms = max(rank0_times)

        # Effective bandwidth approximation for ring all-reduce:
        # 2 * (N-1)/N * bytes / time.
        bytes_per_iter = 2.0 * (world_size - 1) / world_size * bytes_total
        bw_gbps = (bytes_per_iter / (mean_ms / 1000.0)) / 1e9

        result = BenchmarkResult(
            backend=backend,
            device_type=device_type,
            world_size=world_size,
            size_mb=size_mb,
            numel=numel,
            dtype="float32",
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            rank0_mean_ms=mean_ms,
            rank0_stdev_ms=stdev_ms,
            rank0_min_ms=min_ms,
            rank0_max_ms=max_ms,
            rank0_bandwidth_gbps=bw_gbps,
        )
        print(json.dumps({"result": asdict(result)}))
        out_queue.put(asdict(result))

    dist.barrier()
    dist.destroy_process_group()


def run_one_config(
    world_size: int,
    backend: str,
    device_type: str,
    size_mb: int,
    warmup_iters: int,
    measure_iters: int,
    master_addr: str,
    master_port: int,
) -> BenchmarkResult:
    out_queue = mp.SimpleQueue()
    mp.spawn(
        _worker,
        args=(
            world_size,
            backend,
            device_type,
            size_mb,
            warmup_iters,
            measure_iters,
            master_addr,
            master_port,
            out_queue,
        ),
        nprocs=world_size,
        join=True,
    )
    payload = out_queue.get()
    return BenchmarkResult(**payload)


def _backend_spec(name: str) -> tuple[str, str]:
    if name == "gloo_cpu":
        return "gloo", "cpu"
    if name == "nccl_gpu":
        return "nccl", "cuda"
    raise ValueError(f"Unknown backend spec: {name}")


def main() -> None:
    args = parse_args()

    if any(p not in (2, 4) for p in args.process_counts):
        raise ValueError("This script is configured for process counts 2 or 4 only.")

    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkResult] = []

    run_index = 0
    for backend_name in args.backends:
        backend, device_type = _backend_spec(backend_name)
        if device_type == "cuda" and not torch.cuda.is_available():
            print(f"Skipping {backend_name}: CUDA unavailable.")
            continue
        if device_type == "cuda":
            gpu_count = torch.cuda.device_count()
            if max(args.process_counts) > gpu_count:
                print(
                    f"Skipping {backend_name}: requested max world_size={max(args.process_counts)} "
                    f"but only {gpu_count} GPUs visible."
                )
                continue

        for world_size in args.process_counts:
            if device_type == "cuda" and world_size > torch.cuda.device_count():
                print(f"Skipping {backend_name}, world_size={world_size}: not enough GPUs.")
                continue
            for size_mb in args.sizes_mb:
                warmup, measure = _iters_for_size(size_mb, args.warmup_iters, args.measure_iters, args.adaptive_iters)
                master_port = args.master_port_base + run_index
                run_index += 1
                print(
                    f"Running backend={backend} device={device_type} "
                    f"world_size={world_size} size_mb={size_mb} warmup={warmup} measure={measure}"
                )

                result = run_one_config(
                    world_size=world_size,
                    backend=backend,
                    device_type=device_type,
                    size_mb=size_mb,
                    warmup_iters=warmup,
                    measure_iters=measure,
                    master_addr=args.master_addr,
                    master_port=master_port,
                )
                results.append(result)

    if not results:
        raise RuntimeError("No benchmark results were produced.")
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"Wrote CSV: {args.csv_out}")
    print(f"Wrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
