from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import statistics
from timeit import default_timer

import torch

from cs336_basics.model import scaled_dot_product_attention


def synchronize_if_cuda(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def make_causal_mask(seq_len: int, device: str) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    return idx[:, None] >= idx[None, :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch attention for CS336 1.2.1.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dmodels", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--forward-steps", type=int, default=100)
    parser.add_argument("--backward-steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=["uncompiled", "compiled"],
        default=["uncompiled", "compiled"],
        help="Which attention implementations to benchmark.",
    )
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--csv-out", default="profiles/attention/attention_benchmark.csv")
    parser.add_argument("--json-out", default="profiles/attention/attention_benchmark.json")
    return parser.parse_args()


def benchmark_one(
    batch_size: int,
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    device: str,
    warmup_steps: int,
    forward_steps: int,
    backward_steps: int,
    causal: bool,
    implementation: str,
    compile_mode: str,
    compile_backend: str,
) -> dict:
    result = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "dtype": str(dtype).replace("torch.", ""),
        "implementation": implementation,
        "status": "ok",
    }

    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    attn_scores_mb = (batch_size * seq_len * seq_len * bytes_per_elem) / (1024 ** 2)
    result["attention_scores_tensor_mb"] = attn_scores_mb

    mask = make_causal_mask(seq_len, device=device) if causal else None

    try:
        def attn_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, m: torch.Tensor | None):
            return scaled_dot_product_attention(Q=q, K=k, V=v, mask=m)

        if implementation == "compiled":
            attn_fn = torch.compile(attn_fn, mode=compile_mode, backend=compile_backend)

        # Forward benchmark uses no grad.
        q_fwd = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
        k_fwd = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
        v_fwd = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

        for _ in range(warmup_steps):
            with torch.no_grad():
                _ = attn_fn(q_fwd, k_fwd, v_fwd, mask)
            synchronize_if_cuda(device)

        forward_times = []
        for _ in range(forward_steps):
            start = default_timer()
            with torch.no_grad():
                _ = attn_fn(q_fwd, k_fwd, v_fwd, mask)
            synchronize_if_cuda(device)
            forward_times.append(default_timer() - start)

        result["forward_mean_ms"] = 1000.0 * statistics.mean(forward_times)
        result["forward_stdev_ms"] = 1000.0 * statistics.stdev(forward_times) if len(forward_times) > 1 else 0.0

        # Backward benchmark times only loss.backward() and records memory just before backward.
        for _ in range(warmup_steps):
            q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
            o = attn_fn(q, k, v, mask)
            loss = o.sum()
            loss.backward()
            synchronize_if_cuda(device)
            del q, k, v, o, loss

        backward_times = []
        mem_before_backward = []
        for _ in range(backward_steps):
            q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
            o = attn_fn(q, k, v, mask)
            loss = o.sum()

            if device.startswith("cuda"):
                mem_before_backward.append(torch.cuda.memory_allocated() / (1024 ** 2))
            else:
                mem_before_backward.append(0.0)

            start = default_timer()
            loss.backward()
            synchronize_if_cuda(device)
            backward_times.append(default_timer() - start)
            del q, k, v, o, loss

        result["memory_before_backward_mean_mb"] = statistics.mean(mem_before_backward)
        result["memory_before_backward_stdev_mb"] = statistics.stdev(mem_before_backward) if len(mem_before_backward) > 1 else 0.0
        result["backward_mean_ms"] = 1000.0 * statistics.mean(backward_times)
        result["backward_stdev_ms"] = 1000.0 * statistics.stdev(backward_times) if len(backward_times) > 1 else 0.0

    except torch.OutOfMemoryError as e:
        result["status"] = "oom"
        result["error_stage"] = "cuda_oom"
        result["error_message"] = str(e).split("\n")[0]
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            result["status"] = "oom"
            result["error_stage"] = "runtime_oom"
            result["error_message"] = str(e).split("\n")[0]
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        elif "torch.compile" in str(e).lower() or "inductor" in str(e).lower():
            result["status"] = "compile_error"
            result["error_stage"] = "compile_error"
            result["error_message"] = str(e).split("\n")[0]
        else:
            raise

    return result


def main() -> None:
    args = parse_args()
    device = args.device
    dtype = getattr(torch, args.dtype)

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    all_results = []
    for impl, d_model, seq_len in itertools.product(args.implementations, args.dmodels, args.seq_lens):
        row = benchmark_one(
            batch_size=args.batch_size,
            seq_len=seq_len,
            d_model=d_model,
            dtype=dtype,
            device=device,
            warmup_steps=args.warmup_steps,
            forward_steps=args.forward_steps,
            backward_steps=args.backward_steps,
            causal=args.causal,
            implementation=impl,
            compile_mode=args.compile_mode,
            compile_backend=args.compile_backend,
        )
        all_results.append(row)
        print(json.dumps(row))

    pathlib.Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)

    csv_fields = [
        "batch_size",
        "seq_len",
        "d_model",
        "dtype",
        "implementation",
        "status",
        "attention_scores_tensor_mb",
        "memory_before_backward_mean_mb",
        "memory_before_backward_stdev_mb",
        "forward_mean_ms",
        "forward_stdev_ms",
        "backward_mean_ms",
        "backward_stdev_ms",
        "error_stage",
        "error_message",
    ]
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"Wrote CSV: {args.csv_out}")
    print(f"Wrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
