from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from contextlib import nullcontext
from timeit import default_timer

import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import softmax
from einops import einsum


MODEL_SPECS: dict[str, dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7b": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def synchronize_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end benchmarking for CS336 Assignment 2.")
    parser.add_argument("--size", choices=sorted(MODEL_SPECS.keys()), default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=["forward", "forward-backward", "train_step"],
        default="forward-backward",
        help="Measure forward, forward+backward, or complete train step (forward+backward+optimizer).",
    )
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "mean"],
        default="cross_entropy",
        help="Loss used when mode=forward-backward.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--compile-model", action="store_true", help="Compile the full model with torch.compile.")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument(
        "--annotate-attention",
        action="store_true",
        help="Wrap scaled_dot_product_attention with NVTX ranges.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="Model parameter dtype.",
    )
    parser.add_argument(
        "--amp",
        choices=["none", "float16", "bfloat16"],
        default="none",
        help="Autocast dtype for mixed precision compute.",
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Record a PyTorch CUDA memory snapshot over measured steps.",
    )
    parser.add_argument(
        "--memory-snapshot-prefix",
        default="profiles/memory/memory_snapshot",
        help="Prefix for torch.cuda.memory snapshot output file.",
    )
    parser.add_argument("--memory-max-entries", type=int, default=1_000_000)
    return parser.parse_args()


def make_nvtx_annotated_attention():
    def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
        with nvtx.range("scaled dot product attention"):
            d_k = K.shape[-1]
            with nvtx.range("computing attention scores"):
                attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / (d_k ** 0.5)
                if mask is not None:
                    attention_scores = torch.where(mask, attention_scores, float("-inf"))
            with nvtx.range("computing softmax"):
                attention_weights = softmax(attention_scores, dim=-1)
            with nvtx.range("final matmul"):
                return einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")

    return annotated_scaled_dot_product_attention


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    spec = MODEL_SPECS[args.size]
    device = args.device
    dtype = getattr(torch, args.dtype)
    amp_dtype = None if args.amp == "none" else getattr(torch, args.amp)
    if amp_dtype is not None and not device.startswith("cuda"):
        raise ValueError("--amp requires a CUDA device.")

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=spec["d_model"],
        num_layers=spec["num_layers"],
        num_heads=spec["num_heads"],
        d_ff=spec["d_ff"],
        rope_theta=args.rope_theta,
    ).to(device=device, dtype=dtype)
    if args.compile_model:
        model = torch.compile(model, mode=args.compile_mode, backend=args.compile_backend)
    model.train()

    if args.annotate_attention:
        basics_model.scaled_dot_product_attention = make_nvtx_annotated_attention()

    x = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
        dtype=torch.long,
    )

    optimizer = None
    if args.mode == "train_step":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def run_step(profiled: bool) -> tuple[float, float | None, float | None]:
        step_range = "measured_step" if profiled else "warmup_step"
        with nvtx.range(step_range):
            with nvtx.range("step_start"):
                model.zero_grad(set_to_none=True)
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

            with nvtx.range("forward"):
                forward_start = default_timer()
                if amp_dtype is None:
                    autocast_ctx = nullcontext()
                else:
                    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
                with autocast_ctx:
                    logits = model(x)
                synchronize_if_needed(device)
                forward_time = default_timer() - forward_start

            if args.mode == "forward":
                return forward_time, None, None

            with nvtx.range("loss"):
                if args.loss == "cross_entropy":
                    loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), targets.reshape(-1))
                else:
                    loss = logits.float().mean()

            with nvtx.range("backward"):
                backward_start = default_timer()
                loss.backward()
                synchronize_if_needed(device)
                backward_time = default_timer() - backward_start

            optimizer_time = None
            if optimizer is not None:
                with nvtx.range("optimizer_step"):
                    optimizer_start = default_timer()
                    optimizer.step()
                    synchronize_if_needed(device)
                    optimizer_time = default_timer() - optimizer_start

            return forward_time, backward_time, optimizer_time

    # Warm-up (not measured).
    for _ in range(args.warmup_steps):
        run_step(profiled=False)

    memory_snapshot_path = None
    peak_allocated_mb = None
    peak_reserved_mb = None
    if args.memory_profile:
        if not device.startswith("cuda"):
            raise ValueError("--memory-profile requires CUDA.")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory._record_memory_history(max_entries=args.memory_max_entries)

    forward_times: list[float] = []
    backward_times: list[float] = []
    optimizer_times: list[float] = []
    total_times: list[float] = []

    for _ in range(args.measure_steps):
        fwd_t, bwd_t, opt_t = run_step(profiled=True)
        forward_times.append(fwd_t)
        if bwd_t is not None:
            backward_times.append(bwd_t)
        if opt_t is not None:
            optimizer_times.append(opt_t)
        total = fwd_t
        if bwd_t is not None:
            total += bwd_t
        if opt_t is not None:
            total += opt_t
        total_times.append(total)

    if args.memory_profile:
        snapshot_prefix_path = pathlib.Path(args.memory_snapshot_prefix)
        snapshot_prefix_path.parent.mkdir(parents=True, exist_ok=True)
        memory_snapshot_path = (
            f"{args.memory_snapshot_prefix}_{args.size}_ctx{args.context_length}_{args.mode}_amp-{args.amp}.pickle"
        )
        torch.cuda.memory._dump_snapshot(memory_snapshot_path)
        torch.cuda.memory._record_memory_history(enabled=None)
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

    result = {
        "config": {
            "size": args.size,
            "spec": spec,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "vocab_size": args.vocab_size,
            "mode": args.mode,
            "loss": args.loss,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "device": device,
            "dtype": args.dtype,
            "amp": args.amp,
            "annotate_attention": args.annotate_attention,
            "compile_model": args.compile_model,
            "compile_mode": args.compile_mode,
            "compile_backend": args.compile_backend,
            "memory_profile": args.memory_profile,
        },
        "forward_ms": {
            "mean": 1000.0 * statistics.mean(forward_times),
            "stdev": 1000.0 * statistics.stdev(forward_times) if len(forward_times) > 1 else 0.0,
            "all": [1000.0 * t for t in forward_times],
        },
        "total_step_ms": {
            "mean": 1000.0 * statistics.mean(total_times),
            "stdev": 1000.0 * statistics.stdev(total_times) if len(total_times) > 1 else 0.0,
            "all": [1000.0 * t for t in total_times],
        },
    }
    if backward_times:
        result["backward_ms"] = {
            "mean": 1000.0 * statistics.mean(backward_times),
            "stdev": 1000.0 * statistics.stdev(backward_times) if len(backward_times) > 1 else 0.0,
            "all": [1000.0 * t for t in backward_times],
        }
    if optimizer_times:
        result["optimizer_step_ms"] = {
            "mean": 1000.0 * statistics.mean(optimizer_times),
            "stdev": 1000.0 * statistics.stdev(optimizer_times) if len(optimizer_times) > 1 else 0.0,
            "all": [1000.0 * t for t in optimizer_times],
        }
    if args.memory_profile:
        result["memory"] = {
            "snapshot_path": memory_snapshot_path,
            "peak_allocated_mb": peak_allocated_mb,
            "peak_reserved_mb": peak_reserved_mb,
        }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
