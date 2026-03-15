from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency at runtime
    triton = None
    tl = None


def _flash_forward_pytorch_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    q_tile_size: int = 64,
    k_tile_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FlashAttention-style tiled forward pass in pure PyTorch.
    Returns:
      O: [B, Nq, D]
      L: [B, Nq] where L = logsumexp(S, dim=-1)
    """
    batch_size, n_queries, d = q.shape
    n_keys = k.shape[1]
    scale = 1.0 / math.sqrt(d)

    q32 = q.float()
    k32 = k.float()
    v32 = v.float()

    m = torch.full((batch_size, n_queries), float("-inf"), device=q.device, dtype=torch.float32)
    l = torch.zeros((batch_size, n_queries), device=q.device, dtype=torch.float32)
    acc = torch.zeros((batch_size, n_queries, d), device=q.device, dtype=torch.float32)

    q_idx = torch.arange(n_queries, device=q.device)
    for k_start in range(0, n_keys, k_tile_size):
        k_end = min(k_start + k_tile_size, n_keys)
        k_tile = k32[:, k_start:k_end, :]
        v_tile = v32[:, k_start:k_end, :]

        s = torch.matmul(q32, k_tile.transpose(-1, -2)) * scale
        if is_causal:
            k_idx = torch.arange(k_start, k_end, device=q.device)
            mask = q_idx[:, None] >= k_idx[None, :]
            # Match assignment guidance for masking in this section.
            s = torch.where(mask[None, :, :], s, torch.tensor(-1e6, device=q.device, dtype=s.dtype))

        m_tile = s.max(dim=-1).values
        m_new = torch.maximum(m, m_tile)

        alpha = torch.exp(m - m_new)
        p = torch.exp(s - m_new.unsqueeze(-1))
        l_new = alpha * l + p.sum(dim=-1)

        acc = (alpha * l).unsqueeze(-1) * acc + torch.matmul(p, v_tile)

        m = m_new
        l = l_new

    lse = m + torch.log(l)
    o = acc / l.unsqueeze(-1)
    return o.to(dtype=q.dtype), lse


class FlashAttention2PyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        o, l = _flash_forward_pytorch_tiled(q, k, v, is_causal=is_causal)
        ctx.save_for_backward(q, k, v, o, l)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        raise NotImplementedError("FlashAttention2PyTorch backward is implemented in the next assignment step.")


if triton is not None:

    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        is_causal: tl.constexpr,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        q = tl.load(Q_block_ptr)
        m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

        for key_tile_index in range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
            K_block_ptr = tl.make_block_ptr(
                K_ptr + batch_index * stride_kb,
                shape=(N_KEYS, D),
                strides=(stride_kk, stride_kd),
                offsets=(key_tile_index * K_TILE_SIZE, 0),
                block_shape=(K_TILE_SIZE, D),
                order=(1, 0),
            )
            V_block_ptr = tl.make_block_ptr(
                V_ptr + batch_index * stride_vb,
                shape=(N_KEYS, D),
                strides=(stride_vk, stride_vd),
                offsets=(key_tile_index * K_TILE_SIZE, 0),
                block_shape=(K_TILE_SIZE, D),
                order=(1, 0),
            )

            k = tl.load(K_block_ptr)
            v = tl.load(V_block_ptr)
            s = tl.dot(q, tl.trans(k)) * scale

            if is_causal:
                q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                k_idx = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
                causal_mask = q_idx[:, None] >= k_idx[None, :]
                s = tl.where(causal_mask, s, -1e6)

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(v.dtype), v, acc=acc)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        o = acc / l_i[:, None]
        lse = m_i + tl.log(l_i)

        tl.store(O_block_ptr, o.to(O_block_ptr.type.element_ty))

        l_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        L_ptr_batch = L_ptr + batch_index * stride_lb
        tl.store(L_ptr_batch + l_offsets * stride_lq, lse)


def _flash_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    q_tile_size: int = 64,
    k_tile_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is not available in this environment.")
    if not q.is_cuda:
        raise RuntimeError("Triton FlashAttention forward requires CUDA tensors.")

    batch_size, n_queries, d = q.shape
    n_keys = k.shape[1]
    scale = 1.0 / math.sqrt(d)

    o = torch.empty_like(q)
    l = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(n_queries, q_tile_size), batch_size)
    flash_fwd_kernel[grid](
        q, k, v,
        o, l,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        l.stride(0), l.stride(1),
        n_queries, n_keys,
        scale,
        is_causal=is_causal,
        D=d,
        Q_TILE_SIZE=q_tile_size,
        K_TILE_SIZE=k_tile_size,
    )
    return o, l


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        o, l = _flash_forward_triton(q, k, v, is_causal=is_causal)
        ctx.save_for_backward(q, k, v, o, l)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        raise NotImplementedError("FlashAttention2Triton backward is implemented in the next assignment step.")
