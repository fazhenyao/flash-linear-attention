# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CPU reference implementation for the chunked GDN DHU backward kernel."""

from __future__ import annotations

import torch


def chunk_gated_delta_rule_bwd_dhu_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Compute the DHU backward recurrence in float32 on CPU.

    This follows ``chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`` and has
    the same tensor contract as ``chunk_gated_delta_rule_bwd_dhu``. Inputs may
    live on any device; the computation is accumulated in CPU float32, while
    outputs use the same dtypes as the GPU kernel (``dh``/``dv2`` use the
    requested output dtype and ``dh0`` is always float32).
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    B, T, H, K = q.shape
    HV, V = do.shape[2], do.shape[3]
    if HV % H:
        raise ValueError(f"value heads ({HV}) must be a multiple of key heads ({H})")
    if cu_seqlens is None:
        lengths = [T] * B
        sequence_ids = list(range(B))
        state_shape = (V, K) if state_v_first else (K, V)
        output_shape = (B, (T + chunk_size - 1) // chunk_size, HV, *state_shape)
    else:
        cu = cu_seqlens.detach().cpu().tolist()
        lengths = [int(e - s) for s, e in zip(cu[:-1], cu[1:])]
        sequence_ids = [0] * len(lengths)
        state_shape = (V, K) if state_v_first else (K, V)
        output_shape = (1, sum((length + chunk_size - 1) // chunk_size for length in lengths), HV, *state_shape)

    q_f = q.detach().to(device="cpu", dtype=torch.float32)
    k_f = k.detach().to(device="cpu", dtype=torch.float32)
    w_f = w.detach().to(device="cpu", dtype=torch.float32)
    do_f = do.detach().to(device="cpu", dtype=torch.float32)
    dv_f = dv.detach().to(device="cpu", dtype=torch.float32)
    g_f = g.detach().to(device="cpu", dtype=torch.float32) if g is not None else None
    gk_f = gk.detach().to(device="cpu", dtype=torch.float32) if gk is not None else None
    dht_f = dht.detach().to(device="cpu", dtype=torch.float32) if dht is not None else None

    dh_cpu = torch.zeros(output_shape, dtype=torch.float32)
    dv2_cpu = torch.zeros_like(dv_f)
    dh0_cpu = torch.zeros_like(h0.detach().to(device="cpu", dtype=torch.float32)) if h0 is not None else None

    cu = cu_seqlens.detach().cpu().tolist() if cu_seqlens is not None else None
    chunk_out = 0
    for n, length in enumerate(lengths):
        start = 0 if cu is None else int(cu[n])
        end = start + length
        n_chunks = (length + chunk_size - 1) // chunk_size
        q_n = q_f[sequence_ids[n], start:end].repeat_interleave(HV // H, dim=1)
        k_n = k_f[sequence_ids[n], start:end].repeat_interleave(HV // H, dim=1)
        w_n, do_n, dv_n = w_f[sequence_ids[n], start:end], do_f[sequence_ids[n], start:end], dv_f[sequence_ids[n], start:end]
        g_n = g_f[sequence_ids[n], start:end] if g_f is not None else None
        gk_n = gk_f[sequence_ids[n], start:end] if gk_f is not None else None
        state = torch.zeros(HV, K, V, dtype=torch.float32)
        if dht_f is not None:
            state = dht_f[n].transpose(-1, -2).contiguous() if state_v_first else dht_f[n].clone()

        for chunk in range(n_chunks - 1, -1, -1):
            s, e = chunk * chunk_size, min((chunk + 1) * chunk_size, length)
            out_index = chunk if cu is None else chunk_out + chunk
            dh_cpu[sequence_ids[n] if cu is None else 0, out_index] = (
                state.transpose(-1, -2) if state_v_first else state
            )
            k_c, q_c = k_n[s:e], q_n[s:e]
            w_c, do_c, dv_c = w_n[s:e], do_n[s:e], dv_n[s:e]
            kdh = torch.einsum("thk,hkv->thv", k_c, state)
            if g_n is not None:
                g_c, g_last = g_n[s:e], g_n[e - 1]
                b_dv = kdh * torch.exp2(g_last[None, :, None] - g_c[:, :, None]) + dv_c
                q_gated = q_c * torch.exp2(g_c)[:, :, None]
                state = state * torch.exp2(g_last)[:, None, None]
            else:
                b_dv = kdh + dv_c
                q_gated = q_c
            if gk_n is not None:
                state = state * torch.exp2(gk_n[e - 1])[:, :, None]
            dv2_cpu[sequence_ids[n], start + s:start + e] = b_dv
            state = state + torch.einsum("thk,thv->hkv", q_gated, do_c) * scale
            state = state - torch.einsum("thk,thv->hkv", w_c, b_dv)
        if dh0_cpu is not None:
            dh0_cpu[n] = state.transpose(-1, -2) if state_v_first else state
        chunk_out += n_chunks

    dh_out = dh_cpu.to(dtype=output_dtype or q.dtype, device=q.device)
    dh0_out = dh0_cpu.to(dtype=torch.float32, device=h0.device) if dh0_cpu is not None else None
    dv2_out = dv2_cpu.to(dtype=output_dtype or dv.dtype, device=dv.device)
    return dh_out, dh0_out, dv2_out


__all__ = ["chunk_gated_delta_rule_bwd_dhu_cpu"]
