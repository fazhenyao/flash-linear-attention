#!/usr/bin/env python3

"""Generate NPU-compatible FLA fwd_h and bwd_dhu replay dumps."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump-dir', type=Path, required=True)
    parser.add_argument('--dtype', choices=('float16', 'bfloat16'), default='float16')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--sequence-length', type=int, default=128)
    parser.add_argument('--key-heads', type=int, default=2)
    parser.add_argument('--value-heads', type=int, default=4)
    parser.add_argument('--head-dim', type=int, default=128)
    parser.add_argument('--chunk-size', type=int, choices=(64, 128), default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.value_heads % args.key_heads != 0:
        raise ValueError('value-heads must be an integer multiple of key-heads.')
    if args.head_dim != 128:
        raise ValueError('The Ascend C fwd_h comparison currently requires head-dim=128.')

    os.environ['FLA_DISABLE_BACKEND_DISPATCH'] = '1'
    os.environ['FLA_GDN_DUMP_DIR'] = str(args.dump_dir)
    os.environ['FLA_GDN_DUMP_MAX'] = '1'

    import torch

    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
    from fla.ops.utils.constant import RCP_LN2
    from fla.utils import device

    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    B, T, H, HV, D = (
        args.batch_size,
        args.sequence_length,
        args.key_heads,
        args.value_heads,
        args.head_dim,
    )
    if T % args.chunk_size != 0:
        raise ValueError('sequence-length must be divisible by chunk-size for this dense replay case.')

    q = torch.randn(B, T, H, D, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, H, D, dtype=dtype, device=device) * 0.1
    w = torch.randn(B, T, HV, D, dtype=dtype, device=device) * 0.1
    u = torch.randn(B, T, HV, D, dtype=dtype, device=device) * 0.1
    do = torch.randn(B, T, HV, D, dtype=dtype, device=device) * 0.1
    dv = torch.randn(B, T, HV, D, dtype=dtype, device=device) * 0.1

    gate = -torch.rand(B, T // args.chunk_size, args.chunk_size, HV, device=device)
    g = (gate.cumsum(dim=2) * RCP_LN2).reshape(B, T, HV).float()

    chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=None,
        output_final_state=True,
        chunk_size=args.chunk_size,
    )
    chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        g=g,
        h0=None,
        dht=None,
        scale=D ** -0.5,
        chunk_size=args.chunk_size,
    )
    print(f'Dumps written to {args.dump_dir}')


if __name__ == '__main__':
    main()
