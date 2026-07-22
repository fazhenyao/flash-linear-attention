#!/usr/bin/env python3

"""Generate NPU-compatible FLA bwd_dhu replay dumps."""

from __future__ import annotations

import argparse
import itertools
import os
import re
import subprocess
import sys
from pathlib import Path


def _table_sequence_count(description: str) -> int | None:
    """Extract the sequence count mentioned by the NPU table description."""
    match = re.search(r'长度\s*(\d+)', description)
    if match:
        return int(match.group(1))
    match = re.search(r'cu_seqlen(?:s)?\s*长度\s*(\d+)', description)
    if match:
        return int(match.group(1))
    return None


def _chunk_local_cumsum(values, chunk_size: int):
    """Cumulative gate values reset at every chunk, including a partial tail chunk."""
    import torch

    result = torch.empty_like(values)
    for start in range(0, values.shape[1], chunk_size):
        end = min(start + chunk_size, values.shape[1])
        result[:, start:end] = values[:, start:end].cumsum(dim=1)
    return result


def _segmented_chunk_cumsum(values, cu_seqlens, chunk_size):
    result = values.new_empty(values.shape)
    for bos, eos in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        result[:, bos:eos] = _chunk_local_cumsum(values[:, bos:eos], chunk_size)
    return result


def _expand_and_gate_qk(q, k, gk, value_heads, chunk_size, cu_seqlens):
    group_size = value_heads // q.shape[2]
    import torch

    qg = q.repeat_interleave(group_size, dim=2) * torch.exp2(gk).to(q.dtype)
    kg = k.repeat_interleave(group_size, dim=2).clone()
    sequences = [(0, q.shape[1])] if cu_seqlens is None else zip(
        cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()
    )
    for bos, eos in sequences:
        for start in range(bos, eos, chunk_size):
            end = min(start + chunk_size, eos)
            kg[:, start:end] *= torch.exp2(gk[:, end - 1:end] - gk[:, start:end]).to(k.dtype)
    return qg, kg


def _load_gdn_table(table_path: Path, base_seed: int, dtype: str, variants: str) -> list[dict]:
    try:
        import openpyxl
    except ImportError as error:
        raise RuntimeError(
            'The gdn-table profile requires openpyxl. Install it with `pip install openpyxl`.'
        ) from error

    worksheet = openpyxl.load_workbook(table_path, data_only=True, read_only=True).active
    rows = list(worksheet.iter_rows(values_only=True))
    if not rows:
        raise ValueError(f'GDN case table is empty: {table_path}')
    columns = {str(value).strip(): index for index, value in enumerate(rows[0]) if value is not None}
    required = ('新用例 ID', 'B', 'V_H', 'K_H', 'T', 'Vdim', 'Kdim', 'chunk_size', '描述')
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f'GDN case table is missing columns {missing}: {table_path}')

    # Rotate optional arguments so the table sweep exercises every new bwd_dhu input.
    coverage_variants = [
        ('scalar', 'none'), ('scalar', 'h0'), ('scalar', 'dht'), ('scalar', 'both'),
        ('gk', 'none'), ('gk', 'h0'), ('gk', 'dht'), ('gk', 'both'),
        ('both', 'none'), ('both', 'h0'), ('both', 'dht'), ('both', 'both'),
    ]
    variants_to_use = coverage_variants if variants == 'all' else coverage_variants
    cases = []
    dtypes = ('float16', 'bfloat16') if dtype == 'both' else (dtype,)
    for row_index, row in enumerate(rows[1:]):
        if not row or row[columns['新用例 ID']] is None:
            continue
        name = str(row[columns['新用例 ID']])
        batch = int(row[columns['B']])
        value_heads = int(row[columns['V_H']])
        key_heads = int(row[columns['K_H']])
        sequence_length = int(row[columns['T']])
        value_dim = int(row[columns['Vdim']])
        head_dim = int(row[columns['Kdim']])
        chunk_size = int(row[columns['chunk_size']])
        description = str(row[columns['描述']] or '')
        is_varlen = '变长' in description
        sequence_count = _table_sequence_count(description) if is_varlen else None
        if is_varlen and (sequence_count is None or sequence_count < 1):
            raise ValueError(f'Cannot infer cu_seqlens count for {name}: {description}')

        selected = variants_to_use if variants == 'all' else [coverage_variants[row_index % len(coverage_variants)]]
        for dtype_index, selected_dtype in enumerate(dtypes):
            for variant_index, (gate_mode, state_mode) in enumerate(selected):
                cases.append({
                    'table_id': name,
                    'description': description,
                    'dtype': selected_dtype,
                    'batch_size': batch,
                    'sequence_length': sequence_length,
                    'key_heads': key_heads,
                    'value_heads': value_heads,
                    'head_dim': head_dim,
                    'value_dim': value_dim,
                    'chunk_size': chunk_size,
                    'gate_mode': gate_mode,
                    'state_mode': state_mode,
                    'initial_state': state_mode in ('h0', 'both'),
                    'final_state_gradient': state_mode in ('dht', 'both'),
                    'varlen': is_varlen,
                    'sequence_count': sequence_count,
                    'seed': base_seed + row_index * 1000 + dtype_index * 100 + variant_index,
                })
    return cases


def _matrix_cases(profile: str, base_seed: int, *, table_path: Path, dtype: str, table_variants: str) -> list[dict]:
    if profile == 'gdn-table':
        return _load_gdn_table(table_path, base_seed, dtype, table_variants)
    if profile == 'gdn-table-smoke':
        cases = _load_gdn_table(table_path, base_seed, dtype, 'all')
        selected_variants = {
            'BSND_noGVA_V128_13': {('scalar', 'none')},
            'BSND_GVA_V256_28': {('gk', 'h0')},
            'BSND_GVA_V128_26': {('gk', 'dht')},
            'BSND_GVA_V256_33': {('both', 'both')},
        }
        return [
            case for case in cases
            if (case['gate_mode'], case['state_mode']) in selected_variants.get(case['table_id'], set())
        ]
    if profile == 'smoke':
        configurations = [
            ('float16', 64, 64, 'scalar', 'none', 2, 2, 128),
            ('float16', 64, 128, 'scalar', 'h0', 2, 4, 128),
            ('float16', 64, 256, 'gk', 'dht', 2, 4, 256),
            ('float16', 64, 512, 'both', 'both', 2, 4, 128),
            ('bfloat16', 128, 128, 'scalar', 'none', 2, 2, 256),
            ('bfloat16', 128, 256, 'gk', 'h0', 2, 4, 128),
            ('bfloat16', 128, 512, 'both', 'both', 2, 4, 256),
        ]
    else:
        base_configurations = itertools.product(
            ('float16', 'bfloat16'),
            (64, 128),
            (1, 2, 4, 8),
            ('scalar', 'gk', 'both'),
            ('none', 'h0', 'dht', 'both'),
        )
        shapes = ((2, 2, 128), (2, 4, 128), (2, 4, 256))
        configurations = [
            (dtype, chunk_size, chunk_size * chunk_count, gate_mode, state_mode, *shapes[index % len(shapes)])
            for index, (dtype, chunk_size, chunk_count, gate_mode, state_mode) in enumerate(base_configurations)
        ]
    return [
        {
            'dtype': dtype,
            'chunk_size': chunk_size,
            'sequence_length': sequence_length,
            'gate_mode': gate_mode,
            'key_heads': key_heads,
            'value_heads': value_heads,
            'value_dim': value_dim,
            'initial_state': state_mode in ('h0', 'both'),
            'final_state_gradient': state_mode in ('dht', 'both'),
            'state_mode': state_mode,
            'seed': base_seed + index,
            'batch_size': None,
            'varlen': False,
            'sequence_count': None,
        }
        for index, (
            dtype,
            chunk_size,
            sequence_length,
            gate_mode,
            state_mode,
            key_heads,
            value_heads,
            value_dim,
        ) in enumerate(configurations)
    ]


def _run_matrix(args) -> None:
    cases = _matrix_cases(
        args.matrix_profile,
        args.seed,
        table_path=args.gdn_table,
        dtype=args.dtype,
        table_variants=args.table_variants,
    )
    args.dump_dir = args.dump_dir.expanduser().resolve()
    print(f'Generating {len(cases)} accuracy cases under {args.dump_dir}')
    for index, case in enumerate(cases):
        table_prefix = f'{case["table_id"]}_' if case.get('table_id') else ''
        case_name = (
            f'{index:03d}_{table_prefix}{case["dtype"]}_bt{case["chunk_size"]}_t{case["sequence_length"]}_'
            f'h{case["key_heads"]}_hv{case["value_heads"]}_v{case["value_dim"]}_'
            f'{case["gate_mode"]}_{case["state_mode"]}'
        )
        case_dir = args.dump_dir / case_name
        estimated_bytes = (
            (case.get('batch_size') or args.batch_size)
            * case['sequence_length']
            * case['value_heads']
            * (case['head_dim'] + 2 * case['value_dim'])
            * 2
        )
        estimated_gib = estimated_bytes / 1024 ** 3
        if args.max_case_gib is not None and estimated_gib > args.max_case_gib:
            print(f'[{index + 1}/{len(cases)}] SKIP {case_name}: estimated base tensors {estimated_gib:.2f} GiB')
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            '--dump-dir',
            str(case_dir),
            '--dtype',
            case['dtype'],
            '--batch-size',
            str(case.get('batch_size') or args.batch_size),
            '--sequence-length',
            str(case['sequence_length']),
            '--key-heads',
            str(case['key_heads']),
            '--value-heads',
            str(case['value_heads']),
            '--head-dim',
            str(case.get('head_dim', args.head_dim)),
            '--value-dim',
            str(case['value_dim']),
            '--chunk-size',
            str(case['chunk_size']),
            '--gate-mode',
            case['gate_mode'],
            '--seed',
            str(case['seed']),
        ]
        if case['initial_state']:
            command.append('--initial-state')
        if case['final_state_gradient']:
            command.append('--final-state-gradient')
        if case.get('varlen'):
            command.append('--varlen')
            command.extend(['--varlen-sequences', str(case['sequence_count'])])
        if case.get('table_id'):
            command.extend(['--case-name', case['table_id']])
            if case.get('description'):
                command.extend(['--case-description', case['description']])
        print(f'[{index + 1}/{len(cases)}] {case_name}')
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump-dir', type=Path, required=True)
    parser.add_argument('--dtype', choices=('float16', 'bfloat16', 'both'), default='float16')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--sequence-length', type=int, default=128)
    parser.add_argument('--key-heads', type=int, default=2)
    parser.add_argument('--value-heads', type=int, default=4)
    parser.add_argument('--head-dim', type=int, default=128)
    parser.add_argument('--value-dim', type=int)
    parser.add_argument('--chunk-size', type=int, choices=(64, 128), default=64)
    parser.add_argument('--gate-mode', choices=('scalar', 'gk', 'both'), default='scalar')
    parser.add_argument('--initial-state', action='store_true')
    parser.add_argument('--final-state-gradient', action='store_true')
    parser.add_argument(
        '--matrix-profile',
        choices=('off', 'smoke', 'full', 'gdn-table', 'gdn-table-smoke'),
        default='off',
    )
    parser.add_argument(
        '--gdn-table',
        type=Path,
        default=Path('/data/fazhenyao/flash-linear-attention-npu/GDN泛化用例表.xlsx'),
        help='NPU GDN case table used by the gdn-table profiles.',
    )
    parser.add_argument(
        '--table-variants', choices=('coverage', 'all'), default='coverage',
        help='coverage assigns one gate/state variant per row; all runs all 12 variants per row.',
    )
    parser.add_argument(
        '--max-case-gib', type=float,
        help='Skip matrix cases whose q/k/w/do/dv tensors alone exceed this approximate GiB limit.',
    )
    parser.add_argument('--case-name', help=argparse.SUPPRESS)
    parser.add_argument('--case-description', help=argparse.SUPPRESS)
    parser.add_argument('--varlen', action='store_true', help='Use packed variable-length sequence mode.')
    parser.add_argument('--varlen-sequences', type=int, help='Number of packed sequences for generated varlen input.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.matrix_profile != 'off':
        _run_matrix(args)
        return

    if args.dtype == 'both':
        raise ValueError('--dtype both is only valid with a gdn-table matrix profile.')

    if args.value_heads % args.key_heads != 0:
        raise ValueError('value-heads must be an integer multiple of key-heads.')
    if args.head_dim != 128:
        raise ValueError('The Ascend C bwd_dhu comparison currently requires head-dim=128.')
    if args.value_dim is None:
        args.value_dim = args.head_dim
    if args.value_dim not in (128, 256):
        raise ValueError('The Ascend C comparison currently requires value-dim 128 or 256.')

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    args.dump_dir = args.dump_dir.expanduser().resolve()
    os.environ['FLA_DISABLE_BACKEND_DISPATCH'] = '1'
    os.environ['FLA_GDN_DUMP_DIR'] = str(args.dump_dir)
    os.environ['FLA_GDN_DUMP_MAX'] = '1'
    if args.case_name:
        os.environ['FLA_GDN_CASE_NAME'] = args.case_name
    if args.case_description:
        os.environ['FLA_GDN_CASE_DESCRIPTION'] = args.case_description

    import torch

    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
    from fla.ops.utils.constant import RCP_LN2
    from fla.utils import device

    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    B, T, H, HV, D, DV = (
        args.batch_size,
        args.sequence_length,
        args.key_heads,
        args.value_heads,
        args.head_dim,
        args.value_dim,
    )
    q = torch.randn(B, T, H, D, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, H, D, dtype=dtype, device=device) * 0.1
    w = torch.randn(B, T, HV, D, dtype=dtype, device=device) * 0.1
    do = torch.randn(B, T, HV, DV, dtype=dtype, device=device) * 0.1
    dv = torch.randn(B, T, HV, DV, dtype=dtype, device=device) * 0.1
    sequence_count = B
    cu_seqlens = None
    chunk_indices = None
    if args.varlen:
        if B != 1:
            raise ValueError('variable-length mode requires --batch-size 1.')
        sequence_count = args.varlen_sequences or max(1, min(32, T // max(args.chunk_size, 1)))
        if sequence_count > T:
            raise ValueError('varlen-sequences cannot exceed sequence-length.')
        # Use deterministic, uneven segments so chunk-boundary addressing is exercised.
        lengths = torch.full((sequence_count,), T // sequence_count, dtype=torch.long)
        lengths[: T % sequence_count] += 1
        if sequence_count > 1 and T > sequence_count * 2:
            lengths[0] += 1
            lengths[-1] -= 1
        cu_seqlens = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0))).to(device)
        chunk_indices = None
    h0 = torch.randn(sequence_count, HV, D, DV, dtype=torch.float32, device=device) * 0.01 \
        if args.initial_state else None
    dht = torch.randn(sequence_count, HV, D, DV, dtype=torch.float32, device=device) * 0.01 \
        if args.final_state_gradient else None

    if args.gate_mode == 'scalar':
        gate = -torch.rand(B, T, HV, device=device)
        g = torch.empty(B, T, HV, dtype=torch.float32, device=device)
        if args.varlen:
            g = (_segmented_chunk_cumsum(gate, cu_seqlens, args.chunk_size) * RCP_LN2).float()
        else:
            g = (_chunk_local_cumsum(gate, args.chunk_size) * RCP_LN2).float()
        chunk_gated_delta_rule_bwd_dhu(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            h0=h0,
            dht=dht,
            scale=D ** -0.5,
            chunk_size=args.chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
    else:
        gate = -torch.rand(B, T, HV, D, device=device) * 0.01
        gk = torch.empty(B, T, HV, D, dtype=torch.float32, device=device)
        if args.varlen:
            gk = (_segmented_chunk_cumsum(gate, cu_seqlens, args.chunk_size) * RCP_LN2).float()
        else:
            gk = (_chunk_local_cumsum(gate, args.chunk_size) * RCP_LN2).float()
        if args.gate_mode == 'both':
            if args.varlen:
                scalar_gate = -torch.rand(B, T, HV, device=device)
                g = (_segmented_chunk_cumsum(scalar_gate, cu_seqlens, args.chunk_size) * RCP_LN2).float()
            else:
                scalar_gate = -torch.rand(B, T, HV, device=device)
                g = (_chunk_local_cumsum(scalar_gate, args.chunk_size) * RCP_LN2).float()
        else:
            g = None

        # KDA expands key heads to value heads before applying the chunk-local key decay.
        qg, kg = _expand_and_gate_qk(q, k, gk, HV, args.chunk_size, cu_seqlens)
        chunk_gated_delta_rule_bwd_dhu(
            q=qg,
            k=kg,
            w=w,
            do=do,
            dv=dv,
            g=g,
            gk=gk,
            h0=h0,
            dht=dht,
            scale=D ** -0.5,
            chunk_size=args.chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    expected_ops = {'bwd_dhu'}

    dump_files = sorted(args.dump_dir.glob('*.pt'))
    dumped_ops = {path.name.split('_rank', 1)[0] for path in dump_files}
    if not expected_ops.issubset(dumped_ops):
        raise RuntimeError(
            f'Expected {sorted(expected_ops)} dumps in {args.dump_dir}, but found: '
            f'{[path.name for path in dump_files]}. Imported FLA from {repo_root}.'
        )
    print('Dump files:')
    for path in dump_files:
        print(path)


if __name__ == '__main__':
    main()
