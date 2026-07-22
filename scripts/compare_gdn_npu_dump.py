#!/usr/bin/env python3

"""Replay one or more FLA dumps with Ascend C operators and summarize accuracy."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


_QUANTILE_SAMPLE_SIZE = 1_000_000
_STATS_CHUNK_SIZE = 1_000_000


@dataclass
class TensorComparison:
    name: str
    passed: bool
    shape: list[int] | None = None
    max_abs: float | None = None
    mean_abs: float | None = None
    max_rel: float | None = None
    p50_abs: float | None = None
    p90_abs: float | None = None
    p99_abs: float | None = None
    mismatch_ratio: float | None = None
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.passed


@dataclass
class CtDualMetric:
    name: str
    test: float
    bench: float
    ratio: float
    threshold: float
    passed: bool


@dataclass
class CtDualResult:
    name: str
    passed: bool
    returncode: int | None
    metrics: list[CtDualMetric]
    skipped: bool = False
    reason: str | None = None


@dataclass
class ReplayResult:
    passed: bool
    comparisons: list[TensorComparison]
    ct_dual: list[CtDualResult]


@dataclass
class CaseResult:
    dump: str
    op: str | None
    passed: bool
    metadata: dict
    comparisons: list[TensorComparison]
    ct_dual: list[CtDualResult]
    error: str | None = None


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _tensor(data: dict, *names: str) -> torch.Tensor | None:
    for name in names:
        value = data.get(name)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _as_int(value, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return bool(value.item())
    return bool(value)


def _as_int_list(value) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.reshape(-1).tolist()
    return [int(item) for item in value]


def _make_chunk_indices(cu_seqlens: list[int] | None, chunk_size: int) -> list[int] | None:
    if cu_seqlens is None:
        return None
    indices = []
    for sequence_id, (bos, eos) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        for chunk_id in range((eos - bos + chunk_size - 1) // chunk_size):
            indices.extend((sequence_id, chunk_id))
    return indices


def _to_npu(tensor: torch.Tensor | None, *, head_first: bool = False, scale: float = 1.0):
    if tensor is None:
        return None
    tensor = tensor.detach()
    if head_first:
        tensor = tensor.transpose(1, 2)
    if scale != 1.0:
        tensor = tensor * scale
    return tensor.contiguous().to('npu')


def _finite_quantiles(values: torch.Tensor, probabilities: torch.Tensor) -> tuple[list[float], int]:
    values = values.reshape(-1)
    if values.numel() > _QUANTILE_SAMPLE_SIZE:
        step = (values.numel() + _QUANTILE_SAMPLE_SIZE - 1) // _QUANTILE_SAMPLE_SIZE
        values = values[::step]
    return torch.quantile(values, probabilities).tolist(), values.numel()


def _comparison_stats(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, float, float, float, list[float], int, float]:
    actual = actual.detach().cpu().reshape(-1)
    expected = expected.detach().cpu().reshape(-1)
    sample_step = max(1, (actual.numel() + _QUANTILE_SAMPLE_SIZE - 1) // _QUANTILE_SAMPLE_SIZE)
    sample_parts = []
    finite_count = 0
    diff_sum = 0.0
    max_abs = float('-inf')
    max_rel = float('-inf')
    mismatch_count = 0
    passed = True

    for start in range(0, actual.numel(), _STATS_CHUNK_SIZE):
        end = min(start + _STATS_CHUNK_SIZE, actual.numel())
        actual_part = actual[start:end].float()
        expected_part = expected[start:end].float()
        finite = torch.isfinite(actual_part) & torch.isfinite(expected_part)
        both_nan = torch.isnan(actual_part) & torch.isnan(expected_part)
        same_inf = torch.isinf(actual_part) & torch.isinf(expected_part) & (actual_part == expected_part)
        diff = (actual_part - expected_part).abs()
        tolerance = atol + rtol * expected_part.abs()
        close = (finite & (diff <= tolerance)) | both_nan | same_inf
        passed = passed and bool(close.all())

        finite_diff = diff[finite]
        count = finite_diff.numel()
        if count:
            finite_count += count
            diff_sum += finite_diff.double().sum().item()
            max_abs = max(max_abs, finite_diff.max().item())
            finite_rel = diff[finite] / expected_part[finite].abs().clamp_min(atol)
            max_rel = max(max_rel, finite_rel.max().item())
            mismatch_count += int((finite & (diff > tolerance)).sum().item())

        if count:
            sample_parts.append(finite_diff[::sample_step])

    if finite_count:
        samples = torch.cat(sample_parts)
        quantiles, quantile_count = _finite_quantiles(samples, torch.tensor([0.5, 0.9, 0.99]))
        mean_abs = diff_sum / finite_count
    else:
        max_abs = mean_abs = max_rel = float('nan')
        quantiles, quantile_count = [float('nan')] * 3, 0
    mismatch_ratio = mismatch_count / actual.numel() if actual.numel() else 0.0
    return passed, max_abs, mean_abs, max_rel, quantiles, quantile_count, mismatch_ratio


def _compare(name: str, actual, expected, *, rtol: float, atol: float) -> TensorComparison:
    if actual is None or expected is None:
        passed = actual is None and expected is None
        print(f'{name}: {"PASS" if passed else "FAIL"} (actual={actual is not None}, expected={expected is not None})')
        return TensorComparison(name=name, passed=passed, reason='both_missing' if passed else 'missing_tensor')

    if actual.shape != expected.shape:
        print(f'{name}: FAIL shape actual={tuple(actual.shape)} expected={tuple(expected.shape)}')
        return TensorComparison(name=name, passed=False, reason='shape_mismatch')

    passed, max_abs, mean_abs, max_rel, quantiles, quantile_count, mismatch_ratio = _comparison_stats(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
    )
    print(
        f'{name}: {"PASS" if passed else "FAIL"} shape={tuple(actual.shape)} '
        f'max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} max_rel={max_rel:.6e} '
        f'p50_abs={quantiles[0]:.6e} p90_abs={quantiles[1]:.6e} p99_abs={quantiles[2]:.6e} '
        f'quantile_count={quantile_count} mismatch_ratio={mismatch_ratio:.6e} rtol={rtol:g} atol={atol:g}'
    )
    return TensorComparison(
        name=name,
        passed=passed,
        shape=list(actual.shape),
        max_abs=max_abs,
        mean_abs=mean_abs,
        max_rel=max_rel,
        p50_abs=quantiles[0],
        p90_abs=quantiles[1],
        p99_abs=quantiles[2],
        mismatch_ratio=mismatch_ratio,
    )


def _run_ct_viz(
    pairs: list[tuple[str, torch.Tensor | None, torch.Tensor | None]],
    *,
    out_dir: Path,
    workload: int,
    sample_count: int,
) -> None:
    print('=== CT Viz ===')
    ct_command = shutil.which('ct')
    if ct_command is None:
        raise RuntimeError('ct viz is enabled, but the ct command was not found in PATH.')

    out_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir = out_dir / 'tensors'
    tensor_dir.mkdir(parents=True, exist_ok=True)
    print(f'ct.viz.command={ct_command}')
    print(f'ct.viz.out_dir={out_dir}')
    failures = []
    for name, actual, expected in pairs:
        if actual is None or expected is None:
            print(f'ct.viz {name}: SKIP actual={actual is not None} expected={expected is not None}')
            continue
        if actual.shape != expected.shape:
            print(f'ct.viz {name}: SKIP shape actual={tuple(actual.shape)} expected={tuple(expected.shape)}')
            continue

        actual_path = tensor_dir / f'{name}_npu.pt'
        expected_path = tensor_dir / f'{name}_fla.pt'
        torch.save(actual.detach().cpu().contiguous(), actual_path)
        torch.save(expected.detach().cpu().contiguous(), expected_path)
        command = [
            ct_command,
            'viz',
            str(actual_path),
            str(expected_path),
            '--out_dir',
            str(out_dir),
            '--name',
            name,
            '-wl',
            str(workload),
            '-sc',
            str(sample_count),
        ]
        print(f'ct.viz {name}: command={command!r}')
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        print(f'ct.viz {name}: returncode={result.returncode}')
        if result.stdout.strip():
            print(f'ct.viz {name}: stdout={result.stdout.strip()}')
        if result.stderr.strip():
            print(f'ct.viz {name}: stderr={result.stderr.strip()}')
        if result.returncode != 0:
            failures.append(name)
    if failures:
        raise RuntimeError(f'ct viz failed for outputs: {failures}.')


def _dense_fp64_reference(inputs: dict, expected: dict) -> dict[str, torch.Tensor | None]:
    chunk_size, cu_seqlens, _ = _metadata(inputs)
    if cu_seqlens is not None:
        return _varlen_fp64_reference(inputs, expected, chunk_size, cu_seqlens)

    k = _tensor(inputs, 'k').double()
    B, T, H, K = k.shape
    g = _tensor(inputs, 'g')
    gk = _tensor(inputs, 'gk')
    g = g.double() if g is not None else None
    gk = gk.double() if gk is not None else None
    NT = (T + chunk_size - 1) // chunk_size

    q = _tensor(inputs, 'q').double()
    w = _tensor(inputs, 'w').double()
    do = _tensor(inputs, 'do').double()
    dv = _tensor(inputs, 'dv').double()
    HV, V = do.shape[2], do.shape[3]
    q = q.repeat_interleave(HV // q.shape[2], dim=2)
    k = k.repeat_interleave(HV // H, dim=2)
    scale = inputs.get('scale')
    if isinstance(scale, torch.Tensor):
        scale = scale.item()
    scale = float(scale if scale is not None else K ** -0.5)
    dht = _tensor(inputs, 'dht')
    state = dht.double().clone() if dht is not None else torch.zeros(B, HV, K, V, dtype=torch.float64)
    dh = torch.empty(B, NT, HV, K, V, dtype=torch.float64)
    dv2 = torch.empty(B, T, HV, V, dtype=torch.float64)
    for chunk_id in range(NT - 1, -1, -1):
        start = chunk_id * chunk_size
        end = min(start + chunk_size, T)
        dh[:, chunk_id] = state
        value_grad = torch.einsum('bthk,bhkv->bthv', k[:, start:end], state)
        if g is not None:
            g_last = g[:, end - 1]
            value_grad = value_grad * torch.exp2(g_last[:, None] - g[:, start:end]).unsqueeze(-1)
        value_grad = value_grad + dv[:, start:end]
        dv2[:, start:end] = value_grad

        state_update = state
        query = q[:, start:end]
        if g is not None:
            state_update = state_update * torch.exp2(g_last).unsqueeze(-1).unsqueeze(-1)
            query = query * torch.exp2(g[:, start:end]).unsqueeze(-1)
        if gk is not None:
            state_update = state_update * torch.exp2(gk[:, end - 1]).unsqueeze(-1)
        state = (
            state_update
            + torch.einsum('bthk,bthv->bhkv', query, do[:, start:end]) * scale
            - torch.einsum('bthk,bthv->bhkv', w[:, start:end], value_grad)
        )
    return {
        'dh': dh,
        'dh0': state if _tensor(expected, 'dh0') is not None else None,
        'dv2': dv2,
    }


def _varlen_fp64_reference(
    inputs: dict,
    expected: dict,
    chunk_size: int,
    cu_seqlens: list[int],
) -> dict[str, torch.Tensor | None]:
    k = _tensor(inputs, 'k').double()
    _, T, H, K = k.shape
    g = _tensor(inputs, 'g')
    gk = _tensor(inputs, 'gk')
    g = g.double() if g is not None else None
    gk = gk.double() if gk is not None else None
    chunk_offsets = [0]
    for bos, eos in zip(cu_seqlens, cu_seqlens[1:]):
        chunk_offsets.append(chunk_offsets[-1] + (eos - bos + chunk_size - 1) // chunk_size)
    total_chunks = chunk_offsets[-1]
    sequence_count = len(cu_seqlens) - 1

    q = _tensor(inputs, 'q').double()
    w = _tensor(inputs, 'w').double()
    do = _tensor(inputs, 'do').double()
    dv = _tensor(inputs, 'dv').double()
    HV, V = do.shape[2], do.shape[3]
    q = q.repeat_interleave(HV // q.shape[2], dim=2)
    k = k.repeat_interleave(HV // H, dim=2)
    scale = inputs.get('scale')
    if isinstance(scale, torch.Tensor):
        scale = scale.item()
    scale = float(scale if scale is not None else K ** -0.5)
    dht = _tensor(inputs, 'dht')
    dh = torch.empty(1, total_chunks, HV, K, V, dtype=torch.float64)
    dh0 = torch.empty(sequence_count, HV, K, V, dtype=torch.float64)
    dv2 = torch.empty(1, T, HV, V, dtype=torch.float64)
    for sequence_id, (bos, eos) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        state = (
            dht[sequence_id].double().clone()
            if dht is not None
            else torch.zeros(HV, K, V, dtype=torch.float64)
        )
        starts = list(range(bos, eos, chunk_size))
        for local_chunk in range(len(starts) - 1, -1, -1):
            start = starts[local_chunk]
            end = min(start + chunk_size, eos)
            dh[0, chunk_offsets[sequence_id] + local_chunk] = state
            value_grad = torch.einsum('thk,hkv->thv', k[0, start:end], state)
            if g is not None:
                g_last = g[0, end - 1]
                value_grad = value_grad * torch.exp2(g_last[None] - g[0, start:end]).unsqueeze(-1)
            value_grad = value_grad + dv[0, start:end]
            dv2[0, start:end] = value_grad

            state_update = state
            query = q[0, start:end]
            if g is not None:
                state_update = state_update * torch.exp2(g_last).unsqueeze(-1).unsqueeze(-1)
                query = query * torch.exp2(g[0, start:end]).unsqueeze(-1)
            if gk is not None:
                state_update = state_update * torch.exp2(gk[0, end - 1]).unsqueeze(-1)
            state = (
                state_update
                + torch.einsum('thk,thv->hkv', query, do[0, start:end]) * scale
                - torch.einsum('thk,thv->hkv', w[0, start:end], value_grad)
            )
        dh0[sequence_id] = state
    return {
        'dh': dh,
        'dh0': dh0 if _tensor(expected, 'dh0') is not None else None,
        'dv2': dv2,
    }


def _parse_ct_dual_metrics(output: str) -> list[CtDualMetric]:
    ansi_escape = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
    metric_names = ('MARE_ratio', 'MERE_ratio', 'RMSE_ratio', 'ERR_COUNT_ratio')
    number_pattern = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?')
    metrics = []
    for raw_line in output.splitlines():
        line = ansi_escape.sub('', raw_line)
        metric_name = next((name for name in metric_names if name in line), None)
        if metric_name is None:
            continue
        suffix = line.split(metric_name, 1)[1]
        numbers = number_pattern.findall(suffix)
        status = re.search(r'\b(OK|FAIL)\b', suffix)
        if len(numbers) < 4 or status is None:
            continue
        metrics.append(CtDualMetric(
            name=metric_name,
            test=float(numbers[0]),
            bench=float(numbers[1]),
            ratio=float(numbers[2]),
            threshold=float(numbers[3]),
            passed=status.group(1) == 'OK',
        ))
    return metrics


def _run_ct_dual(
    triples: list[tuple[str, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]],
    *,
    out_dir: Path,
    level: str,
) -> list[CtDualResult]:
    print('=== CT Dual ===')
    ct_command = shutil.which('ct')
    if ct_command is None:
        raise RuntimeError('ct dual is enabled, but the ct command was not found in PATH.')

    tensor_dir = out_dir / 'tensors'
    tensor_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, actual, fp64_reference, low_precision_reference in triples:
        if actual is None or fp64_reference is None or low_precision_reference is None:
            print(
                f'ct.dual {name}: SKIP actual={actual is not None} fp64={fp64_reference is not None} '
                f'low_precision={low_precision_reference is not None}'
            )
            results.append(
                CtDualResult(name=name, passed=True, returncode=None, metrics=[], skipped=True, reason='missing_tensor')
            )
            continue
        shapes = {actual.shape, fp64_reference.shape, low_precision_reference.shape}
        if len(shapes) != 1:
            print(
                f'ct.dual {name}: SKIP shape actual={tuple(actual.shape)} fp64={tuple(fp64_reference.shape)} '
                f'low_precision={tuple(low_precision_reference.shape)}'
            )
            results.append(CtDualResult(name=name, passed=False, returncode=None, metrics=[], reason='shape_mismatch'))
            continue

        actual_path = tensor_dir / f'{name}_npu.pt'
        fp64_path = tensor_dir / f'{name}_fp64.pt'
        low_precision_path = tensor_dir / f'{name}_fla.pt'
        torch.save(actual.detach().cpu().contiguous(), actual_path)
        torch.save(fp64_reference.detach().cpu().contiguous(), fp64_path)
        torch.save(low_precision_reference.detach().cpu().contiguous(), low_precision_path)
        result_dir = out_dir / name
        result_dir.mkdir(parents=True, exist_ok=True)
        command = [
            ct_command,
            'dual',
            str(actual_path),
            str(fp64_path),
            str(low_precision_path),
            '--out_dir',
            str(result_dir),
            '--level',
            level,
        ]
        print(f'ct.dual {name}: command={command!r}')
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        print(f'ct.dual {name}: returncode={result.returncode}')
        if result.stdout.strip():
            print(f'ct.dual {name}: stdout={result.stdout.strip()}')
        if result.stderr.strip():
            print(f'ct.dual {name}: stderr={result.stderr.strip()}')
        metrics = _parse_ct_dual_metrics(result.stdout)
        passed = result.returncode == 0 and bool(metrics) and all(metric.passed for metric in metrics)
        reason = None
        if result.returncode != 0:
            reason = 'nonzero_returncode'
        elif not metrics:
            reason = 'metrics_not_found'
        elif not passed:
            reason = 'metric_failure'
        results.append(
            CtDualResult(
                name=name,
                passed=passed,
                returncode=result.returncode,
                metrics=metrics,
                reason=reason,
            )
        )
        print(f'ct.dual {name}: parsed_result={"PASS" if passed else "FAIL"} metrics={len(metrics)}')
    return results


def _tensor_summary(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is None:
        print(f'{name}: None')
        return
    value = tensor.detach().cpu().reshape(-1)
    finite_count = 0
    value_sum = 0.0
    min_value = float('inf')
    max_value = float('-inf')
    abs_max = float('-inf')
    for start in range(0, value.numel(), _STATS_CHUNK_SIZE):
        part = value[start:start + _STATS_CHUNK_SIZE].float()
        finite_part = part[torch.isfinite(part)]
        if finite_part.numel():
            finite_count += finite_part.numel()
            value_sum += finite_part.double().sum().item()
            min_value = min(min_value, finite_part.min().item())
            max_value = max(max_value, finite_part.max().item())
            abs_max = max(abs_max, finite_part.abs().max().item())
    if finite_count:
        stats = (
            f'min={min_value:.6e} max={max_value:.6e} '
            f'mean={value_sum / finite_count:.6e} abs_max={abs_max:.6e}'
        )
    else:
        stats = 'no_finite_values'
    print(
        f'{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} '
        f'finite_ratio={finite_count / value.numel() if value.numel() else 1.0:.6e} {stats}'
    )


def _print_matrix_error_detail(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    topk: int = 16,
) -> None:
    actual = actual.detach().cpu().float()
    expected = expected.detach().cpu().float()
    if actual.shape != expected.shape or actual.ndim < 2:
        return
    diff = (actual - expected).abs()
    mismatch = torch.isfinite(actual) & torch.isfinite(expected) & (diff > atol + rtol * expected.abs())
    print(f'--- {name} Error Coordinates ---')
    print(f'mismatch_count={int(mismatch.sum().item())} total={mismatch.numel()}')
    if not mismatch.any():
        return

    matrix_mismatch = mismatch.reshape(-1, mismatch.shape[-2], mismatch.shape[-1])
    k_counts = matrix_mismatch.sum(dim=(0, 2))
    v_counts = matrix_mismatch.sum(dim=(0, 1))
    for axis_name, counts in (('k', k_counts), ('v', v_counts)):
        count = min(16, counts.numel())
        values, indices = torch.topk(counts, count)
        entries = [f'{int(index)}:{int(value)}' for value, index in zip(values, indices) if value > 0]
        print(f'top_{axis_name}_mismatch_counts={",".join(entries) if entries else "NONE"}')

    flat_diff = diff.flatten()
    count = min(topk, flat_diff.numel())
    _, flat_indices = torch.topk(flat_diff, count)
    for rank, flat_index in enumerate(flat_indices.tolist()):
        index = []
        remainder = flat_index
        for size in reversed(actual.shape):
            index.append(remainder % size)
            remainder //= size
        index = tuple(reversed(index))
        print(
            f'top_error[{rank}] index={index} actual={actual[index].item():.6e} '
            f'expected={expected[index].item():.6e} abs={diff[index].item():.6e}'
        )


def _print_environment(dump: Path, payload: dict) -> None:
    import fla_npu
    import torch_npu

    print('=== Environment ===')
    print(f'dump={dump.resolve()}')
    print(f'python={sys.version.replace(chr(10), " ")}')
    print(f'platform={platform.platform()}')
    print(f'torch={torch.__version__}')
    print(f'torch_npu={getattr(torch_npu, "__version__", "unknown")}')
    print(f'fla_npu={Path(fla_npu.__file__).resolve()}')
    print(f'ASCEND_HOME_PATH={os.environ.get("ASCEND_HOME_PATH")}')
    print(f'ASCEND_OPP_PATH={os.environ.get("ASCEND_OPP_PATH")}')
    print(f'ASCEND_CUSTOM_OPP_PATH={os.environ.get("ASCEND_CUSTOM_OPP_PATH")}')
    print(f'format_version={payload.get("format_version")} layout={payload.get("layout")}')
    print(f'op={payload.get("op")} gate_exponent_base={payload.get("gate_exponent_base", 2)}')


def _print_dump_summary(inputs: dict, expected: dict) -> None:
    chunk_size, cu_seqlens, chunk_indices = _metadata(inputs)
    print('=== Dump Metadata ===')
    print(f'chunk_size={chunk_size} state_v_first={_as_bool(inputs.get("state_v_first"))}')
    print(f'cu_seqlens={cu_seqlens}')
    print(f'chunk_indices={chunk_indices}')
    scale = inputs.get('scale')
    print(f'scale={scale.item() if isinstance(scale, torch.Tensor) else scale}')
    print('=== Input Tensors ===')
    for name in ('q', 'k', 'w', 'do', 'dv', 'g', 'gk', 'h0', 'dht'):
        _tensor_summary(name, _tensor(inputs, name))
    print('=== Expected Tensors ===')
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            _tensor_summary(name, value)


def _diagnose_bwd(actual_dh, actual_dh0, actual_dv2, inputs: dict, expected: dict, rtol: float, atol: float) -> None:
    expected_dh = _tensor(expected, 'dh')
    expected_dv2 = _tensor(expected, 'dv2')
    dht = _tensor(inputs, 'dht')
    chunk_size, cu_seqlens, chunk_indices = _metadata(inputs)

    print('=== Backward Diagnostics ===')
    if actual_dh is not None and expected_dh is not None and actual_dh.shape == expected_dh.shape:
        for chunk_id in range(actual_dh.shape[1]):
            _compare(
                f'dh_chunk[{chunk_id}]', actual_dh[:, chunk_id], expected_dh[:, chunk_id],
                rtol=rtol, atol=atol,
            )
            _print_matrix_error_detail(
                f'dh_chunk[{chunk_id}]', actual_dh[:, chunk_id], expected_dh[:, chunk_id],
                rtol=rtol, atol=atol,
            )
    if dht is not None and actual_dh is not None:
        if cu_seqlens is None:
            actual_last = actual_dh[:, -1]
            _compare('dh_last_vs_dht', actual_last, dht, rtol=rtol, atol=atol)
            if expected_dh is not None:
                _compare('expected_dh_last_vs_dht', expected_dh[:, -1], dht, rtol=rtol, atol=atol)
            if actual_last.shape == dht.shape and actual_last.ndim == 4:
                half_k = actual_last.shape[-2] // 2
                for head in range(actual_last.shape[1]):
                    _compare(
                        f'dh_last_head[{head}]_vs_dht', actual_last[:, head], dht[:, head],
                        rtol=rtol, atol=atol,
                    )
                _compare(
                    'dh_last_k_first_half_vs_dht', actual_last[..., :half_k, :], dht[..., :half_k, :],
                    rtol=rtol, atol=atol,
                )
                _compare(
                    'dh_last_k_second_half_vs_dht', actual_last[..., half_k:, :], dht[..., half_k:, :],
                    rtol=rtol, atol=atol,
                )
                if actual_last.shape[-2] % 2 == 0:
                    _compare(
                        'dh_last_k_second_half_vs_dht_first_half',
                        actual_last[..., half_k:, :], dht[..., :half_k, :],
                        rtol=rtol, atol=atol,
                    )
        else:
            print('dh_last_vs_dht: skipped for varlen; sequence-specific final chunks are non-contiguous in dh.')
    if actual_dh0 is not None and _tensor(expected, 'dh0') is not None:
        for sequence_id in range(actual_dh0.shape[0]):
            _compare(
                f'dh0_sequence[{sequence_id}]', actual_dh0[sequence_id], _tensor(expected, 'dh0')[sequence_id],
                rtol=rtol, atol=atol,
            )
            _print_matrix_error_detail(
                f'dh0_sequence[{sequence_id}]', actual_dh0[sequence_id], _tensor(expected, 'dh0')[sequence_id],
                rtol=rtol, atol=atol,
            )
    if actual_dv2 is not None and expected_dv2 is not None and actual_dv2.shape == expected_dv2.shape:
        if cu_seqlens is None:
            for chunk_id, start in enumerate(range(0, actual_dv2.shape[1], chunk_size)):
                end = min(start + chunk_size, actual_dv2.shape[1])
                _compare(
                    f'dv2_chunk[{chunk_id}]', actual_dv2[:, start:end], expected_dv2[:, start:end],
                    rtol=rtol, atol=atol,
                )
        else:
            for sequence_id, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
                _compare(
                    f'dv2_sequence[{sequence_id}]', actual_dv2[:, start:end], expected_dv2[:, start:end],
                    rtol=rtol, atol=atol,
                )


def _metadata(inputs: dict) -> tuple[int, list[int] | None, list[int] | None]:
    chunk_size = _as_int(inputs.get('BT'), 64)
    cu_seqlens = _as_int_list(inputs.get('cu_seqlens'))
    chunk_indices = _as_int_list(inputs.get('chunk_indices'))
    if chunk_indices is None:
        chunk_indices = _make_chunk_indices(cu_seqlens, chunk_size)
    return chunk_size, cu_seqlens, chunk_indices


def _case_metadata(inputs: dict) -> dict:
    chunk_size, cu_seqlens, _ = _metadata(inputs)
    k = _tensor(inputs, 'k')
    value = _tensor(inputs, 'v', 'u', 'do')
    if k is None or value is None:
        return {'chunk_size': chunk_size, 'varlen': cu_seqlens is not None}
    sequence_length = int(k.shape[1])
    return {
        'dtype': str(k.dtype).removeprefix('torch.'),
        'batch_size': int(k.shape[0]),
        'sequence_length': sequence_length,
        'key_heads': int(k.shape[2]),
        'value_heads': int(value.shape[2]),
        'key_dim': int(k.shape[-1]),
        'value_dim': int(value.shape[-1]),
        'chunk_size': chunk_size,
        'chunk_count': (sequence_length + chunk_size - 1) // chunk_size,
        'gate_mode': 'both' if _tensor(inputs, 'g') is not None and _tensor(inputs, 'gk') is not None else (
            'gk' if _tensor(inputs, 'gk') is not None else 'scalar'
        ),
        'initial_state': _tensor(inputs, 'h0') is not None,
        'final_state_gradient': _tensor(inputs, 'dht') is not None,
        'varlen': cu_seqlens is not None,
    }


def _check_supported(op: str, inputs: dict) -> None:
    if _as_bool(inputs.get('state_v_first')):
        raise ValueError('The Ascend C replay does not support state_v_first=True.')

    k = _tensor(inputs, 'k')
    value = _tensor(inputs, 'v', 'u', 'do')
    if k is None or value is None:
        raise ValueError('The dump is missing k or a value-side tensor.')
    K, V = k.shape[-1], value.shape[-1]
    chunk_size, cu_seqlens, _ = _metadata(inputs)
    if chunk_size not in (64, 128):
        raise ValueError(f'Ascend C supports chunk_size 64 or 128, got {chunk_size}.')
    if cu_seqlens is not None and k.shape[0] != 1:
        raise ValueError(f'Ascend C varlen mode requires B=1, got B={k.shape[0]}.')
    if op == 'bwd_dhu' and (K > 128 or V > 256):
        raise ValueError(f'Ascend C bwd_dhu requires K<=128 and V<=256, got K={K}, V={V}.')
    dht = _tensor(inputs, 'dht')
    if op == 'bwd_dhu' and dht is not None and dht.dtype != torch.float32:
        raise ValueError(f'Ascend C bwd_dhu requires dht float32, got {dht.dtype}.')


def _replay_bwd(
    inputs: dict,
    expected: dict,
    rtol: float,
    atol: float,
    ct_viz_dir: Path | None,
    ct_viz_workload: int,
    ct_viz_sample_count: int,
    ct_dual_dir: Path | None,
    ct_dual_level: str,
) -> ReplayResult:
    from fla_npu.ops.ascendc import chunk_gated_delta_rule_bwd_dhu

    chunk_size, cu_seqlens, chunk_indices = _metadata(inputs)
    q = _tensor(inputs, 'q')
    g = _tensor(inputs, 'g')
    gk = _tensor(inputs, 'gk')
    if g is None:
        if gk is None:
            raise ValueError('Ascend C bwd_dhu requires g; the dump contains neither g nor gk.')
        # The NPU kernel requires scalar g even for KDA. Zeros preserve the gk-only FLA semantics.
        g = torch.zeros(gk.shape[:-1], dtype=gk.dtype)
    if gk is not None and gk.dtype != g.dtype:
        gk = gk.to(g.dtype)

    scale = inputs.get('scale')
    if isinstance(scale, torch.Tensor):
        scale = scale.item()
    if scale is None:
        scale = q.shape[-1] ** -0.5
    with torch.no_grad():
        dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu(
            _to_npu(q, head_first=True),
            _to_npu(_tensor(inputs, 'k'), head_first=True),
            _to_npu(_tensor(inputs, 'w'), head_first=True),
            _to_npu(_tensor(inputs, 'do'), head_first=True),
            _to_npu(_tensor(inputs, 'dv'), head_first=True),
            float(scale),
            chunk_size,
            g=_to_npu(g, head_first=True),
            gK=_to_npu(gk, head_first=True),
            h0=_to_npu(_tensor(inputs, 'h0')),
            dht=_to_npu(_tensor(inputs, 'dht')),
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=False,
            transpose_state_layout=False,
        )
    torch.npu.synchronize()
    actual_dh = dh.transpose(1, 2).detach().cpu()
    actual_dh0 = dh0.detach().cpu() if dh0 is not None else None
    actual_dv2 = dv2.transpose(1, 2).detach().cpu()
    del dh, dh0, dv2
    torch.npu.empty_cache()
    comparisons = [
        _compare('dh', actual_dh, _tensor(expected, 'dh'), rtol=rtol, atol=atol),
        _compare('dh0', actual_dh0, _tensor(expected, 'dh0'), rtol=rtol, atol=atol),
        _compare('dv2', actual_dv2, _tensor(expected, 'dv2'), rtol=rtol, atol=atol),
    ]
    _diagnose_bwd(actual_dh, actual_dh0, actual_dv2, inputs, expected, rtol, atol)
    pairs = [
        ('dh', actual_dh, _tensor(expected, 'dh')),
        ('dh0', actual_dh0, _tensor(expected, 'dh0')),
        ('dv2', actual_dv2, _tensor(expected, 'dv2')),
    ]
    if ct_viz_dir is not None:
        _run_ct_viz(
            pairs,
            out_dir=ct_viz_dir,
            workload=ct_viz_workload,
            sample_count=ct_viz_sample_count,
        )
    ct_dual_results = []
    if ct_dual_dir is not None:
        fp64_reference = _dense_fp64_reference(inputs, expected)
        ct_dual_results = _run_ct_dual(
            [
                ('dh', actual_dh, fp64_reference['dh'], _tensor(expected, 'dh')),
                ('dh0', actual_dh0, fp64_reference['dh0'], _tensor(expected, 'dh0')),
                ('dv2', actual_dv2, fp64_reference['dv2'], _tensor(expected, 'dv2')),
            ],
            out_dir=ct_dual_dir,
            level=ct_dual_level,
        )
    return ReplayResult(
        passed=all(result.passed for result in comparisons) and all(result.passed for result in ct_dual_results),
        comparisons=comparisons,
        ct_dual=ct_dual_results,
    )


def _discover_dumps(paths: list[Path]) -> list[Path]:
    dumps = []
    prefixes = ('bwd_dhu_',)
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path.is_file():
            dumps.append(path)
        elif path.is_dir():
            dumps.extend(candidate for candidate in path.rglob('*.pt') if candidate.name.startswith(prefixes))
        else:
            raise FileNotFoundError(f'Dump path does not exist: {path}')
    return sorted(set(dumps))


def _case_output_dir(root: Path | None, dump: Path, index: int) -> Path | None:
    if root is None:
        return None
    name = re.sub(r'[^A-Za-z0-9_.-]+', '_', f'{index:04d}_{dump.parent.name}_{dump.stem}')
    return root / name


def _run_case(dump: Path, args, index: int) -> CaseResult:
    payload = torch.load(dump, map_location='cpu', weights_only=False)
    op, inputs, expected = payload['op'], payload['inputs'], payload['outputs']
    if op != 'bwd_dhu':
        raise ValueError(f'Only bwd_dhu dumps are supported, got {op}.')
    _check_supported(op, inputs)

    print(f'\n=== Case {index + 1}/{args.case_count}: {dump} ===')
    _print_environment(dump, payload)
    if payload.get('case_name'):
        print(f'case_name={payload["case_name"]}')
    if payload.get('case_description'):
        print(f'case_description={payload["case_description"]}')
    _print_dump_summary(inputs, expected)
    print('=== Comparison ===')
    replay = _replay_bwd(
        inputs,
        expected,
        args.rtol,
        args.atol,
        _case_output_dir(args.ct_viz_dir, dump, index),
        args.ct_viz_workload,
        args.ct_viz_sample_count,
        _case_output_dir(args.ct_dual_dir, dump, index),
        args.ct_dual_level,
    )
    print(f'=== Case Result: {"PASS" if replay.passed else "FAIL"} ===')
    metadata = _case_metadata(inputs)
    metadata['case_name'] = payload.get('case_name')
    metadata['case_description'] = payload.get('case_description')
    return CaseResult(
        dump=str(dump),
        op=op,
        passed=replay.passed,
        metadata=metadata,
        comparisons=replay.comparisons,
        ct_dual=replay.ct_dual,
    )


def _print_batch_summary(results: list[CaseResult]) -> None:
    print('\n=== Accuracy Matrix Summary ===')
    print('status case dtype T BT chunks H HV V gate h0 dht dump')
    for result in results:
        metadata = result.metadata
        print(
            f'{"PASS" if result.passed else "FAIL":4} '
            f'{metadata.get("case_name") or "-":24} '
            f'{metadata.get("dtype", "-"):8} {metadata.get("sequence_length", "-")!s:>4} '
            f'{metadata.get("chunk_size", "-")!s:>3} {metadata.get("chunk_count", "-")!s:>6} '
            f'{metadata.get("key_heads", "-")!s:>2} {metadata.get("value_heads", "-")!s:>2} '
            f'{metadata.get("value_dim", "-")!s:>3} '
            f'{metadata.get("gate_mode", "-"):6} {str(metadata.get("initial_state", "-")):>5} '
            f'{str(metadata.get("final_state_gradient", "-")):>5} {result.dump}'
        )
        for dual in result.ct_dual:
            failed_metrics = ','.join(metric.name for metric in dual.metrics if not metric.passed)
            ratios = ' '.join(
                f'{metric.name.removesuffix("_ratio")}={metric.ratio:.4g}/{metric.threshold:g}'
                for metric in dual.metrics
            )
            print(
                f'  ct.dual {dual.name}: {"PASS" if dual.passed else "FAIL"}'
                f'{f" failed={failed_metrics}" if failed_metrics else ""} {ratios}'
            )
        if result.error:
            print(f'  error={result.error}')
    passed = sum(result.passed for result in results)
    print(f'cases={len(results)} passed={passed} failed={len(results) - passed}')


def _release_case_memory() -> None:
    gc.collect()
    npu = getattr(torch, 'npu', None)
    if npu is None:
        return
    try:
        npu.synchronize()
        npu.empty_cache()
    except (AttributeError, RuntimeError):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dumps', type=Path, nargs='+', help='Dump files or directories searched recursively.')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--rtol', type=float, default=6e-3)
    parser.add_argument('--atol', type=float, default=6e-3)
    parser.add_argument('--ct-viz', choices=('on', 'off'), default='on')
    parser.add_argument('--ct-viz-dir', type=Path)
    parser.add_argument('--ct-viz-workload', type=int, default=1)
    parser.add_argument('--ct-viz-sample-count', type=int, default=10000)
    parser.add_argument('--ct-dual', choices=('on', 'off'), default='off')
    parser.add_argument('--ct-dual-dir', type=Path)
    parser.add_argument('--ct-dual-level', choices=('L0', 'L1', 'L2'), default='L1')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--report', type=Path, default=Path('compare_gdn_npu_dump_report.txt'))
    parser.add_argument('--summary-json', type=Path)
    args = parser.parse_args()

    dump_paths = _discover_dumps(args.dumps)
    if not dump_paths:
        raise ValueError(f'No matching dump files found under: {args.dumps}')
    args.case_count = len(dump_paths)
    args.report = args.report.expanduser().resolve()
    if args.summary_json is None:
        args.summary_json = args.report.with_name(f'{args.report.stem}_summary.json')
    else:
        args.summary_json = args.summary_json.expanduser().resolve()
    if args.ct_viz_dir is None:
        args.ct_viz_dir = args.report.parent / f'{args.report.stem}_ct_viz'
    else:
        args.ct_viz_dir = args.ct_viz_dir.expanduser().resolve()
    if args.ct_viz == 'off':
        args.ct_viz_dir = None
    if args.ct_dual_dir is None:
        args.ct_dual_dir = args.report.parent / f'{args.report.stem}_ct_dual'
    else:
        args.ct_dual_dir = args.ct_dual_dir.expanduser().resolve()
    if args.ct_dual == 'off':
        args.ct_dual_dir = None
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('w', encoding='utf-8') as report_file:
        with contextlib.redirect_stdout(_Tee(sys.stdout, report_file)), contextlib.redirect_stderr(
            _Tee(sys.stderr, report_file)
        ):
            try:
                import torch_npu  # noqa: F401

                torch.npu.set_device(args.device)
            except Exception:
                print('=== Environment Exception ===')
                traceback.print_exc()
                print(f'report={args.report}')
                return 2
            results = []
            for index, dump in enumerate(dump_paths):
                try:
                    results.append(_run_case(dump, args, index))
                except Exception as error:
                    print(f'\n=== Case {index + 1}/{args.case_count}: {dump} ===')
                    print('=== Exception ===')
                    traceback.print_exc()
                    results.append(
                        CaseResult(
                            dump=str(dump),
                            op=None,
                            passed=False,
                            metadata={},
                            comparisons=[],
                            ct_dual=[],
                            error=f'{type(error).__name__}: {error}',
                        )
                    )
                    if args.fail_fast:
                        break
                finally:
                    _release_case_memory()
            _print_batch_summary(results)
            args.summary_json.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding='utf-8')
            passed = all(result.passed for result in results) and len(results) == len(dump_paths)
            has_error = any(result.error is not None for result in results)
            print(f'=== Result: {"PASS" if passed else "FAIL"} ===')
            print(f'report={args.report}')
            print(f'summary_json={args.summary_json}')
            return 0 if passed else 2 if has_error else 1


if __name__ == '__main__':
    raise SystemExit(main())
