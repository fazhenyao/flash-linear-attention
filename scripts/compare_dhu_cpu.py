# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Dump DHU GPU data, compare with CPU, and optionally run CT viz."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
from fla.ops.common.chunk_o import chunk_bwd_dv_local
from fla.ops.common.dhu_cpu import chunk_gated_delta_rule_bwd_dhu_cpu
from fla.ops.common.dhu_capture import capture_dhu


def capture_network_case(args: argparse.Namespace) -> dict:
    if args.network_script is None:
        raise ValueError("--network-script is required for network capture")
    script_path = args.network_script.resolve()
    spec = importlib.util.spec_from_file_location("dhu_network_runner", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load network script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = getattr(module, args.network_entry, None)
    if entry is None or not callable(entry):
        raise AttributeError(f"network script must define callable {args.network_entry}()")
    args.capture_dir.mkdir(parents=True, exist_ok=True)
    with capture_dhu(args.capture_dir, max_calls=args.max_calls):
        entry()
    captures = sorted(args.capture_dir.glob("call_*.pt"))
    if not captures:
        raise RuntimeError("network run completed but no DHU call was captured")
    if args.capture_index >= len(captures):
        raise IndexError(f"capture index {args.capture_index} out of range; found {len(captures)} calls")
    selected = captures[args.capture_index]
    print(f"selected network DHU capture: {selected}")
    return _normalize_payload(torch.load(selected, map_location="cpu", weights_only=False))
def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _random_tensor(shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype, args: argparse.Namespace) -> torch.Tensor:
    if args.input_range is not None:
        low, high = args.input_range
        return torch.rand(shape, device=device, dtype=dtype) * (high - low) + low
    return torch.randn(shape, device=device, dtype=dtype) * args.input_scale

def dump_case(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("dump mode requires CUDA")
    B, T, H, HV, K, V = args.batch, args.seq_len, args.heads, args.value_heads, args.key_dim, args.value_dim
    if HV % H:
        raise ValueError("value_heads must be divisible by heads")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    dtype = _dtype(args.dtype)
    q = _random_tensor((B, T, H, K), device=device, dtype=dtype, args=args)
    k = _random_tensor((B, T, H, K), device=device, dtype=dtype, args=args)
    w = _random_tensor((B, T, HV, K), device=device, dtype=dtype, args=args)
    do = _random_tensor((B, T, HV, V), device=device, dtype=dtype, args=args)
    g = torch.randn(B, T, HV, device=device, dtype=torch.float32) * 0.1 if args.gate else None
    gk = torch.randn(B, T, HV, K, device=device, dtype=torch.float32) * 0.1 if args.gk else None
    cu = None
    if args.varlen:
        if B != 1:
            raise ValueError("--varlen currently requires --batch 1")
        if args.varlen_avg_len <= 0:
            raise ValueError("--varlen-avg-len must be positive")
        n_sequences = max(1, (T + args.varlen_avg_len - 1) // args.varlen_avg_len)
        base_len, remainder = divmod(T, n_sequences)
        lengths = [base_len + (index < remainder) for index in range(n_sequences)]
        cu = torch.tensor([0, *torch.cumsum(torch.tensor(lengths), dim=0).tolist()], device=device, dtype=torch.long)
    N = B if cu is None else len(cu) - 1
    state_shape = (N, HV, V, K) if args.state_v_first else (N, HV, K, V)
    dht = _random_tensor(state_shape, device=device, dtype=torch.float32, args=args) if args.final_state_grad else None
    h0 = torch.zeros(*state_shape, device=device, dtype=torch.float32) if args.initial_state else None
    scale = args.scale if args.scale is not None else K ** -0.5
    dv = chunk_bwd_dv_local(q=q, k=k, do=do, g=g, scale=scale, cu_seqlens=cu, chunk_size=args.chunk_size)
    dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu(
        q=q, k=k, w=w, do=do, dv=dv, g=g, gk=gk, h0=h0, dht=dht,
        scale=scale, state_v_first=args.state_v_first, cu_seqlens=cu, chunk_size=args.chunk_size,
    )
    torch.cuda.synchronize(device)
    return {
        "q": q.cpu(), "k": k.cpu(), "w": w.cpu(), "do": do.cpu(), "dv": dv.cpu(),
        "g": None if g is None else g.cpu(), "gk": None if gk is None else gk.cpu(),
        "h0": None if h0 is None else h0.cpu(), "dht": None if dht is None else dht.cpu(),
        "cu_seqlens": None if cu is None else cu.cpu(), "gpu_dh": dh.cpu(),
        "gpu_dh0": None if dh0 is None else dh0.cpu(), "gpu_dv2": dv2.cpu(),
        "scale": scale, "chunk_size": args.chunk_size, "state_v_first": args.state_v_first,
    }


def _normalize_payload(payload: dict) -> dict:
    """Normalize random dumps and DHUCapture ``kwargs/outputs`` payloads."""
    if "kwargs" not in payload or "outputs" not in payload:
        return payload
    kwargs = payload["kwargs"]
    outputs = payload["outputs"]
    if not isinstance(kwargs, dict):
        raise ValueError("captured DHU payload must store keyword arguments in a dict")
    dh, dh0, dv2 = outputs
    return {
        "q": kwargs["q"],
        "k": kwargs["k"],
        "w": kwargs["w"],
        "do": kwargs["do"],
        "dv": kwargs["dv"],
        "g": kwargs.get("g"),
        "gk": kwargs.get("gk"),
        "h0": kwargs.get("h0"),
        "dht": kwargs.get("dht"),
        "cu_seqlens": kwargs.get("cu_seqlens"),
        "scale": kwargs.get("scale"),
        "chunk_size": kwargs.get("chunk_size", 64),
        "state_v_first": kwargs.get("state_v_first", False),
        "gpu_dh": dh,
        "gpu_dh0": dh0,
        "gpu_dv2": dv2,
    }

def _ct_run(command: list[str], export_only: bool) -> bool:
    print(" ".join(command))
    return export_only or subprocess.run(command, check=False).returncode == 0


def _run_viz(payload: dict, cpu_outputs: tuple, args: argparse.Namespace) -> bool:
    if args.ct_export_dir is None:
        return True
    if not args.export_only and shutil.which(args.ct) is None:
        raise RuntimeError(f"CT executable not found: {args.ct!r}; pass --ct PATH or use --export-only")
    args.ct_export_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = args.ct_export_dir / "viz_result"
    failed = False
    for index, name in enumerate(("dh", "dh0", "dv2")):
        gpu_output = (payload["gpu_dh"], payload["gpu_dh0"], payload["gpu_dv2"])[index]
        if gpu_output is None:
            continue
        test_path = args.ct_export_dir / f"{name}_test.pt"
        golden_path = args.ct_export_dir / f"{name}_golden.pt"
        torch.save(gpu_output, test_path)
        torch.save(cpu_outputs[index], golden_path)
        command = [args.ct, "viz", str(test_path), str(golden_path), "--out_dir", str(viz_dir), "--name", name]
        if args.viz_spatial:
            command.append("--spatial")
        if args.viz_sample_ratio is not None:
            command += ["--sample-ratio", str(args.viz_sample_ratio)]
        failed |= not _ct_run(command, args.export_only)
    return not failed

def compare_case(payload: dict, args: argparse.Namespace) -> int:
    payload = _normalize_payload(payload)
    common = dict(
        q=payload["q"], k=payload["k"], w=payload["w"], do=payload["do"], dv=payload["dv"],
        g=payload["g"], gk=payload["gk"], h0=payload["h0"], dht=payload["dht"],
        scale=payload["scale"], state_v_first=payload["state_v_first"],
        cu_seqlens=payload["cu_seqlens"], chunk_size=payload["chunk_size"],
    )
    cpu_outputs = chunk_gated_delta_rule_bwd_dhu_cpu(**common, output_dtype=payload["q"].dtype)
    pairs = [("dh", payload["gpu_dh"], cpu_outputs[0]), ("dv2", payload["gpu_dv2"], cpu_outputs[2])]
    if payload["gpu_dh0"] is not None:
        pairs.append(("dh0", payload["gpu_dh0"], cpu_outputs[1]))
    failed = False
    for name, gpu, cpu in pairs:
        diff = (gpu.float() - cpu.float()).abs()
        rel = diff / cpu.float().abs().clamp_min(1e-8)
        ok = torch.isfinite(diff).all().item() and torch.allclose(gpu.float(), cpu.float(), rtol=args.rtol, atol=args.atol)
        print(f"{name}: max_abs={diff.max().item():.6e} max_rel={rel.max().item():.6e} mean_abs={diff.mean().item():.6e} {'OK' if ok else 'FAIL'}")
        failed |= not ok
    viz_ok = _run_viz(payload, cpu_outputs, args)
    print(f"comparison: {'FAIL' if failed or not viz_ok else 'PASS'} (rtol={args.rtol:g}, atol={args.atol:g})")
    return int(failed or not viz_ok)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("dump", "capture", "compare", "run"), default="run")
    p.add_argument("--path", type=Path, default=Path("dhu_case.pt"))
    p.add_argument("--network-script", type=Path)
    p.add_argument("--network-entry", default="run")
    p.add_argument("--capture-dir", type=Path, default=Path("dhu_capture"))
    p.add_argument("--capture-index", type=int, default=0)
    p.add_argument("--max-calls", type=int, default=None)
    p.add_argument("--ct", default="ct"); p.add_argument("--ct-export-dir", type=Path, default=Path("dhu_ct"))
    p.add_argument("--viz-spatial", action="store_true")
    p.add_argument("--viz-sample-ratio", type=float, default=None)
    p.add_argument("--export-only", action="store_true"); p.add_argument("--no-ct", action="store_true")
    p.add_argument("--device", default="cuda"); p.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    p.add_argument("--batch", type=int, default=2); p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--input-scale", type=float, default=0.01)
    p.add_argument("--input-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None)
    p.add_argument("--num-cases", type=int, default=1)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--value-heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--key-dim", type=int, default=64); p.add_argument("--value-dim", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=64); p.add_argument("--scale", type=float, default=None)
    p.add_argument("--varlen", action="store_true"); p.add_argument("--varlen-avg-len", type=int, default=64)
    gate_group = p.add_mutually_exclusive_group()
    gate_group.add_argument("--gate", dest="gate", action="store_true")
    gate_group.add_argument("--no-gate", dest="gate", action="store_false")
    p.set_defaults(gate=True)
    p.add_argument("--gk", action="store_true"); p.add_argument("--initial-state", action="store_true")
    state_group = p.add_mutually_exclusive_group()
    state_group.add_argument("--final-state-grad", dest="final_state_grad", action="store_true")
    state_group.add_argument("--no-final-state-grad", dest="final_state_grad", action="store_false")
    p.set_defaults(final_state_grad=True)
    p.add_argument("--state-v-first", action="store_true")
    p.add_argument("--rtol", type=float, default=6e-3)
    p.add_argument("--atol", type=float, default=6e-3)
    args = p.parse_args()
    if args.input_scale <= 0:
        p.error("--input-scale must be positive")
    if args.input_range is not None and args.input_range[0] >= args.input_range[1]:
        p.error("--input-range requires MIN < MAX")
    if args.num_cases <= 0:
        p.error("--num-cases must be positive")
    args.scale = args.scale if args.scale is not None else args.key_dim ** -0.5
    if args.viz_sample_ratio is not None and not 0 < args.viz_sample_ratio <= 1:
        p.error("--viz-sample-ratio must be in (0, 1]")
    if args.no_ct:
        args.ct_export_dir = None
    if args.mode == "dump":
        payload = dump_case(args)
        torch.save(payload, args.path)
        print(f"saved {args.path}")
        return 0
    if args.mode in ("capture", "run") and args.network_script is not None:
        payload = capture_network_case(args)
        torch.save(payload, args.path)
        print(f"saved normalized network capture to {args.path}")
        return compare_case(payload, args)
    if args.mode == "run" and args.num_cases == 1:
        payload = dump_case(args)
        torch.save(payload, args.path)
        print(f"saved {args.path}")
        return compare_case(payload, args)
    if args.mode == "run":
        failures = 0
        base_path = args.path
        base_ct_dir = args.ct_export_dir
        base_seed = args.seed
        for case_index in range(args.num_cases):
            args.seed = base_seed + case_index
            args.path = base_path.with_name(f"{base_path.stem}_case{case_index + 1:03d}{base_path.suffix}")
            args.ct_export_dir = None if base_ct_dir is None else base_ct_dir / f"case{case_index + 1:03d}"
            payload = dump_case(args)
            torch.save(payload, args.path)
            print(f"saved {args.path}")
            failures += compare_case(payload, args)
        print(f"random cases: {args.num_cases}, failures: {failures}")
        return int(failures != 0)
    payload = _normalize_payload(torch.load(args.path, map_location="cpu", weights_only=False))
    return compare_case(payload, args)

if __name__ == "__main__":
    raise SystemExit(main())

