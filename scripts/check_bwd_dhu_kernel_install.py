#!/usr/bin/env python3

"""Check whether the latest bwd_dhu precision kernel is built and installed."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


OP_NAME = 'chunk_gated_delta_rule_bwd_dhu'
KERNEL_PREFIX = 'ChunkGatedDeltaRuleBwdDhu_'
CONFIG_RELATIVE = Path(
    'opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/kernel/config/'
    'ascend910_93/chunk_gated_delta_rule_bwd_dhu.json'
)
SOURCE_DIRECTORY = Path(
    'fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/op_kernel'
)
SOURCE_FILES = (
    'chunk_gated_delta_rule_bwd_dhu_base.h',
    'chunk_gated_delta_rule_bwd_dhu_cube.h',
    'chunk_gated_delta_rule_bwd_dhu_vec.h',
)
SOURCE_MARKERS = (
    ('ub_buffers_do_not_overlap', 'shared half input/output'),
    ('dht_conversion_does_not_overlap', 'qCastLocal and qLocal do not overlap'),
    ('fp32_state_tensor', 'GlobalTensor<float> bdhStateGm'),
    ('fp32_state_workspace_layout', 'FP32 bdhState'),
    ('fp32_state_initialized_from_dht', 'this->bdhStateGm[bdhStateOffset_ + offset]'),
    ('fp32_state_used_for_recurrence', 'CopyIn(this->bdhCastLocal, this->bdhCastLocal, this->bdhStateGm'),
    ('fp32_state_written_after_update', 'CopyOut(this->qdoCastLocal, this->qdoCastLocal, this->bdhStateGm'),
    ('fp32_cube_output_type', 'using ElementBdh = float'),
    ('fp32_term1_workspace', 'GlobalTensor<ElementBdh> gmDhTerm1'),
    ('fp32_term2_workspace', 'GlobalTensor<ElementBdh> gmDhTerm2'),
    ('fp32_cube_output_layout', 'params.layoutBdh'),
    ('high_precision_gate_exp_config', 'ExpAlgo::PRECISION_1ULP_FTZ_FALSE'),
    ('high_precision_gate_exp_call', 'Exp<float, GATE_EXP_CONFIG>'),
    ('gate_exp_helper_used', 'GateExp('),
)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _find_artifacts(root: Path) -> list[Path]:
    if not root.exists():
        return []
    paths = []
    for path in root.rglob('*'):
        if not path.is_file() or OP_NAME not in str(path):
            continue
        if path.suffix in ('.o', '.json') and path.name.startswith(KERNEL_PREFIX):
            paths.append(path.resolve())
    return sorted(set(paths), key=str)


def _print_artifacts(title: str, paths: list[Path]) -> dict[str, list[Path]]:
    print(f'=== {title} ===')
    if not paths:
        print('NONE')
        return {}
    by_hash: dict[str, list[Path]] = {}
    for path in paths:
        digest = _sha256(path)
        by_hash.setdefault(digest, []).append(path)
        stat = path.stat()
        print(f'sha256={digest} size={stat.st_size} mtime_ns={stat.st_mtime_ns} path={path}')
    return by_hash


def _print_packages(repo: Path) -> None:
    print('=== Local Install Packages ===')
    packages = sorted((repo / 'build_out').glob('*.run'))
    if not packages:
        print('NONE')
        return
    for path in packages:
        stat = path.stat()
        print(f'sha256={_sha256(path)} size={stat.st_size} mtime_ns={stat.st_mtime_ns} path={path.resolve()}')


def _configured_kernel_stems(config: Path) -> set[str]:
    print('=== Installed Kernel Configuration ===')
    if not config.is_file():
        print(f'config_missing={config}')
        return set()
    print(f'config={config}')
    print(f'config_sha256={_sha256(config)}')
    try:
        data = json.loads(config.read_text(encoding='utf-8'))
    except Exception as error:
        print(f'config_error={error!r}')
        return set()
    stems = set()
    for index, item in enumerate(data.get('binList', [])):
        json_path = item.get('binInfo', {}).get('jsonFilePath')
        stem = Path(json_path).stem if json_path else None
        print(f'bin[{index}]={json_path}')
        if stem:
            stems.add(stem)
    print(f'configured_kernel_stems={sorted(stems)}')
    return stems


def _check_markers(title: str, paths: list[Path]) -> bool:
    print(f'=== {title} ===')
    if not paths:
        print('NONE')
        return False
    contents = []
    for path in paths:
        try:
            contents.append(path.read_text(encoding='utf-8', errors='replace'))
        except OSError as error:
            print(f'path={path} error={error!r}')
        print(f'path={path}')
    combined = '\n'.join(contents)
    marker_results = [(name, marker in combined) for name, marker in SOURCE_MARKERS]
    for name, present in marker_results:
        print(f'{name}={present}')
    result = len(contents) == len(paths) and all(present for _, present in marker_results)
    print(f'all_latest_precision_markers={result}')
    return result


def _run_command(name: str, command: list[str]) -> None:
    print(f'{name}_command={command!r}')
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    except Exception as error:
        print(f'{name}_error={error!r}')
        return
    print(f'{name}_returncode={result.returncode}')
    if result.stdout.strip():
        print(f'{name}_stdout={result.stdout.strip()}')
    if result.stderr.strip():
        print(f'{name}_stderr={result.stderr.strip()}')


def _print_runtime_info() -> None:
    print('=== Runtime Discovery ===')
    _run_command('npu_smi', ['npu-smi', 'info'])
    try:
        import torch_npu
    except Exception as error:
        print(f'torch_npu_import_error={error!r}')
        return
    print(f'torch_npu={Path(torch_npu.__file__).resolve()}')
    try:
        print(f'npu_device_count={torch_npu.npu.device_count()}')
        print(f'npu_device={torch_npu.npu.current_device()}')
    except Exception as error:
        print(f'npu_runtime_error={error!r}')


def _installed_root() -> Path | None:
    try:
        import fla_npu
    except Exception as error:
        print(f'fla_npu_import_error={error!r}')
        return None
    path = Path(fla_npu.__file__).resolve().parent
    print(f'fla_npu={Path(fla_npu.__file__).resolve()}')
    return path


def _compare_hashes(
    source_has_fix: bool,
    build_has_fix: bool,
    build_hashes: dict[str, list[Path]],
    installed_hashes: dict[str, list[Path]],
    configured_installed_paths: list[Path],
) -> None:
    print('=== Verdict ===')
    build_objects = {digest for digest, paths in build_hashes.items() if any(path.suffix == '.o' for path in paths)}
    installed_objects = {
        digest for digest, paths in installed_hashes.items() if any(path.suffix == '.o' for path in paths)
    }
    configured_objects = {_sha256(path) for path in configured_installed_paths if path.suffix == '.o'}
    common = build_objects & configured_objects
    print(f'build_object_hashes={sorted(build_objects)}')
    print(f'installed_object_hashes={sorted(installed_objects)}')
    print(f'configured_installed_object_hashes={sorted(configured_objects)}')
    print(f'common_object_hashes={sorted(common)}')
    print(f'source_has_latest_precision_fix={source_has_fix}')
    print(f'build_copy_has_latest_precision_fix={build_has_fix}')
    if not source_has_fix:
        print('result=SOURCE_OLD reason=source tree does not contain every latest precision marker')
    elif not build_has_fix:
        print('result=BUILD_OLD reason=build copy does not contain every latest precision marker; rebuild the operator')
    elif not build_objects:
        print('result=UNKNOWN reason=no local build kernel objects found')
    elif not installed_objects:
        print('result=UNKNOWN reason=no installed kernel objects found under fla_npu')
    elif not configured_objects:
        print('result=UNKNOWN reason=installed kernel configuration does not reference any discovered objects')
    elif build_objects == configured_objects:
        print('result=FULL_ACTIVE_MATCH configured installed kernels exactly match the local build')
        if installed_objects != configured_objects:
            print('note=unreferenced historical kernel objects exist but are not selected by the installed config')
    elif common:
        print('result=PARTIAL_ACTIVE_MATCH only some configured installed kernels match the local build')
    else:
        print('result=MISMATCH installed OPP kernel objects do not match the local build')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--npu-repo', '--repo',
        type=Path,
        default=Path('/data/fazhenyao/flash-linear-attention-npu'),
        dest='npu_repo',
        help='Path to the flash-linear-attention-npu source tree.',
    )
    parser.add_argument('--report', type=Path, default=Path('bwd_dhu_kernel_install_report.txt'))
    args = parser.parse_args()

    repo = args.npu_repo.expanduser().resolve()
    report = args.report.expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('w', encoding='utf-8') as report_file:
        with contextlib.redirect_stdout(_Tee(sys.stdout, report_file)), contextlib.redirect_stderr(
            _Tee(sys.stderr, report_file)
        ):
            print('=== Environment ===')
            print(f'python={sys.executable}')
            print(f'python_version={sys.version.replace(chr(10), " ")}')
            print(f'platform={platform.platform()}')
            print(f'npu_repo={repo}')
            print(f'ASCEND_HOME_PATH={os.environ.get("ASCEND_HOME_PATH")}')
            print(f'ASCEND_OPP_PATH={os.environ.get("ASCEND_OPP_PATH")}')
            print(f'ASCEND_CUSTOM_OPP_PATH={os.environ.get("ASCEND_CUSTOM_OPP_PATH")}')

            installed_root = _installed_root()
            _print_runtime_info()
            sources = [repo / SOURCE_DIRECTORY / name for name in SOURCE_FILES]
            copied_sources = sorted(
                path.resolve()
                for name in SOURCE_FILES
                for path in (repo / 'build').glob(f'**/{OP_NAME}/op_kernel/{name}')
                if path.is_file()
            )
            source_has_fix = _check_markers('Source Fix Markers', [path for path in sources if path.is_file()])
            build_has_fix = _check_markers('Build Copy Fix Markers', copied_sources)
            _print_packages(repo)

            build_paths = _find_artifacts(repo / 'build') + _find_artifacts(repo / 'build_out')
            installed_paths = _find_artifacts(installed_root / 'opp') if installed_root is not None else []
            configured_stems = _configured_kernel_stems(installed_root / CONFIG_RELATIVE) if installed_root else set()
            configured_installed_paths = [path for path in installed_paths if path.stem in configured_stems]
            build_hashes = _print_artifacts('Build Artifacts', sorted(set(build_paths), key=str))
            installed_hashes = _print_artifacts('Installed Artifacts', installed_paths)
            _print_artifacts('Configured Installed Artifacts', configured_installed_paths)
            _compare_hashes(
                source_has_fix,
                build_has_fix,
                build_hashes,
                installed_hashes,
                configured_installed_paths,
            )
            print(f'report={report}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
