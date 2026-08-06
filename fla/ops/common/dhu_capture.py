# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Capture real DHU calls made by an end-to-end FLA network run."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu


def _cpu_detach(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_detach(item) for item in value)
    if isinstance(value, dict):
        return {key: _cpu_detach(item) for key, item in value.items()}
    return value


class DHUCapture:
    """Capture every DHU invocation while the context is active.

    Each call is saved as ``call_000000.pt`` under ``output_dir``. The payload
    contains CPU copies of the keyword/positional inputs, GPU outputs, and
    scalar kernel metadata. Capturing is intended for debugging and numerical
    comparison, not for production training.
    """

    def __init__(self, output_dir: str | Path, *, max_calls: int | None = None):
        self.output_dir = Path(output_dir)
        self.max_calls = max_calls
        self.count = 0
        self._original = chunk_gated_delta_rule_bwd_dhu
        self._patched: list[tuple[ModuleType, str]] = []

    def _wrapper(self, *args, **kwargs):
        result = self._original(*args, **kwargs)
        if self.max_calls is None or self.count < self.max_calls:
            payload = {
                "args": _cpu_detach(args),
                "kwargs": _cpu_detach(kwargs),
                "outputs": _cpu_detach(result),
                "call_index": self.count,
            }
            self.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, self.output_dir / f"call_{self.count:06d}.pt")
        self.count += 1
        return result

    def __enter__(self):
        wrapper = self._wrapper
        for module in tuple(sys.modules.values()):
            if module is None:
                continue
            try:
                namespace = vars(module)
            except TypeError:
                continue
            for name, value in tuple(namespace.items()):
                if value is self._original:
                    setattr(module, name, wrapper)
                    self._patched.append((module, name))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for module, name in self._patched:
            setattr(module, name, self._original)
        self._patched.clear()
        return False


@contextlib.contextmanager
def capture_dhu(output_dir: str | Path, *, max_calls: int | None = None):
    """Context-manager shorthand for :class:`DHUCapture`."""
    with DHUCapture(output_dir, max_calls=max_calls) as capture:
        yield capture


__all__ = ["DHUCapture", "capture_dhu"]
