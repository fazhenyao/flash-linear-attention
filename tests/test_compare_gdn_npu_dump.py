# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'compare_gdn_npu_dump.py'
SPEC = importlib.util.spec_from_file_location('compare_gdn_npu_dump', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_ct_dual_metrics_with_ansi_and_mojibake():
    output = '\n'.join([
        '[CT]  a\x94\x82 MARE_ratio a\x94\x82 0.343282 a\x94\x82 0.283387 a\x94\x82 1.2114 '
        'a\x94\x82 5.00 a\x94\x82 \x1b[32mOK\x1b[0m a\x94\x82',
        '[CT]  a\x94\x82 MERE_ratio a\x94\x82 0.000562 a\x94\x82 0.000542 a\x94\x82 1.0372 '
        'a\x94\x82 1.50 a\x94\x82 \x1b[32mOK\x1b[0m a\x94\x82',
        '[CT]  a\x94\x82 RMSE_ratio a\x94\x82 0.000010 a\x94\x82 0.000010 a\x94\x82 2.2112 '
        'a\x94\x82 1.50 a\x94\x82 \x1b[31mFAIL\x1b[0m a\x94\x82',
        '[CT]  a\x94\x82 ERR_COUNT_ratio a\x94\x82 44 a\x94\x82 44 a\x94\x82 1.0000 '
        'a\x94\x82 2.00 a\x94\x82 \x1b[32mOK\x1b[0m a\x94\x82',
    ])

    metrics = MODULE._parse_ct_dual_metrics(output)

    assert [metric.name for metric in metrics] == [
        'MARE_ratio', 'MERE_ratio', 'RMSE_ratio', 'ERR_COUNT_ratio',
    ]
    assert [metric.passed for metric in metrics] == [True, True, False, True]
    assert metrics[2].ratio == 2.2112


def test_parse_ct_dual_metrics_with_unicode_table():
    output = '│ RMSE_ratio │ 0.000009 │ 0.000009 │ 0.1482 │ 1.50 │ OK │'

    metrics = MODULE._parse_ct_dual_metrics(output)

    assert len(metrics) == 1
    assert metrics[0].name == 'RMSE_ratio'
    assert metrics[0].passed


def test_finite_quantiles_samples_large_tensors(monkeypatch):
    monkeypatch.setattr(MODULE, '_QUANTILE_SAMPLE_SIZE', 10)
    values = torch.arange(101, dtype=torch.float32)

    quantiles, sample_count = MODULE._finite_quantiles(values, torch.tensor([0.5]))

    assert sample_count <= 10
    assert quantiles == [49.5]


def test_to_npu_detaches_before_transfer(monkeypatch):
    source = torch.ones(2, 3, requires_grad=True)
    observed = {}

    def fake_to(self, device):
        observed['requires_grad'] = self.requires_grad
        observed['device'] = device
        return self

    monkeypatch.setattr(torch.Tensor, 'to', fake_to)

    result = MODULE._to_npu(source)

    assert result.shape == source.shape
    assert observed == {'requires_grad': False, 'device': 'npu'}


def test_compare_moves_to_cpu_before_cast(monkeypatch):
    calls = []
    original_cpu = torch.Tensor.cpu
    original_float = torch.Tensor.float

    def tracked_cpu(self, *args, **kwargs):
        calls.append('cpu')
        return original_cpu(self, *args, **kwargs)

    def tracked_float(self, *args, **kwargs):
        calls.append('float')
        return original_float(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'cpu', tracked_cpu)
    monkeypatch.setattr(torch.Tensor, 'float', tracked_float)

    result = MODULE._compare('value', torch.ones(2), torch.ones(2), rtol=0.0, atol=0.0)

    assert result.passed
    assert calls[:2] == ['cpu', 'cpu']
    assert calls.index('float') > calls.index('cpu')


def test_comparison_stats_streams_and_matches_allclose(monkeypatch):
    monkeypatch.setattr(MODULE, '_STATS_CHUNK_SIZE', 3)
    actual = torch.tensor([0.0, 1.0, float('nan'), float('inf'), 4.2, 5.0, 6.0])
    expected = torch.tensor([0.0, 1.0, float('nan'), float('inf'), 4.0, 5.0, 6.0])

    stats = MODULE._comparison_stats(actual, expected, rtol=0.0, atol=0.1)

    assert not stats[0]
    assert stats[1] == pytest.approx(0.2)
    assert stats[-1] == pytest.approx(1 / 7)
