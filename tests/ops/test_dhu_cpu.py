# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.common.dhu_cpu import chunk_gated_delta_rule_bwd_dhu_cpu


def test_chunk_gated_delta_rule_bwd_dhu_cpu_no_gate():
    torch.manual_seed(42)
    B, T, H, HV, K, V, BT = 2, 5, 1, 2, 3, 4, 3
    q = torch.randn(B, T, H, K, dtype=torch.float16)
    k = torch.randn(B, T, H, K, dtype=torch.float16)
    w = torch.randn(B, T, HV, K, dtype=torch.float16)
    do = torch.randn(B, T, HV, V, dtype=torch.float16)
    dv = torch.randn(B, T, HV, V, dtype=torch.float16)
    dht = torch.randn(B, HV, K, V, dtype=torch.float32)
    h0 = torch.zeros_like(dht)

    dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu_cpu(
        q=q, k=k, w=w, do=do, dv=dv, dht=dht, h0=h0, chunk_size=BT, scale=0.5,
    )

    assert dh.shape == (B, 2, HV, K, V)
    assert dh0.shape == h0.shape
    assert dv2.shape == dv.shape
    assert dh.dtype == q.dtype
    assert dv2.dtype == dv.dtype
    assert dh0.dtype == torch.float32
    assert torch.isfinite(dh.float()).all()
    assert torch.isfinite(dh0).all()
    assert torch.isfinite(dv2.float()).all()


def test_chunk_gated_delta_rule_bwd_dhu_cpu_state_v_first_and_varlen():
    torch.manual_seed(42)
    T, H, HV, K, V, BT = 5, 1, 1, 2, 3, 4
    q = torch.randn(1, T, H, K, dtype=torch.float16)
    k = torch.randn(1, T, H, K, dtype=torch.float16)
    w = torch.randn(1, T, HV, K, dtype=torch.float16)
    do = torch.randn(1, T, HV, V, dtype=torch.float16)
    dv = torch.randn(1, T, HV, V, dtype=torch.float16)
    g = torch.randn(1, T, HV, dtype=torch.float32) * 0.1
    gk = torch.randn(1, T, HV, K, dtype=torch.float32) * 0.1
    dht = torch.randn(2, HV, V, K, dtype=torch.float32)
    h0 = torch.zeros_like(dht)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.long)

    dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu_cpu(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        g=g,
        gk=gk,
        dht=dht,
        h0=h0,
        state_v_first=True,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
    )

    assert dh.shape == (1, 2, HV, V, K)
    assert dh0.shape == h0.shape
    assert dv2.shape == dv.shape
    assert torch.isfinite(dh.float()).all()
    assert torch.isfinite(dh0).all()
    assert torch.isfinite(dv2.float()).all()
@pytest.mark.skipif(
    not os.environ.get("DHU_DUMP_PATH"),
    reason="set DHU_DUMP_PATH to run the GPU dump comparison",
)
def test_chunk_gated_delta_rule_bwd_dhu_cpu_matches_dump():
    """Compare the CPU reference with GPU outputs stored by compare_dhu_cpu.py."""
    payload = torch.load(os.environ["DHU_DUMP_PATH"], map_location="cpu", weights_only=False)
    common = dict(
        q=payload["q"], k=payload["k"], w=payload["w"], do=payload["do"], dv=payload["dv"],
        g=payload.get("g"), gk=payload.get("gk"), h0=payload.get("h0"), dht=payload.get("dht"),
        scale=payload.get("scale"), state_v_first=payload.get("state_v_first", False),
        cu_seqlens=payload.get("cu_seqlens"), chunk_size=payload.get("chunk_size", 64),
    )
    dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu_cpu(**common, output_dtype=payload["q"].dtype)
    torch.testing.assert_close(dh, payload["gpu_dh"].to(dtype=dh.dtype), rtol=6e-3, atol=6e-3)
    torch.testing.assert_close(dv2, payload["gpu_dv2"].to(dtype=dv2.dtype), rtol=6e-3, atol=6e-3)
    if payload.get("gpu_dh0") is not None:
        torch.testing.assert_close(dh0, payload["gpu_dh0"].float(), rtol=6e-3, atol=6e-3)
