# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FLA Kimi Delta Attention kernels against the FP32 token recurrence (forward, every gradient, varlen)."""

import pytest
import torch

from veomni.ops import VeomniOp
from veomni.utils.import_utils import is_package_available


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_package_available("fla"), reason="FLA KDA kernels need CUDA + fla"
)

NUM_HEADS, HEAD_K, HEAD_V = 3, 64, 32


def _inputs(batch: int, seq_len: int, dtype: torch.dtype, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=gen) * scale).to(dtype).requires_grad_()

    q = randn(batch, seq_len, NUM_HEADS, HEAD_K)
    k = randn(batch, seq_len, NUM_HEADS, HEAD_K)
    v = randn(batch, seq_len, NUM_HEADS, HEAD_V)
    g = randn(batch, seq_len, NUM_HEADS, HEAD_K)
    beta = torch.rand(batch, seq_len, NUM_HEADS, device="cuda", generator=gen).to(dtype).requires_grad_()
    A_log = torch.empty(NUM_HEADS, device="cuda").uniform_(1, 16, generator=gen).log().requires_grad_()
    dt_bias = (torch.randn(NUM_HEADS * HEAD_K, device="cuda", generator=gen) * 0.5).requires_grad_()
    return q, k, v, g, beta, A_log, dt_bias


def _run(op: str, impl: str, inputs, cu_seqlens, grad_out=None):
    leaves = [x.detach().clone().requires_grad_() for x in inputs]
    o, _ = VeomniOp(op, "standard", impl)(*leaves, None, cu_seqlens, use_qk_l2norm_in_kernel=True)
    if grad_out is None:
        return o.detach(), None
    (o.float() * grad_out).sum().backward()
    return o.detach(), [x.grad for x in leaves]


def _assert_close(actual, expected, name, rtol):
    """Relative to the reference's scale: the chunked kernel reassociates the recurrence."""
    err = (actual.float() - expected.float()).abs().max().item()
    scale = expected.float().abs().max().item()
    assert err <= rtol * max(scale, 1e-6), f"{name}: max err {err:.3e} vs scale {scale:.3e}"


@pytest.mark.parametrize(
    "batch,seq_len,cu_seqlens",
    [(2, 150, None), (1, 230, [0, 37, 100, 230]), (1, 130, [0, 1, 64, 65, 130])],
    ids=["dense", "varlen", "varlen_chunk_edges"],
)
def test_chunk_kda_forward_and_backward_match_reference(batch, seq_len, cu_seqlens):
    inputs = _inputs(batch, seq_len, torch.bfloat16)
    cu = None if cu_seqlens is None else torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    grad_out = torch.randn(batch, seq_len, NUM_HEADS, HEAD_V, device="cuda")

    o_ref, grads_ref = _run("chunk_kda", "eager", inputs, cu, grad_out)
    o_fla, grads_fla = _run("chunk_kda", "fla", inputs, cu, grad_out)

    _assert_close(o_fla, o_ref, "o", rtol=2e-2)
    for name, grad_fla, grad_ref in zip(("q", "k", "v", "g", "beta", "A_log", "dt_bias"), grads_fla, grads_ref):
        _assert_close(grad_fla, grad_ref, f"d{name}", rtol=3e-2)


def test_varlen_segments_are_independent():
    """A packed batch equals its segments run separately: KDA state resets at every boundary."""
    bounds = [0, 40, 41, 120]
    inputs = _inputs(1, bounds[-1], torch.bfloat16, seed=1)
    cu = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    o_packed, _ = _run("chunk_kda", "fla", inputs, cu)
    pieces = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        seg = [x[:, start:end] if x.ndim > 1 else x for x in inputs]
        pieces.append(_run("chunk_kda", "fla", seg, None)[0])
    _assert_close(o_packed, torch.cat(pieces, dim=1), "o", rtol=1e-2)


@pytest.mark.parametrize("cu_seqlens", [None, [0, 5, 33, 48]], ids=["dense", "varlen"])
def test_fused_recurrent_kda_matches_reference(cu_seqlens):
    batch = 2 if cu_seqlens is None else 1
    seq_len = 48
    inputs = _inputs(batch, seq_len, torch.bfloat16, seed=2)
    cu = None if cu_seqlens is None else torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    with torch.no_grad():
        o_ref, _ = _run("fused_recurrent_kda", "eager", inputs, cu)
        o_fla, _ = _run("fused_recurrent_kda", "fla", inputs, cu)
        o_chunk, _ = _run("chunk_kda", "fla", inputs, cu)
    _assert_close(o_fla, o_ref, "fused_recurrent o", rtol=1e-2)
    _assert_close(o_chunk, o_fla, "chunk vs fused_recurrent o", rtol=2e-2)
