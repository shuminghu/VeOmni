# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Spectral (muP) Muon update scaling and per-step Newton-Schulz coefficient schedules."""

import math

import pytest
import torch


pytest.importorskip("torch.optim._muon")

from veomni.optim.muon import DistributedMuon, adjust_muon_lr, batched_newton_schulz  # noqa: E402


QUINTIC_SCHEDULE = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]


@pytest.mark.parametrize("d_out,d_in", [(64, 64), (256, 64), (64, 256)])
def test_spectral_lr_is_sqrt_fan_ratio(d_out, d_in):
    assert adjust_muon_lr(0.02, "spectral", (d_out, d_in)) == pytest.approx(0.02 * math.sqrt(d_out / d_in))


def test_spectral_step_matches_scaled_orthogonal_update():
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(96, 32))
    grad = torch.randn(96, 32)
    lr = 0.05
    opt = DistributedMuon(
        [weight], lr=lr, weight_decay=0.0, momentum=0.0, nesterov=False,
        ns_coefficients=QUINTIC_SCHEDULE, ns_steps=len(QUINTIC_SCHEDULE),
        adjust_lr_fn="spectral", ns_implementation="std",
    )  # fmt: skip
    before = weight.detach().clone()
    weight.grad = grad.clone()
    opt.step()
    expected = batched_newton_schulz(grad, ns_coefficients=QUINTIC_SCHEDULE, ns_steps=len(QUINTIC_SCHEDULE))
    torch.testing.assert_close(before - weight.detach(), lr * math.sqrt(96 / 32) * expected, rtol=1e-4, atol=1e-6)


def test_quintic_schedule_orthogonalizes():
    torch.manual_seed(0)
    grad = torch.randn(128, 64, dtype=torch.float32)
    ortho = batched_newton_schulz(grad, ns_coefficients=QUINTIC_SCHEDULE, ns_steps=5, compute_dtype=torch.float32)
    singular_values = torch.linalg.svdvals(ortho)
    assert singular_values.min() > 0.7 and singular_values.max() < 1.2
