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

"""Ternary-QAT weight statistics: grouping, zero fraction, flip rate, and the snapshot / sample schedule."""

import math
from types import SimpleNamespace

import torch
from torch import nn

from veomni.ops.qat.ternary import apply_ternary_qat, twn_codes
from veomni.trainer.callbacks.qat_stats_callback import QATStatsCallback, module_groups, ternary_stats


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(16, 32, bias=False)
        self.linear_attn = nn.Module()
        self.linear_attn.b_proj = nn.Linear(16, 3, bias=False)


class _Toy(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_Block() for _ in range(num_layers))
        self.lm_head = nn.Linear(16, 50, bias=False)

    def get_output_embeddings(self):
        return self.lm_head


def _model():
    torch.manual_seed(0)
    model = _Toy()
    apply_ternary_qat(model)
    return model


def test_grouping():
    assert module_groups("model.layers.3.linear_attn.f_proj.0") == ("linear_attn.f_proj.0", "layer_03")
    assert module_groups("model.layers.11.mlp.down_proj") == ("mlp.down_proj", "layer_11")


def test_zero_frac_and_flip_rate_match_manual_counts():
    model = _model()
    modules = [(n, m) for n, m in model.named_modules() if type(m).__name__ == "TernaryLinear"]
    assert {n for n, _ in modules} == {
        f"model.layers.{i}.{k}" for i in (0, 1) for k in ("mlp.up_proj", "linear_attn.b_proj")
    }
    previous = {n: twn_codes(m.weight)[0].clone() for n, m in modules}
    with torch.no_grad():
        model.model.layers[0].mlp.up_proj.weight.add_(torch.randn(32, 16) * 0.3)
    stats = ternary_stats(modules, previous, fsdp_group=None)

    up0 = model.model.layers[0].mlp.up_proj.weight
    codes_now = twn_codes(up0)[0]
    flips_up0 = (codes_now != previous["model.layers.0.mlp.up_proj"]).sum().item()
    assert flips_up0 > 0
    # Only layer 0's up_proj moved: the per-kind rate averages it with layer 1's unchanged up_proj.
    assert math.isclose(stats["qat/flip_rate/mlp.up_proj"], flips_up0 / (2 * up0.numel()))
    assert stats["qat/flip_rate/linear_attn.b_proj"] == 0.0
    assert stats["qat/flip_rate/layer_01"] == 0.0
    total = sum(m.weight.numel() for _, m in modules)
    assert math.isclose(stats["qat/flip_rate/all"], flips_up0 / total)

    zeros = sum((twn_codes(m.weight)[0] == 0).sum().item() for _, m in modules)
    assert math.isclose(stats["qat/zero_frac/all"], zeros / total)
    alpha = torch.cat([twn_codes(m.weight)[1] for n, m in modules if n.endswith("b_proj")])
    assert math.isclose(stats["qat/alpha/linear_attn.b_proj"], alpha.mean().item(), rel_tol=1e-6)
    assert "qat/alpha/layer_00" not in stats and "qat/alpha/all" not in stats


def test_without_previous_codes_no_flip_rate():
    model = _model()
    modules = [(n, m) for n, m in model.named_modules() if type(m).__name__ == "TernaryLinear"]
    stats = ternary_stats(modules, None, fsdp_group=None)
    assert "qat/zero_frac/all" in stats and not any(k.startswith("qat/flip_rate/") for k in stats)


def test_callback_schedule():
    model = _model()
    trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(qat_stats=SimpleNamespace(enable=True, interval=5))),
        model=SimpleNamespace(unwrapped_module=model),
        step_env_metrics={},
    )
    callback = QATStatsCallback(trainer)
    logged = {}
    for step in range(1, 11):
        trainer.step_env_metrics = {}
        with torch.no_grad():  # an "optimizer step"
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        callback.on_step_end(SimpleNamespace(global_step=step))
        logged[step] = dict(trainer.step_env_metrics)
    assert set(logged[1]) and not any(k.startswith("qat/flip_rate/") for k in logged[1])  # no snapshot before step 1
    assert all(not logged[s] for s in (2, 3, 4, 6, 7, 8, 9))
    for s in (5, 10):  # snapshot at s - 1, flips over one step
        assert 0 < logged[s]["qat/flip_rate/all"] < 1


def test_disabled_without_ternary_linears():
    trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(qat_stats=SimpleNamespace(enable=True, interval=5))),
        model=SimpleNamespace(unwrapped_module=_Toy()),
        step_env_metrics={},
    )
    assert not QATStatsCallback(trainer).enabled
