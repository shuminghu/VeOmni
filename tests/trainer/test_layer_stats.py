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

"""Per-layer stats helpers: grouping, norms that add up to the global grad norm, and checkpoint-safe hooks."""

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from veomni.trainer.callbacks.layer_stats_callback import (
    LayerStatsCallback,
    find_decoder_layers,
    grouped_grad_norms,
    param_group_key,
)


class _Layer(nn.Module):
    def __init__(self, width, use_checkpoint):
        super().__init__()
        self.proj = nn.Linear(width, width)
        self.use_checkpoint = use_checkpoint

    def _body(self, x):
        return x + torch.tanh(self.proj(x))

    def __call__(self, x):
        if self.use_checkpoint:
            return checkpoint(super().__call__, x, use_reentrant=False)
        return super().__call__(x)

    def forward(self, x):
        return self._body(x)


class _Toy(nn.Module):
    def __init__(self, width=8, num_layers=3, use_checkpoint=False):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(16, width)
        self.model.layers = nn.ModuleList(_Layer(width, use_checkpoint) for _ in range(num_layers))
        self.lm_head = nn.Linear(width, 16, bias=False)

    def forward(self, ids):
        h = self.model.embed_tokens(ids)
        for layer in self.model.layers:
            h = layer(h)
        return self.lm_head(h)


def test_grouping():
    assert param_group_key("model.layers.3.self_attn.q_proj.weight") == "layer_03"
    assert param_group_key("model.layers.11.mlp.down_proj.weight") == "layer_11"
    assert param_group_key("model.embed_tokens.weight") == "model.embed_tokens"
    assert param_group_key("lm_head.weight") == "lm_head"
    assert [k for k, _ in find_decoder_layers(_Toy(num_layers=12))] == [f"layer_{i:02d}" for i in range(12)]


def test_group_norms_add_up_to_global_norm():
    torch.manual_seed(0)
    model = _Toy()
    model(torch.randint(0, 16, (2, 5))).square().mean().backward()
    norms = grouped_grad_norms(model, fsdp_group=None)
    assert set(norms) == {"layer_00", "layer_01", "layer_02", "model.embed_tokens", "lm_head"}
    total = torch.nn.utils.get_total_norm([p.grad for p in model.parameters()]).item()
    assert math.isclose(math.sqrt(sum(v * v for v in norms.values())), total, rel_tol=1e-5)
    layer1 = torch.nn.utils.get_total_norm([p.grad for p in model.model.layers[1].parameters()]).item()
    assert math.isclose(norms["layer_01"], layer1, rel_tol=1e-5)


def _callback_for(model, interval=1):
    runtime_hooks = []
    trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(layer_stats=SimpleNamespace(enable=True, interval=interval))),
        model=SimpleNamespace(unwrapped_module=model, register_pre_clip_grad_hook=runtime_hooks.append),
        step_env_metrics={},
    )
    callback = LayerStatsCallback(trainer)
    return callback, trainer, runtime_hooks


def test_activation_stats_count_each_forward_once_under_checkpointing():
    torch.manual_seed(0)
    plain, ckpt = _Toy(use_checkpoint=False), _Toy(use_checkpoint=True)
    ckpt.load_state_dict(plain.state_dict())
    ids = torch.randint(0, 16, (2, 5))
    results = []
    for model in (plain, ckpt):
        callback, trainer, hooks = _callback_for(model)
        state = SimpleNamespace(global_step=1)
        callback.on_step_begin(state)
        for _ in range(2):  # two micro-batches
            model(ids).square().mean().backward()
        hooks[0](model)  # what VeOmniModelRuntime.clip_grad_norm does
        callback.on_step_end(state)
        results.append(trainer.step_env_metrics)
    plain_metrics, ckpt_metrics = results
    assert plain_metrics.keys() == ckpt_metrics.keys()
    assert {
        "act_rms/layer_02",
        "act_absmax/layer_00",
        "grad_norm/layer_01",
        "grad_norm/lm_head",
    } <= plain_metrics.keys()
    for key, value in plain_metrics.items():
        assert math.isclose(value, ckpt_metrics[key], rel_tol=1e-5), key
    # The RMS is over both micro-batches' tokens: recompute is not double-counted.
    with torch.no_grad():
        h = plain.model.embed_tokens(ids)
        for layer in plain.model.layers:
            h = layer(h)
    assert math.isclose(plain_metrics["act_rms/layer_02"], h.square().mean().sqrt().item(), rel_tol=1e-5)


def test_inactive_steps_log_nothing():
    model = _Toy()
    callback, trainer, hooks = _callback_for(model, interval=10)
    state = SimpleNamespace(global_step=3)
    callback.on_step_begin(state)
    model(torch.randint(0, 16, (1, 4))).sum().backward()
    hooks[0](model)
    callback.on_step_end(state)
    assert trainer.step_env_metrics == {}
