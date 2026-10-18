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

"""Ternary (TWN) weight QAT: quantizer numerics, straight-through gradient, and the model-level class swap."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from veomni.arguments import QATConfig
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models import build_foundation_model
from veomni.ops.kernels.swiglu_mlp import has_plain_linear_projections
from veomni.ops.qat import (
    TernaryLinear,
    apply_ternary_qat,
    ternarize_linears_,
    ternary_fake_quant_weight,
    twn_ternarize,
)


def _reference_twn(w: torch.Tensor) -> torch.Tensor:
    """The TWN heuristic as originally specified (per row, threshold 0.7 * mean|w|)."""
    w_fp = w.float()
    abs_w = w_fp.abs()
    mask = (abs_w > abs_w.mean(-1, keepdim=True) * 0.7).float()
    alpha = (abs_w * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1.0)
    return (w_fp.sign() * mask * alpha).to(w.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_twn_matches_reference_and_is_ternary(dtype):
    torch.manual_seed(0)
    w = torch.randn(48, 256, dtype=dtype)
    q = twn_ternarize(w)
    torch.testing.assert_close(q, _reference_twn(w), rtol=0, atol=0)
    for row in q.float():
        assert len(row.abs().unique()) <= 2  # {0, alpha}
    # Gaussian weights: ~58% survive the 0.7 * mean|w| threshold.
    assert (q != 0).float().mean().item() == pytest.approx(0.58, abs=0.03)


def test_twn_group_size():
    torch.manual_seed(0)
    w = torch.randn(8, 256)
    q = twn_ternarize(w, group_size=64)
    torch.testing.assert_close(q, _reference_twn(w.reshape(-1, 64)).reshape(8, 256))
    with pytest.raises(ValueError, match="does not divide"):
        twn_ternarize(w, group_size=48)


def test_straight_through_gradient():
    torch.manual_seed(0)
    w = torch.randn(32, 64, requires_grad=True)
    x = torch.randn(5, 64)
    out = F.linear(x, ternary_fake_quant_weight(w))
    torch.testing.assert_close(out, F.linear(x, twn_ternarize(w.detach())))
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    # STE: d out / d w is computed as if the ternarized weight were w itself.
    torch.testing.assert_close(w.grad, grad_out.t() @ x)


def _tiny_qwen3():
    config = Qwen3Config(
        vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, tie_word_embeddings=False,
        hidden_act="silu", architectures=["Qwen3ForCausalLM"],
    )  # fmt: skip
    ops = OpsImplementationConfig(
        attn_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
    )
    torch.manual_seed(0)
    return build_foundation_model(config, torch_dtype="float32", init_device="cpu", ops_implementation=ops)


def test_apply_ternary_qat_matches_pre_ternarized_model():
    """QAT forward must equal a model whose weights were ternarized up front. This catches any fused path that
    reads a projection's raw ``weight`` instead of calling its forward (the swiglu_mlp op did)."""
    model = _tiny_qwen3()
    reference = copy.deepcopy(model)
    keys = list(model.state_dict())

    names = apply_ternary_qat(model)
    assert len(names) == 2 * 7  # q, k, v, o, gate, up, down per layer
    assert type(model.lm_head) is nn.Linear
    assert isinstance(model.model.layers[0].mlp.gate_proj, TernaryLinear)
    assert isinstance(model.model.layers[0].mlp.gate_proj, nn.Linear)  # init / optimizer grouping still apply
    assert not has_plain_linear_projections(model.model.layers[0].mlp)
    assert list(model.state_dict()) == keys

    assert ternarize_linears_(reference) == names
    ids = torch.randint(0, 512, (2, 16))
    torch.testing.assert_close(model(input_ids=ids).logits, reference(input_ids=ids).logits)

    model(input_ids=ids, labels=ids).loss.backward()
    for name in names:
        assert model.get_submodule(name).weight.grad.abs().sum() > 0, name


def test_apply_ternary_qat_exclude():
    model = _tiny_qwen3()
    names = apply_ternary_qat(model, group_size=32, exclude=[r"self_attn\."])
    assert len(names) == 2 * 3
    assert type(model.model.layers[0].self_attn.q_proj) is nn.Linear
    assert model.model.layers[1].mlp.down_proj.ternary_group_size == 32


def test_qat_config_validation():
    assert QATConfig().weight == "none"
    with pytest.raises(ValueError, match="group_size"):
        QATConfig(weight="ternary_twn", group_size=-1)
