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

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from veomni.arguments import InitConfig
from veomni.models.initialization import apply_mup_init, spectral_std


def _tiny_qwen3(tie: bool) -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=4096, hidden_size=256, intermediate_size=1024, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, tie_word_embeddings=tie,
    )  # fmt: skip
    return Qwen3ForCausalLM(config)


def test_mup_init_stds():
    torch.manual_seed(0)
    model = _tiny_qwen3(tie=False)
    norm_before = model.model.norm.weight.detach().clone()
    counts = apply_mup_init(model, InitConfig(scheme="mup", embedding_std=1.0, hidden_std=1.0, unembedding_std=2.0))

    assert counts["hidden"] == 2 * 7  # q, k, v, o, gate, up, down per layer
    assert model.model.embed_tokens.weight.std().item() == pytest.approx(1.0, rel=0.02)
    assert model.lm_head.weight.std().item() == pytest.approx(2.0 / 256, rel=0.02)
    layer = model.model.layers[0]
    for linear in (layer.self_attn.q_proj, layer.mlp.up_proj, layer.mlp.down_proj):
        fan_out, fan_in = linear.weight.shape
        assert linear.weight.std().item() == pytest.approx(spectral_std(fan_out, fan_in), rel=0.05)
    assert spectral_std(256, 1024) < 1 / 1024**0.5  # down_proj shrinks below 1/sqrt(fan_in)
    torch.testing.assert_close(model.model.norm.weight, norm_before)


def test_mup_init_rejects_tied_embeddings():
    with pytest.raises(ValueError, match="untied"):
        apply_mup_init(_tiny_qwen3(tie=True), InitConfig(scheme="mup"))
