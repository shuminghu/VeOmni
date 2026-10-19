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

"""Qwen3-Hybrid (KDA + NoPE attention): layout, parameter parity, init, and packed / padded isolation."""

import math

import pytest
import torch

from tests.models.compare import ops_config_scope
from tests.tools.training_utils import make_eager_ops_config
from veomni.models.transformers.qwen3_hybrid.configuration_qwen3_hybrid import Qwen3HybridConfig
from veomni.models.transformers.qwen3_hybrid.modeling_qwen3_hybrid import (
    KimiDeltaAttention,
    Qwen3HybridAttention,
    Qwen3HybridForCausalLM,
    Qwen3HybridModel,
    _no_rope,
)
from veomni.ops.qat.ternary import TernaryLinear, apply_ternary_qat
from veomni.utils.import_utils import is_package_available


QWEN3_S = "configs/model_configs/qwen/Qwen3-S-longctx.json"
QWEN3_S_HYBRID = "configs/model_configs/qwen/Qwen3-S-hybrid-kda.json"

needs_fla_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_package_available("fla"), reason="FLA KDA kernels need CUDA + fla"
)


def _tiny_config(**overrides) -> Qwen3HybridConfig:
    kwargs = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_heads=2,
        linear_head_dim=32,
        linear_value_head_dim=32,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    kwargs.update(overrides)
    config = Qwen3HybridConfig(**kwargs)
    config._attn_implementation = "sdpa"
    return config


def test_layer_pattern_and_nope():
    config = _tiny_config(num_hidden_layers=8)
    assert config.layer_types == (["linear_attention"] * 3 + ["full_attention"]) * 2
    with ops_config_scope(make_eager_ops_config()):
        model = Qwen3HybridModel(config)
    assert model.rotary_emb is None
    for layer, layer_type in zip(model.layers, config.layer_types):
        if layer_type == "linear_attention":
            assert isinstance(layer.linear_attn, KimiDeltaAttention) and not hasattr(layer, "self_attn")
        else:
            assert isinstance(layer.self_attn, Qwen3HybridAttention) and layer.self_attn.veomni_rope is _no_rope


def test_qwen3_s_hybrid_matches_qwen3_s_shape_and_params():
    """Same depth / width / MLP as Qwen3-S; each KDA mixer within 2% of the attention mixer it replaces."""
    from transformers import AutoConfig, AutoModelForCausalLM

    import veomni.models  # noqa: F401  (registers qwen3_hybrid on the Auto classes)

    with ops_config_scope(make_eager_ops_config()), torch.device("meta"):
        base = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(QWEN3_S))
        hybrid = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(QWEN3_S_HYBRID))
    assert isinstance(hybrid, Qwen3HybridForCausalLM)
    assert len(base.model.layers) == len(hybrid.model.layers) == 12

    def numel(module):
        return sum(p.numel() for p in module.parameters())

    attn = numel(base.model.layers[0].self_attn)
    for i, layer in enumerate(hybrid.model.layers):
        if i % 4 == 3:
            assert numel(layer.self_attn) == attn
        else:
            assert abs(numel(layer.linear_attn) / attn - 1) < 0.02
        assert numel(layer.mlp) == numel(base.model.layers[i].mlp)
    assert abs(numel(hybrid) / numel(base) - 1) < 0.005


def test_kda_init():
    with ops_config_scope(make_eager_ops_config()):
        model = Qwen3HybridForCausalLM(_tiny_config())
    kda = model.model.layers[0].linear_attn
    assert torch.all((kda.A_log >= 0) & (kda.A_log <= math.log(16)))
    assert torch.all(kda.dt_bias == 0)
    bound = 1 / math.sqrt(kda.conv_kernel_size)
    for conv in (kda.q_conv1d, kda.k_conv1d, kda.v_conv1d):
        assert conv.weight.abs().max() <= bound and conv.weight.std() > 0.5 * bound / math.sqrt(3)
    assert torch.all(kda.g_proj[1].bias == 0) and torch.all(kda.o_norm.weight == 1)


def test_ternary_qat_covers_kda_projections():
    with ops_config_scope(make_eager_ops_config()):
        model = Qwen3HybridForCausalLM(_tiny_config())
    names = apply_ternary_qat(model)
    kda = model.model.layers[0].linear_attn
    for linear in (kda.q_proj, kda.k_proj, kda.v_proj, kda.o_proj, kda.b_proj, kda.f_proj[0], kda.g_proj[1]):
        assert isinstance(linear, TernaryLinear)
    assert "lm_head" not in names


def test_eager_forward_backward_cpu():
    """The whole stack runs on the eager reference ops (dense batch, no packing)."""
    with ops_config_scope(make_eager_ops_config()):
        torch.manual_seed(0)
        model = Qwen3HybridForCausalLM(_tiny_config()).float()
        input_ids = torch.randint(0, 128, (2, 12))
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
    assert torch.isfinite(out.loss)
    assert model.model.layers[0].linear_attn.A_log.grad is not None


def _assert_close_to_scale(actual, expected, rtol, name=""):
    """Max error relative to the reference's max magnitude: a state leak across a boundary is O(1) here, while
    kernel reassociation (Triton ``tl.dot`` keeps TF32 for FP32 inputs) stays around 1e-4."""
    err = (actual - expected).abs().max().item()
    scale = expected.abs().max().item()
    print(f"{name}: rel err {err / max(scale, 1e-12):.2e}")
    assert err <= rtol * max(scale, 1e-12), f"{name}: max err {err:.3e} vs scale {scale:.3e}"


@pytest.fixture
def no_tf32():
    """Exact-isolation comparisons: TF32 matmul rounding (~1e-3) would otherwise swamp the comparison."""
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def _gpu_ops():
    return make_eager_ops_config(
        attn_implementation="sdpa",
        kda_implementation="fla",
        causal_conv1d_implementation="fla",
        rms_norm_gated_implementation="fla",
    )


@needs_fla_gpu
def test_packed_matches_separate_outputs_and_gradients(no_tf32):
    """cu_seq_lens_q isolates packed samples in both the KDA layers and the (SDPA) attention layers."""
    with ops_config_scope(_gpu_ops()):
        torch.manual_seed(0)
        config = _tiny_config()
        packed_model = Qwen3HybridModel(config).cuda()
        separate_model = Qwen3HybridModel(config).cuda()
        separate_model.load_state_dict(packed_model.state_dict())

        lengths = (70, 3, 150)
        input_ids = torch.randint(0, config.vocab_size, (1, sum(lengths)), device="cuda")
        position_ids = torch.cat([torch.arange(n) for n in lengths]).view(1, -1).cuda()
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
        packed = packed_model(input_ids=input_ids, position_ids=position_ids, cu_seq_lens_q=cu).last_hidden_state
        separate = torch.cat(
            [separate_model(input_ids=seg).last_hidden_state for seg in input_ids.split(lengths, dim=1)], dim=1
        )
        _assert_close_to_scale(packed, separate, 1e-3, "hidden")

        weights = torch.randn_like(packed)
        (packed * weights).sum().backward()
        (separate * weights).sum().backward()
        for (name, p), q in zip(packed_model.named_parameters(), separate_model.parameters()):
            _assert_close_to_scale(p.grad, q.grad, 2e-3, name)


@needs_fla_gpu
@pytest.mark.parametrize("max_len", [40, 150], ids=["fused_recurrent", "chunk"])
def test_padded_batch_matches_unpadded_in_eval(max_len, no_tf32):
    """A right-padded batch with a 2-D attention mask equals each sequence run alone (eval / scoring path)."""
    with ops_config_scope(_gpu_ops()):
        torch.manual_seed(0)
        config = _tiny_config()
        model = Qwen3HybridModel(config).cuda().eval()
        lengths = [max_len, max_len // 2, 7]
        input_ids = torch.randint(0, config.vocab_size, (len(lengths), max_len), device="cuda")
        mask = torch.zeros_like(input_ids)
        for i, n in enumerate(lengths):
            mask[i, :n] = 1
        with torch.no_grad():
            batched = model(input_ids=input_ids, attention_mask=mask).last_hidden_state
            for i, n in enumerate(lengths):
                alone = model(input_ids=input_ids[i : i + 1, :n]).last_hidden_state
                # A short sequence alone takes the fused-recurrent kernel while the batch takes the chunked one.
                same_kernel = (n <= 64) == (max_len <= 64)
                _assert_close_to_scale(batched[i : i + 1, :n], alone, 1e-3 if same_kernel else 5e-3, f"seq{i}")


def _left_padded(lengths, vocab_size, device="cuda"):
    max_len = max(lengths)
    input_ids = torch.zeros(len(lengths), max_len, dtype=torch.long, device=device)
    mask = torch.zeros_like(input_ids)
    for i, n in enumerate(lengths):
        input_ids[i, max_len - n :] = torch.randint(1, vocab_size, (n,), device=device)
        mask[i, max_len - n :] = 1
    return input_ids, mask


@needs_fla_gpu
@pytest.mark.parametrize(
    "lengths,rtol",
    # Short prompts: prefill, decode and the reference all run the fused-recurrent kernel (exact up to FP32).
    # Long prompts: the reference runs the chunked kernel over the whole sequence while decode steps are recurrent,
    # which differ by ~1e-3 (see test_padded_batch_matches_unpadded_in_eval).
    [((5, 30, 17), 1e-5), ((3, 130, 90), 5e-3)],
    ids=["recurrent_prefill", "chunk_prefill"],
)
def test_cached_generate_matches_full_forward(lengths, rtol, no_tf32):
    """Greedy ``generate`` (hybrid KV + KDA conv / recurrent cache) reproduces full no-cache forwards at every step
    on a left-padded batch."""
    with ops_config_scope(_gpu_ops()):
        torch.manual_seed(0)
        config = _tiny_config(pad_token_id=0, eos_token_id=None, bos_token_id=None)
        assert config.use_cache  # generate inherits it: False would silently recompute the prefix every step
        model = Qwen3HybridForCausalLM(config).cuda().eval()
        input_ids, mask = _left_padded(lengths, config.vocab_size)
        new_tokens = 8
        calls = []

        def record(module, args, kwargs):
            pkv = kwargs.get("past_key_values")
            calls.append((kwargs["input_ids"].shape[1], pkv.get_seq_length() if pkv is not None else None))

        handle = model.model.register_forward_pre_hook(record, with_kwargs=True)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=mask,
                max_new_tokens=new_tokens,
                min_new_tokens=new_tokens,
                do_sample=False,
                pad_token_id=0,
                output_scores=True,
                return_dict_in_generate=True,
            )
            handle.remove()
            assert out.sequences.shape[1] == input_ids.shape[1] + new_tokens
            # The cache is really used: one prefill of the prompt, then one token per step on top of the cache.
            prompt_len = input_ids.shape[1]
            assert calls[0] == (prompt_len, 0), calls[0]
            assert calls[1:] == [(1, prompt_len + i) for i in range(new_tokens - 1)], calls
            full_mask = torch.cat([mask, mask.new_ones(len(lengths), new_tokens)], dim=1)
            # Teacher-force the reference on generate's own tokens, so near-ties cannot make the two diverge.
            ref = model(input_ids=out.sequences, attention_mask=full_mask, use_cache=False).logits.float()
        for step, scores in enumerate(out.scores):
            position = input_ids.shape[1] - 1 + step
            _assert_close_to_scale(scores.float(), ref[:, position], rtol, f"step{step}")


@needs_fla_gpu
def test_left_padding_matches_unpadded_prefill(no_tf32):
    """Zeroed left padding leaves the KDA state at zero: padded rows equal the same prompts run alone."""
    with ops_config_scope(_gpu_ops()):
        torch.manual_seed(0)
        config = _tiny_config()
        model = Qwen3HybridModel(config).cuda().eval()
        lengths = (150, 70, 9)
        input_ids, mask = _left_padded(lengths, config.vocab_size)
        with torch.no_grad():
            batched = model(input_ids=input_ids, attention_mask=mask).last_hidden_state
            for i, n in enumerate(lengths):
                alone = model(input_ids=input_ids[i : i + 1, -n:]).last_hidden_state
                same_kernel = (n <= 64) == (input_ids.shape[1] <= 64)
                _assert_close_to_scale(batched[i : i + 1, -n:], alone, 1e-3 if same_kernel else 5e-3, f"seq{i}")


@needs_fla_gpu
def test_standalone_model_uses_fla_kernels():
    """Without an installed ops config (eval / generation scripts calling from_pretrained), the KDA layer must not
    fall back to the token-by-token eager reference."""
    with ops_config_scope(None):
        kda = KimiDeltaAttention(_tiny_config(), 0)
    assert kda.veomni_chunk_kda.impl == "fla"
    assert kda.veomni_fused_recurrent_kda.impl == "fla"
    assert kda.veomni_causal_conv1d.impl == "fla"
    assert kda.o_norm.veomni_rms_norm_gated.impl == "fla"
