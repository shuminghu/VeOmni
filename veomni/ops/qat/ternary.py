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

"""Ternary (1.58-bit) weight fake quantization for quantization-aware training.

Each weight row (or each ``group_size`` slice of a row) becomes ``alpha * {-1, 0, +1}`` with the Ternary Weight
Networks heuristic (Li et al., 2016): entries with ``|w| <= threshold * mean|w|`` go to 0, and ``alpha`` is the mean
``|w|`` of the entries that survive. With the default threshold 0.7 this is the closed-form approximation of the
L2-optimal ternary projection for Gaussian weights.

Post-training ternarization destroys a bf16-trained model (about 44% relative weight error), so ternary weights
have to be trained. :class:`TernaryLinear` keeps the latent full-precision weight as its parameter and runs the
forward with the ternarized copy. The backward is a straight-through estimator: the gradient of the ternarized
weight is passed to the latent weight unchanged. Only weights are quantized (W1.58A16); activations stay in the
compute dtype.

Unlike the FP8 recipe in this package, this is pure PyTorch and model agnostic: :func:`apply_ternary_qat` swaps
the class of every hidden ``nn.Linear`` of any model, so no per-model call sites are needed. Parameters,
state-dict keys and ``isinstance(m, nn.Linear)`` are unchanged, so init, optimizer grouping, FSDP2 sharding and
checkpointing see an ordinary model. Checkpoints therefore hold the latent weights; apply :func:`twn_ternarize`
to obtain the deployed ternary model.
"""

import re
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...utils import logging


logger = logging.get_logger(__name__)

__all__ = [
    "TernaryLinear",
    "apply_ternary_qat",
    "ternarize_linears_",
    "ternary_fake_quant_weight",
    "twn_ternarize",
]


def twn_ternarize(weight: torch.Tensor, group_size: int = 0, threshold: float = 0.7) -> torch.Tensor:
    """TWN projection of ``weight`` (``[..., in]``) onto ``alpha * {-1, 0, 1}``, computed in FP32.

    Args:
        weight: Weight whose last dimension is the reduction (input) dimension, as in ``nn.Linear``.
        group_size: 0 gives one ``(threshold, alpha)`` pair per row (per output channel); ``g > 0`` gives one per
            contiguous ``g`` entries of a row. ``g`` must divide the last dimension.
        threshold: Zero out entries with ``|w| <= threshold * mean|w|`` of their group.
    """
    shape = weight.shape
    w = weight.float()
    if group_size:
        if shape[-1] % group_size:
            raise ValueError(f"group_size={group_size} does not divide the input dimension {shape[-1]}.")
        w = w.reshape(-1, group_size)
    abs_w = w.abs()
    mask = (abs_w > threshold * abs_w.mean(-1, keepdim=True)).float()
    alpha = (abs_w * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1.0)
    return (w.sign() * mask * alpha).reshape(shape).to(weight.dtype)


def ternary_fake_quant_weight(weight: torch.Tensor, group_size: int = 0, threshold: float = 0.7) -> torch.Tensor:
    """Ternarized ``weight`` in the forward, identity (straight-through) in the backward."""
    return weight + (twn_ternarize(weight.detach(), group_size, threshold) - weight).detach()


class TernaryLinear(nn.Linear):
    """``nn.Linear`` whose forward uses the TWN-ternarized weight (straight-through gradient to the latent weight).

    Created by :func:`apply_ternary_qat` through a class swap, never constructed directly, so it carries no state
    beyond the two quantizer settings set there.
    """

    ternary_group_size: int = 0
    ternary_threshold: float = 0.7

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = ternary_fake_quant_weight(self.weight, self.ternary_group_size, self.ternary_threshold)
        return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, ternary(group_size={self.ternary_group_size}, threshold={self.ternary_threshold})"


def _hidden_linears(model: nn.Module, exclude: Optional[List[str]]):
    """``(name, module)`` of every plain ``nn.Linear`` except the output embedding and ``exclude`` regex matches."""
    output_embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    patterns = [re.compile(p) for p in exclude or []]
    for name, module in model.named_modules():
        # Exact type: subclasses (LoRA, other QAT wrappers) own their forward and are left alone.
        if type(module) is not nn.Linear or module is output_embeddings:
            continue
        if any(p.search(name) for p in patterns):
            continue
        yield name, module


def apply_ternary_qat(
    model: nn.Module, group_size: int = 0, threshold: float = 0.7, exclude: Optional[List[str]] = None
) -> List[str]:
    """Turn every hidden ``nn.Linear`` of ``model`` into a :class:`TernaryLinear`, in place.

    The output embedding (``lm_head``) and input embeddings are never quantized; ``exclude`` adds module-name regexes
    to skip. Works on meta, materialized and FSDP2-sharded models: only the module class changes, and the forward
    runs on whatever (unsharded) weight FSDP2 has gathered. Returns the names of the converted modules.
    """
    names = []
    for name, module in _hidden_linears(model, exclude):
        module.__class__ = TernaryLinear
        module.ternary_group_size = group_size
        module.ternary_threshold = threshold
        names.append(name)
    if not names:
        raise ValueError("apply_ternary_qat found no hidden nn.Linear to quantize.")
    logger.info_rank0(
        f"Ternary QAT (TWN, group_size={group_size or 'row'}, threshold={threshold}): "
        f"{len(names)} linears, e.g. {names[0]}"
    )
    return names


@torch.no_grad()
def ternarize_linears_(
    model: nn.Module, group_size: int = 0, threshold: float = 0.7, exclude: Optional[List[str]] = None
) -> List[str]:
    """Overwrite the weight of every hidden ``nn.Linear`` with its ternarized value (deployment / evaluation of a
    ternary-QAT checkpoint, or post-training ternarization). Selects the same modules as :func:`apply_ternary_qat`."""
    names = []
    for name, module in _hidden_linears(model, exclude):
        module.weight.copy_(twn_ternarize(module.weight, group_size, threshold))
        names.append(name)
    return names
