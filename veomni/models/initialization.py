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

"""Width-aware (muP) parameter initialization for from-scratch training.

Pairs with Muon's ``adjust_lr_fn="spectral"`` update scaling (``sqrt(d_out / d_in)``):

- input embedding: ``N(0, embedding_std^2)`` -- width independent (one-hot fan-in);
- hidden ``nn.Linear`` ``[d_out, d_in]``: spectral init, std ``hidden_std / sqrt(d_in) * min(1, sqrt(d_out / d_in))``
  (Yang, Simon & Bernstein, "A Spectral Condition for Feature Learning"); equals ``1/sqrt(fan_in)`` for
  square / widening layers;
- output embedding (``lm_head``): ``N(0, (unembedding_std / d_in)^2)`` so initial logits vanish with width;
- biases zero; norm weights and non-``Linear`` parameters keep the model's own init.

Works on plain and FSDP2-sharded (DTensor) parameters: shapes are global and in-place RNG ops on DTensors
draw distinct values per shard.
"""

import math
from typing import TYPE_CHECKING, Dict

import torch
import torch.nn as nn

from ..utils import logging


if TYPE_CHECKING:
    from ..arguments import InitConfig


logger = logging.get_logger(__name__)


def spectral_std(fan_out: int, fan_in: int) -> float:
    return min(1.0, math.sqrt(fan_out / fan_in)) / math.sqrt(fan_in)


@torch.no_grad()
def apply_mup_init(model: nn.Module, config: "InitConfig") -> Dict[str, int]:
    """Re-initialize ``model`` in place. Returns how many tensors each rule touched."""
    embedding = model.get_input_embeddings()
    unembedding = model.get_output_embeddings()
    if embedding is None or unembedding is None:
        raise ValueError("muP init needs a model with both input and output embeddings.")
    if embedding.weight is unembedding.weight:
        raise ValueError("muP init needs untied embeddings: set tie_word_embeddings=false in the model config.")

    counts = {"embedding": 0, "hidden": 0, "unembedding": 0, "bias": 0}
    embedding.weight.normal_(mean=0.0, std=config.embedding_std)
    counts["embedding"] += 1

    fan_in = unembedding.weight.shape[1]
    unembedding.weight.normal_(mean=0.0, std=config.unembedding_std / fan_in)
    counts["unembedding"] += 1

    for module in model.modules():
        if module is unembedding or not isinstance(module, nn.Linear):
            continue
        fan_out, fan_in = module.weight.shape
        module.weight.normal_(mean=0.0, std=config.hidden_std * spectral_std(fan_out, fan_in))
        counts["hidden"] += 1
        if module.bias is not None:
            module.bias.zero_()
            counts["bias"] += 1
    if unembedding.bias is not None:
        unembedding.bias.zero_()
        counts["bias"] += 1

    logger.info_rank0(
        f"muP init: embedding_std={config.embedding_std}, hidden_std={config.hidden_std} (spectral), "
        f"unembedding_std={config.unembedding_std}/fan_in; tensors re-initialized: {counts}"
    )
    return counts
