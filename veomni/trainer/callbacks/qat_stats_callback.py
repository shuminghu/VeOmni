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

"""Ternary-QAT weight statistics (``train.qat_stats``).

On step 1 and every ``interval``-th step, for the :class:`~veomni.ops.qat.ternary.TernaryLinear` weights:

- ``qat/flip_rate/<group>``: fraction of ternary codes that changed during that step's optimizer update (codes are
  snapshotted at the end of the previous step). High flip rates mean latent weights oscillating around a threshold.
- ``qat/zero_frac/<group>``: fraction of zero codes (TWN zeroes ``|w| <= threshold * mean|w|``).
- ``qat/alpha/<kind>``: mean per-group scale.

Groups are the projection type with the layer index removed (``linear_attn.b_proj``, ``mlp.down_proj``, ...), the
decoder layer (``layer_XX``) and ``all``. Codes are computed from each rank's local weight shard exactly as the
forward does (:func:`~veomni.ops.qat.ternary.twn_codes`); FSDP2 shards whole rows, so per-row groups are exact.
"""

import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from ...ops.qat.ternary import TernaryLinear, twn_codes
from ...utils import logging
from .base import Callback, TrainerState
from .layer_stats_callback import layer_key


if TYPE_CHECKING:
    from ..base import BaseTrainer


logger = logging.get_logger(__name__)

_LAYER_PREFIX_RE = re.compile(r"^(?:.*\.)?layers\.(\d+)\.")
# Per group: zeros, codes, flips, compared codes, alpha sum, alpha count.
_FIELDS = 6


def module_groups(name: str) -> Tuple[str, Optional[str]]:
    """``(kind, layer)`` of a module name: the projection type without its layer prefix, and ``layer_XX``."""
    match = _LAYER_PREFIX_RE.match(name)
    if match is None:
        return name, None
    return name[match.end() :], layer_key(int(match.group(1)))


def ternary_modules(model: nn.Module) -> List[Tuple[str, TernaryLinear]]:
    return [(name, m) for name, m in model.named_modules() if isinstance(m, TernaryLinear)]


def _local(weight: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if isinstance(weight, DTensor):
        return weight.to_local(), True
    return weight, False


@torch.no_grad()
def module_codes(module: TernaryLinear) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """``(codes, alpha, sharded)`` of the module's local weight shard."""
    local, sharded = _local(module.weight)
    codes, alpha = twn_codes(local.detach(), module.ternary_group_size, module.ternary_threshold)
    return codes, alpha, sharded


@torch.no_grad()
def ternary_stats(
    modules: List[Tuple[str, TernaryLinear]],
    previous: Optional[Dict[str, torch.Tensor]],
    fsdp_group: Optional[dist.ProcessGroup],
) -> Dict[str, float]:
    """Zero fraction, mean alpha and (with ``previous`` codes) flip rate, reduced over the FSDP group."""
    sums: Dict[Tuple[str, bool], torch.Tensor] = {}
    device = None
    for name, module in modules:
        codes, alpha, sharded = module_codes(module)
        if codes.numel() == 0:
            continue
        device = codes.device
        row = torch.zeros(_FIELDS, dtype=torch.float64, device=device)
        row[0] = (codes == 0).sum()
        row[1] = codes.numel()
        if previous is not None and name in previous and previous[name].shape == codes.shape:
            row[2] = (codes != previous[name]).sum()
            row[3] = codes.numel()
        row[4] = alpha.double().sum()
        row[5] = alpha.numel()
        kind, layer = module_groups(name)
        for group in (kind, layer, "all"):
            if group is None:
                continue
            key = (group, sharded)
            sums[key] = sums[key] + row if key in sums else row.clone()
    if device is None:
        return {}

    groups = sorted({g for g, _ in sums})
    zero = torch.zeros(_FIELDS, dtype=torch.float64, device=device)
    sharded_t = torch.stack([sums.get((g, True), zero) for g in groups])
    replicated_t = torch.stack([sums.get((g, False), zero) for g in groups])
    if fsdp_group is not None and dist.is_initialized() and dist.get_world_size(fsdp_group) > 1:
        dist.all_reduce(sharded_t, op=dist.ReduceOp.SUM, group=fsdp_group)
    total = (sharded_t + replicated_t).tolist()

    metrics = {}
    for group, (zeros, n, flips, compared, alpha_sum, alpha_n) in zip(groups, total):
        metrics[f"qat/zero_frac/{group}"] = zeros / max(n, 1)
        if compared:
            metrics[f"qat/flip_rate/{group}"] = flips / compared
        if not group.startswith("layer_") and group != "all":
            metrics[f"qat/alpha/{group}"] = alpha_sum / max(alpha_n, 1)
    return metrics


class QATStatsCallback(Callback):
    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        config = trainer.args.train.qat_stats
        self.enabled = config.enable
        self.interval = config.interval
        self._previous: Optional[Dict[str, torch.Tensor]] = None
        if not self.enabled:
            return
        self._modules = ternary_modules(trainer.model.unwrapped_module)
        if not self._modules:
            logger.warning_rank0("train.qat_stats is enabled but the model has no ternary-QAT linears; disabling.")
            self.enabled = False
            return
        logger.info_rank0(f"QAT stats every {self.interval} steps over {len(self._modules)} ternary linears.")

    def _snapshot(self) -> None:
        self._previous = {name: module_codes(m)[0].clone() for name, m in self._modules}

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        # Runs after the optimizer step, so the codes here are the ones the next forward uses.
        if not self.enabled:
            return
        step = state.global_step
        if step == 1 or step % self.interval == 0:
            fsdp_group = self.parallel_state.fsdp_group if self.parallel_state.dp_mode == "fsdp2" else None
            metrics = ternary_stats(self._modules, self._previous, fsdp_group)
            # After EnvironMeterCallback (which rebuilds step_env_metrics), before WandbTraceCallback.
            self.trainer.step_env_metrics.update(metrics)
            self._previous = None
        if (step + 1) % self.interval == 0:
            self._snapshot()


__all__ = ["QATStatsCallback", "module_groups", "ternary_stats", "ternary_modules"]
