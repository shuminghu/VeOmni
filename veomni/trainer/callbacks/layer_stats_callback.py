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

"""Per-decoder-layer gradient norms and activation statistics (``train.layer_stats``).

On sampled steps (step 1 and every ``interval``-th):

- ``grad_norm/layer_XX``: L2 norm of the step's accumulated, *pre-clip* gradient of each decoder layer (the same
  gradients ``training/grad_norm`` measures, so the per-group norms add up to it in quadrature). Parameters
  outside the decoder layers are grouped by module path (``grad_norm/model.embed_tokens``, ``grad_norm/lm_head``, ...).
- ``act_rms/layer_XX`` and ``act_absmax/layer_XX``: RMS and max ``|x|`` of each decoder layer's output hidden states
  (the residual stream after that layer) over every token of the step, all micro-batches and ranks.

Decoder layers are the modules named ``...layers.<i>`` (the HF convention every VeOmni text model follows).
Gradients are read through :meth:`VeOmniModelRuntime.register_pre_clip_grad_hook`, so every trainer that clips
through the model runtime is covered; activations through forward hooks that skip activation-checkpoint recompute.
"""

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from ...utils import logging
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import BaseTrainer


logger = logging.get_logger(__name__)

_LAYER_MODULE_RE = re.compile(r"(?:.*\.)?layers\.(\d+)")
_LAYER_PARAM_RE = re.compile(r"(?:^|.*\.)layers\.(\d+)\.")


def layer_key(index: int) -> str:
    return f"layer_{index:02d}"


def param_group_key(param_name: str) -> str:
    """``layer_XX`` for a decoder-layer parameter, else the owning module's path."""
    match = _LAYER_PARAM_RE.match(param_name)
    if match:
        return layer_key(int(match.group(1)))
    return param_name.rsplit(".", 1)[0] if "." in param_name else param_name


def find_decoder_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """``[(layer_XX, module)]`` for every module named ``...layers.<i>``, in layer order."""
    layers = []
    for name, module in model.named_modules():
        match = _LAYER_MODULE_RE.fullmatch(name)
        if match:
            layers.append((int(match.group(1)), module))
    return [(layer_key(i), module) for i, module in sorted(layers, key=lambda item: item[0])]


@torch.no_grad()
def grouped_grad_norms(model: nn.Module, fsdp_group: Optional[dist.ProcessGroup]) -> Dict[str, float]:
    """L2 norm of the gradients of each :func:`param_group_key` group.

    FSDP2 (DTensor) gradients are shards: their squared sums are added over ``fsdp_group``. Plain tensors are full
    replicas and stay local. One all-reduce for all groups.
    """
    sharded: Dict[str, List[torch.Tensor]] = defaultdict(list)
    replicated: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        if isinstance(grad, DTensor):
            sharded[param_group_key(name)].append(grad.to_local())
        else:
            replicated[param_group_key(name)].append(grad)
    keys = sorted(set(sharded) | set(replicated))
    if not keys:
        return {}

    device = next(t for group in (sharded, replicated) for ts in group.values() for t in ts).device

    def squared_sums(groups: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
        sums = torch.zeros(len(keys), dtype=torch.float32, device=device)
        for i, key in enumerate(keys):
            tensors = [t for t in groups.get(key, ()) if t.numel()]
            if tensors:
                sums[i] = torch.stack(torch._foreach_norm(tensors)).float().square().sum()
        return sums

    sharded_sums = squared_sums(sharded)
    if fsdp_group is not None and dist.is_initialized() and dist.get_world_size(fsdp_group) > 1:
        dist.all_reduce(sharded_sums, op=dist.ReduceOp.SUM, group=fsdp_group)
    norms = (sharded_sums + squared_sums(replicated)).sqrt().tolist()
    return dict(zip(keys, norms))


class LayerStatsCallback(Callback):
    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        config = trainer.args.train.layer_stats
        self.enabled = config.enable
        self.interval = config.interval
        self._active = False
        self._grad_metrics: Dict[str, float] = {}
        if not self.enabled:
            return
        if self.parallel_state.any_extra_parallel_enabled:
            raise NotImplementedError("train.layer_stats does not reduce expert/extra-parallel gradients yet.")

        self._layers = find_decoder_layers(trainer.model.unwrapped_module)
        if not self._layers:
            raise ValueError("train.layer_stats found no decoder layers (modules named '...layers.<i>').")
        self._hooks = [module.register_forward_hook(self._make_act_hook(key)) for key, module in self._layers]
        trainer.model.register_pre_clip_grad_hook(self._collect_grad_norms)
        self._reset_acts()
        logger.info_rank0(f"Layer stats every {self.interval} steps over {len(self._layers)} decoder layers.")

    def _reset_acts(self) -> None:
        self._act_sumsq: Dict[str, torch.Tensor] = {}
        self._act_count: Dict[str, int] = defaultdict(int)
        self._act_absmax: Dict[str, torch.Tensor] = {}

    def _make_act_hook(self, key: str):
        def hook(module: nn.Module, args: Any, output: Any) -> None:
            # A forward replayed by activation checkpointing runs inside the backward graph task: skip it, or every
            # checkpointed layer would be counted twice.
            if not self._active or torch._C._current_graph_task_id() != -1:
                return
            hidden = (output[0] if isinstance(output, (tuple, list)) else output).detach().float()
            sumsq, absmax = hidden.square().sum(), hidden.abs().amax()
            if key in self._act_sumsq:
                self._act_sumsq[key] += sumsq
                self._act_absmax[key] = torch.maximum(self._act_absmax[key], absmax)
            else:
                self._act_sumsq[key], self._act_absmax[key] = sumsq, absmax
            self._act_count[key] += hidden.numel()

        return hook

    def _collect_grad_norms(self, model: nn.Module) -> None:
        if not self._active:
            return
        fsdp_group = self.parallel_state.fsdp_group if self.parallel_state.dp_mode == "fsdp2" else None
        self._grad_metrics = {f"grad_norm/{k}": v for k, v in grouped_grad_norms(model, fsdp_group).items()}

    @torch.no_grad()
    def _reduce_acts(self) -> Dict[str, float]:
        keys = [key for key, _ in self._layers if key in self._act_sumsq]
        if not keys:
            return {}
        sums = torch.stack([self._act_sumsq[k] for k in keys])
        counts = torch.tensor([float(self._act_count[k]) for k in keys], device=sums.device, dtype=torch.float64)
        absmax = torch.stack([self._act_absmax[k] for k in keys])
        sums = sums.double()
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(absmax, op=dist.ReduceOp.MAX)
        rms = (sums / counts.clamp(min=1)).sqrt().tolist()
        metrics = {f"act_rms/{k}": v for k, v in zip(keys, rms)}
        metrics.update({f"act_absmax/{k}": v for k, v in zip(keys, absmax.tolist())})
        return metrics

    def on_step_begin(self, state: TrainerState, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        if not self.enabled:
            return
        self._active = state.global_step == 1 or state.global_step % self.interval == 0
        self._grad_metrics = {}
        self._reset_acts()

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        if not self._active:
            return
        metrics = {**self._grad_metrics, **self._reduce_acts()}
        # Runs after EnvironMeterCallback (which rebuilds step_env_metrics) and before WandbTraceCallback.
        self.trainer.step_env_metrics.update(metrics)
        self._active = False
        self._reset_acts()
