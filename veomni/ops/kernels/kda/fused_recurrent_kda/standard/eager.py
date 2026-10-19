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

"""standard fused_recurrent_kda eager math (FP32 token recurrence)."""

from __future__ import annotations

from torch import Tensor

from ....gated_delta_rule.optional import optional_tensor
from ...reference import kda_gate, recurrent_kda


def wrapper(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    A_log: Tensor,
    dt_bias: Tensor | None = None,
    initial_state: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    *,
    scale: float | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """KDA with the gate activation fused in: raw gate *g* becomes -exp(A_log) * softplus(g + dt_bias).

    Differentiable through autograd, but O(T) Python steps: tests and tiny inputs only.
    """
    return recurrent_kda(
        q,
        k,
        v,
        kda_gate(g, A_log, optional_tensor(dt_bias)),
        beta,
        scale=scale,
        initial_state=optional_tensor(initial_state),
        cu_seqlens=optional_tensor(cu_seqlens),
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
    )
