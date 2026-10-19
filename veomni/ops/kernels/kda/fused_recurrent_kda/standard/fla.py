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

"""standard fused_recurrent_kda FLA adapter."""

from __future__ import annotations

from torch import Tensor

from ....gated_delta_rule.optional import optional_tensor


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
    """FLA Triton token-recurrent KDA after the fused gate kernel. Lazy-imports ``fla``.

    Forward only: FLA's fused recurrent kernel raises in backward, so training uses ``chunk_kda``. This row
    serves short-sequence inference, where it avoids the chunked kernel's per-chunk overhead.
    """
    from fla.ops.kda import fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate

    initial_state = optional_tensor(initial_state)
    return fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=fused_kda_gate(g=g, A_log=A_log, dt_bias=optional_tensor(dt_bias)),
        beta=beta,
        scale=scale,
        initial_state=None if initial_state is None else initial_state.float(),
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=optional_tensor(cu_seqlens),
    )
