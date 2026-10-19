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

"""standard chunk_kda FLA adapter."""

from __future__ import annotations

from functools import cache

from torch import Tensor

from ....gated_delta_rule.optional import optional_tensor


# fla 0.4.x's chunk_kda backward (``chunk_kda_bwd_kernel_wy_dqkg_fused``) hits an illegal memory access while
# autotuning on H200 / Triton 3.6; 0.5.x reworked it.
MIN_FLA_VERSION = (0, 5)


@cache
def _check_fla_version() -> None:
    import fla

    version = tuple(int(part) for part in fla.__version__.split(".")[:2])
    if version < MIN_FLA_VERSION:
        raise RuntimeError(
            f"chunk_kda needs flash-linear-attention >= {'.'.join(map(str, MIN_FLA_VERSION))} (found "
            f"{fla.__version__}): older chunk_kda backward kernels crash with an illegal memory access."
        )


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
    """FLA Triton chunked KDA (forward and backward), gate activation computed in-kernel. Lazy-imports ``fla``."""
    from fla.ops.kda import chunk_kda

    _check_fla_version()
    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=optional_tensor(initial_state),
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=True,
        cu_seqlens=optional_tensor(cu_seqlens),
        A_log=A_log,
        dt_bias=optional_tensor(dt_bias),
    )
