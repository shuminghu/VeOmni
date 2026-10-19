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

"""Pure-PyTorch Kimi Delta Attention (KDA) math shared by the eager rows.

KDA is the gated delta rule with a per-channel (fine-grained) forget gate. Per head, with state ``S`` of shape
``[K, V]``:

    S_t = diag(exp(g_t)) S_{t-1}
    S_t = S_t + beta_t * k_t (v_t - S_t^T k_t)^T
    o_t = scale * S_t^T q_t

where the log-space decay is ``g_t = -exp(A_log) * softplus(f_t + dt_bias)`` (``f_t`` is the raw gate input).
This is the O(T) token-by-token recurrence in FP32: a numerical reference, not a training kernel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def kda_gate(g: Tensor, A_log: Tensor, dt_bias: Tensor | None = None) -> Tensor:
    """Log-space KDA decay ``-exp(A_log) * softplus(g + dt_bias)`` in FP32 for raw gate ``g`` ``[..., H, K]``."""
    num_heads, head_dim = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.float().view(num_heads, head_dim)
    return -A_log.float().exp().view(num_heads, 1) * F.softplus(g)


def l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """L2-normalize the last dimension, as the FLA kernels do with ``use_qk_l2norm_in_kernel``."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def recurrent_kda(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    scale: float | None = None,
    initial_state: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Token-by-token KDA on FLA layouts ``q/k/g [B, T, H, K]``, ``v [B, T, H, V]``, ``beta [B, T, H]``.

    ``g`` is the already-activated log-space decay. With ``cu_seqlens`` the batch must be 1 and every segment
    starts from its own initial state (zeros, or ``initial_state[i]``); the final state is ``[N, H, K, V]``.
    """
    dtype = v.dtype
    if use_qk_l2norm:
        q, k = l2norm(q.float()), l2norm(k.float())
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    q = q * scale

    if cu_seqlens is None:
        segments = [(b, 0, q.shape[1]) for b in range(q.shape[0])]
    else:
        if q.shape[0] != 1:
            raise ValueError(f"cu_seqlens requires batch size 1, got {q.shape[0]}")
        bounds = cu_seqlens.tolist()
        segments = [(0, s, e) for s, e in zip(bounds[:-1], bounds[1:])]

    num_heads, head_k, head_v = q.shape[2], q.shape[3], v.shape[3]
    o = torch.zeros_like(v)
    final_states = []
    for i, (b, start, end) in enumerate(segments):
        state = q.new_zeros(num_heads, head_k, head_v)
        if initial_state is not None:
            state = state + initial_state[i].float()
        for t in range(start, end):
            state = state * g[b, t].exp()[..., None]
            k_t = k[b, t]
            delta = v[b, t] - torch.einsum("hk,hkv->hv", k_t, state)
            state = state + torch.einsum("hk,hv->hkv", beta[b, t][:, None] * k_t, delta)
            o[b, t] = torch.einsum("hk,hkv->hv", q[b, t], state)
        final_states.append(state)
    final_state = torch.stack(final_states) if output_final_state else None
    return o.to(dtype), final_state
