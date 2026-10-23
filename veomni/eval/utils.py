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

"""Helpers shared by the standalone eval scripts: torchrun data parallelism and HF checkpoint loading."""

import os
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


def init_dist() -> Tuple[int, int, torch.device]:
    """``(rank, world_size, device)``; initializes NCCL when launched with torchrun on more than one process."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)
    return 0, 1, torch.device("cuda")


def gather(records: list, world_size: int) -> list:
    """All-gather picklable per-rank lists into one flat list (rank order)."""
    if world_size == 1:
        return records
    gathered = [None] * world_size
    dist.all_gather_object(gathered, records)
    return [r for part in gathered for r in part]


def token_budget_batches(lengths: Sequence[int], max_tokens: int) -> Iterator[List[int]]:
    """Indices grouped (sorted by length) so that ``batch_size * max_len <= max_tokens``."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batch: List[int] = []
    for i in order:
        if batch and (len(batch) + 1) * lengths[i] > max_tokens:
            yield batch
            batch = []
        batch.append(i)
    if batch:
        yield batch


def load_causal_lm(
    model_path: str,
    device: torch.device,
    attn_implementation: str = "sdpa",
    ternary: Optional[dict] = None,
):
    """bf16 HF causal LM in eval mode. ``ternary`` (``{"group_size", "threshold"}``): the checkpoint holds
    ternary-QAT latent weights, so ternarize the hidden linears to evaluate the model training saw."""
    from transformers import AutoModelForCausalLM

    from .. import models  # noqa: F401  (registers VeOmni-only model types, e.g. qwen3_hybrid, on the HF Auto classes)

    model = AutoModelForCausalLM.from_pretrained(
        os.path.expanduser(model_path), dtype=torch.bfloat16, attn_implementation=attn_implementation
    ).to(device)
    if ternary is not None:
        from ..ops.qat.ternary import ternarize_linears_

        ternarize_linears_(model, **ternary)
    return model.eval()


def add_qat_args(parser) -> None:
    parser.add_argument(
        "--qat_weight",
        default="none",
        choices=["none", "ternary_twn"],
        help="Match the training run's model.qat.weight: ternarize hidden linears before evaluating.",
    )
    parser.add_argument("--qat_group_size", type=int, default=0, help="model.qat.group_size of the run.")
    parser.add_argument("--qat_threshold", type=float, default=0.7, help="model.qat.threshold of the run.")
    parser.add_argument(
        "--qat_exclude",
        nargs="*",
        default=[],
        help="model.qat.exclude_modules of the run: module-name regexes of hidden linears trained in full precision.",
    )


def ternary_from_args(args) -> Optional[dict]:
    if args.qat_weight == "ternary_twn":
        return {
            "group_size": args.qat_group_size,
            "threshold": args.qat_threshold,
            "exclude": args.qat_exclude or None,
        }
    return None
