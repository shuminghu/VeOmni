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

"""Interface shared by all synthetic tasks.

A synthetic dataset is stored as *recipes*: small parquet rows such as
``{"task": "selective_copy", "seed": 17, "ctx_len": 4096, "n_copy": 32, ...}``. A task turns one recipe into
token ids deterministically, so recipe files are tiny, fully reproducible, and flow through the regular
VeOmni dataset / multisource / resume machinery like any other parquet source.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence

from ...utils.constants import IGNORE_INDEX
from ...utils.registry import Registry


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


SYNTHETIC_TASK_REGISTRY = Registry("SyntheticTask")


@dataclass
class SyntheticSample:
    """One rendered instance. ``labels`` is unshifted (HF convention): ``labels[i]`` is the target at ``i``."""

    input_ids: List[int]
    labels: List[int]
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def label_positions(self) -> List[int]:
        return [i for i, label in enumerate(self.labels) if label != IGNORE_INDEX]


class SyntheticTask(ABC):
    """Base class: ``build`` renders a recipe, ``score`` grades predictions at the label positions."""

    name: str

    def __init__(self, tokenizer: "PreTrainedTokenizer"):
        self.tokenizer = tokenizer
        if tokenizer.eos_token_id is None:
            raise ValueError("Synthetic tasks need a tokenizer with an eos token.")
        self.eos_id = tokenizer.eos_token_id

    @abstractmethod
    def build(self, recipe: Mapping[str, Any]) -> SyntheticSample:
        """Render a recipe row into a sample."""

    @abstractmethod
    def score(self, sample: SyntheticSample, pred_ids: Sequence[int]) -> Dict[str, Any]:
        """Grade predictions. ``pred_ids[k]`` is the model's prediction for ``sample.label_positions[k]``."""

    def check_canonical(self, sample: SyntheticSample) -> None:
        """Raise if the sample's ids are not what the tokenizer would produce from its own decoded text.

        Non-canonical boundaries would make the synthetic text a distribution shift w.r.t. natural text and
        would make positions differ when the text is re-tokenized by another pipeline.
        """
        ids = sample.input_ids[:-1] if sample.input_ids[-1] == self.eos_id else sample.input_ids
        text = self.tokenizer.decode(ids)
        reencoded = self.tokenizer.encode(text, add_special_tokens=False)
        if reencoded != list(ids):
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(ids, reencoded)) if a != b), min(len(ids), len(reencoded))
            )
            window = slice(max(0, first_diff - 3), first_diff + 3)
            raise ValueError(
                f"{self.name}: sample is not canonically tokenized at index {first_diff}: "
                f"built {self.tokenizer.convert_ids_to_tokens(list(ids[window]))} vs "
                f"re-encoded {self.tokenizer.convert_ids_to_tokens(reencoded[window])}"
            )


_TASK_CACHE: Dict[tuple, SyntheticTask] = {}


def get_synthetic_task(name: str, tokenizer: "PreTrainedTokenizer") -> SyntheticTask:
    """Instantiate (once per tokenizer) the task registered under ``name``."""
    key = (name, id(tokenizer))
    if key not in _TASK_CACHE:
        _TASK_CACHE[key] = SYNTHETIC_TASK_REGISTRY[name](tokenizer)
    return _TASK_CACHE[key]


def build_synthetic_sample(recipe: Mapping[str, Any], tokenizer: "PreTrainedTokenizer") -> SyntheticSample:
    return get_synthetic_task(recipe["task"], tokenizer).build(recipe)
