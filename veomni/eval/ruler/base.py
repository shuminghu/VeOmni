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

"""Task interface and length fitting shared by all RULER task families.

A family (``niah``, ``variable_tracking``, ...) renders a prompt from a haystack *size* (words, sentences, docs...)
and a per-sample RNG. ``RulerTask.generate`` picks the largest size whose prompt fits
``max_seq_len - tokens_to_generate`` tokens, as upstream does with its binary search.

Differences from upstream (NVIDIA/RULER) generation, all distribution-preserving:
- every sample has its own RNG (seeded by ``seed/task/max_seq_len/index``), so samples are independent of
  generation order and can be built in parallel; upstream draws all samples from one global stream, and uses a
  *fixed*-seed shuffle in a few places (needle order, CWE / FWE word order, QA doc order) that we draw per sample;
- the haystack size is searched once per (task, length) on sample 0 and only re-searched for samples that
  overflow (upstream decrements by a fixed step, which can loop forever for QA at short lengths).
"""

import bisect
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List

from ...utils.registry import Registry
from .resources import RulerResources


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


RULER_FAMILY_REGISTRY = Registry("RulerTaskFamily")


@dataclass
class RulerSample:
    """One prompt. The model sees ``input + answer_prefix`` and continues it (upstream ``base`` template)."""

    task: str
    index: int
    max_seq_len: int
    input: str
    answer_prefix: str
    outputs: List[str]
    length: int  # prompt tokens + tokens_to_generate (upstream ``length``)
    tokens_to_generate: int
    metric: str
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return self.input + self.answer_prefix

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "RulerSample":
        return cls(**row)


@dataclass
class Rendered:
    input: str
    answer_prefix: str
    outputs: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return self.input + self.answer_prefix


def sample_rng(seed: int, task: str, max_seq_len: int, index: int) -> random.Random:
    # str seeds are hashed with sha512 by random.Random (version 2): stable across processes and Python builds.
    return random.Random(f"ruler/{seed}/{task}/{max_seq_len}/{index}")


def largest_fitting(count: Callable[[int], int], budget: int, lo: int, guess: int) -> int:
    """Largest ``size >= lo`` with ``count(size) <= budget``, assuming ``count`` is non-decreasing.

    Gallops up from ``guess`` (or bisects below it), so the cost is O(log(size / guess)) calls when the guess is
    good. Raises if even ``lo`` does not fit.
    """
    if count(lo) > budget:
        raise ValueError(f"even the minimum haystack size {lo} exceeds the {budget}-token budget")
    ok, hi = lo, max(guess, lo + 1)
    while count(hi) <= budget:
        ok, hi = hi, hi * 2
    while hi - ok > 1:
        mid = (ok + hi) // 2
        if count(mid) <= budget:
            ok = mid
        else:
            hi = mid
    return ok


class RulerTask(ABC):
    """A configured task (e.g. ``niah_multikey_2`` = family ``niah`` + its arguments)."""

    family: ClassVar[str]
    template: ClassVar[str]
    answer_prefix: ClassVar[str]
    tokens_to_generate: ClassVar[int]
    metric: ClassVar[str] = "string_match_all"
    min_size: int = 1  # smallest valid haystack size (instances may raise it)
    probe_size: ClassVar[int] = 100  # haystack size rendered to estimate tokens per unit

    def __init__(self, name: str, tokenizer: "PreTrainedTokenizer", resources: RulerResources):
        self.name = name
        self.tokenizer = tokenizer
        self.resources = resources
        self._size_cache: Dict[int, int] = {}

    # ------------------------------------------------------------------ to implement

    @abstractmethod
    def render(self, size: int, rng: random.Random, max_seq_len: int, index: int) -> Rendered:
        """Render sample ``index`` with ``size`` haystack units, drawing randomness only from ``rng``.

        ``meta["answer_char_positions"]`` (optional): offsets in ``input`` of the evidence for each output; they are
        converted to token positions for depth analysis.
        """

    def config(self) -> Dict[str, Any]:
        """Task arguments, stored with the generated data."""
        return {}

    # ------------------------------------------------------------------ shared

    def num_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def budget(self, max_seq_len: int) -> int:
        return max_seq_len - self.tokens_to_generate

    def _render(self, size: int, seed: int, max_seq_len: int, index: int) -> Rendered:
        return self.render(size, sample_rng(seed, self.name, max_seq_len, index), max_seq_len, index)

    def _count_fn(self, seed: int, max_seq_len: int, index: int) -> Callable[[int], int]:
        return lambda size: self.num_tokens(self._render(size, seed, max_seq_len, index).prompt)

    def fitted_size(self, max_seq_len: int, seed: int) -> int:
        """Haystack size for this length, searched on sample 0 (upstream searches once per task and length)."""
        if max_seq_len not in self._size_cache:
            probe = max(self.probe_size, self.min_size)
            probe_tokens = self.num_tokens(self._render(probe, seed, max_seq_len, 0).prompt)
            guess = max(self.min_size, self.budget(max_seq_len) * probe // probe_tokens)
            count = self._count_fn(seed, max_seq_len, 0)
            self._size_cache[max_seq_len] = largest_fitting(count, self.budget(max_seq_len), self.min_size, guess)
        return self._size_cache[max_seq_len]

    def generate(self, index: int, max_seq_len: int, seed: int = 42) -> RulerSample:
        budget = self.budget(max_seq_len)
        size = self.fitted_size(max_seq_len, seed)
        rendered = self._render(size, seed, max_seq_len, index)
        num_tokens = self.num_tokens(rendered.prompt)
        if num_tokens > budget:  # this sample's draws made it longer than sample 0: shrink just this one
            size = largest_fitting(self._count_fn(seed, max_seq_len, index), budget, self.min_size, size)
            rendered = self._render(size, seed, max_seq_len, index)
            num_tokens = self.num_tokens(rendered.prompt)
        meta = dict(rendered.meta)
        if "answer_char_positions" in meta:
            positions = self.char_to_token_positions(rendered.input, meta.pop("answer_char_positions"))
            meta["answer_token_positions"] = positions
        return RulerSample(
            task=self.name,
            index=index,
            max_seq_len=max_seq_len,
            input=rendered.input,
            answer_prefix=rendered.answer_prefix,
            outputs=rendered.outputs,
            length=num_tokens + self.tokens_to_generate,
            tokens_to_generate=self.tokens_to_generate,
            metric=self.metric,
            meta={"haystack_size": size, **meta},
        )

    def char_to_token_positions(self, text: str, char_positions: List[int]) -> List[int]:
        """Token index containing each character offset of ``text`` (for answer-depth analysis)."""
        offsets = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        starts = [s for s, _ in offsets]
        return [max(0, bisect.bisect_right(starts, c) - 1) for c in char_positions]


# ---------------------------------------------------------------------- standard RULER suite

# Upstream ``scripts/synthetic.yaml``: task name -> (family, arguments).
RULER_TASKS: Dict[str, tuple] = {
    "niah_single_1": ("niah", dict(type_haystack="noise", type_needle_k="words", type_needle_v="numbers")),
    "niah_single_2": ("niah", dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers")),
    "niah_single_3": ("niah", dict(type_haystack="essay", type_needle_k="words", type_needle_v="uuids")),
    "niah_multikey_1": (
        "niah",
        dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers", num_needle_k=4),
    ),
    "niah_multikey_2": ("niah", dict(type_haystack="needle", type_needle_k="words", type_needle_v="numbers")),
    "niah_multikey_3": ("niah", dict(type_haystack="needle", type_needle_k="uuids", type_needle_v="uuids")),
    "niah_multivalue": (
        "niah",
        dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers", num_needle_v=4),
    ),
    "niah_multiquery": (
        "niah",
        dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers", num_needle_q=4),
    ),
    "vt": ("variable_tracking", dict(type_haystack="noise", num_chains=1, num_hops=4)),
    "cwe": ("common_words_extraction", dict(freq_cw=30, freq_ucw=3, num_cw=10)),
    "fwe": ("freq_words_extraction", dict(alpha=2.0)),
    "qa_1": ("qa", dict(dataset="squad")),
    "qa_2": ("qa", dict(dataset="hotpotqa")),
}

# Coarse categories used in the RULER paper's analysis.
RULER_CATEGORIES: Dict[str, List[str]] = {
    "retrieval": [t for t in RULER_TASKS if t.startswith("niah")],
    "multi_hop_tracing": ["vt"],
    "aggregation": ["cwe", "fwe"],
    "qa": ["qa_1", "qa_2"],
}


def build_ruler_task(
    name: str, tokenizer: "PreTrainedTokenizer", resources: RulerResources, **overrides: Any
) -> RulerTask:
    """Build a standard RULER task by name, or ``"<family>:<custom name>"`` with ``overrides`` as arguments."""
    if name in RULER_TASKS:
        family, kwargs = RULER_TASKS[name]
        kwargs = {**kwargs, **overrides}
    elif ":" in name:
        family, name = name.split(":", 1)
        kwargs = overrides
    else:
        raise ValueError(f"Unknown RULER task {name!r}; standard tasks: {list(RULER_TASKS)}")
    return RULER_FAMILY_REGISTRY[family](name, tokenizer, resources, **kwargs)
