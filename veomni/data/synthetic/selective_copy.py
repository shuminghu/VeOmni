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

"""Selective copying: a few content words are scattered among filler; the model must repeat them in order.

Layout (token level)::

    [prefix] [body: ctx_len single-token items] [suffix] [answer: n_copy items] [eos]
                                                         ^ labels only here (answer + eos)

``ctx_len`` controls how far the model must look back, ``n_copy`` how much it must remember.
"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ...utils.constants import IGNORE_INDEX
from .base import SYNTHETIC_TASK_REGISTRY, SyntheticSample, SyntheticTask
from .harness import SELECTIVE_COPY_TEMPLATES, tokenize_templates
from .lexicon import TokenLexicon, build_lexicon


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


# Fraction of the body in which content may be placed.
PLACEMENT_SPANS: Dict[str, Tuple[float, float]] = {
    "uniform": (0.0, 1.0),
    "early": (0.0, 0.25),
    "late": (0.75, 1.0),
}


@dataclass(frozen=True)
class SelectiveCopyRecipe:
    seed: int
    ctx_len: int
    n_copy: int
    placement: str = "uniform"
    template_split: str = "train"
    template_id: int = -1  # -1: drawn from ``seed``
    fmt: str = "nl"
    content_size: int = 2048
    noise_size: int = 1
    # False (pre-capacity-axis recipes): content drawn with replacement, so a word can repeat within a sample.
    unique_content: bool = False

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SelectiveCopyRecipe":
        kwargs = {}
        for f in fields(cls):
            value = row.get(f.name)
            if value is not None:
                kwargs[f.name] = f.type(value)
        return cls(**kwargs)


@SYNTHETIC_TASK_REGISTRY.register("selective_copy")
class SelectiveCopyTask(SyntheticTask):
    name = "selective_copy"

    def __init__(self, tokenizer: "PreTrainedTokenizer"):
        super().__init__(tokenizer)
        self.templates = tokenize_templates(tokenizer, SELECTIVE_COPY_TEMPLATES)

    def lexicon(self, recipe: SelectiveCopyRecipe) -> TokenLexicon:
        return build_lexicon(self.tokenizer, content_size=recipe.content_size, noise_size=recipe.noise_size)

    def max_overhead(self, n_copy: int, fmt: str = "nl", template_split: str = "train") -> int:
        """Worst-case number of non-body tokens, used to keep ``ctx_len`` within a sequence budget."""
        return max(t.overhead for t in self.templates[fmt][template_split]) + n_copy + 1

    def sample_layout(
        self, recipe: SelectiveCopyRecipe, rng: np.random.Generator
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Draw (template_id, body positions, content token ids) for a recipe."""
        num_templates = len(self.templates[recipe.fmt][recipe.template_split])
        template_id = recipe.template_id if recipe.template_id >= 0 else int(rng.integers(num_templates))

        lo, hi = PLACEMENT_SPANS[recipe.placement]
        span_lo, span_hi = int(lo * recipe.ctx_len), int(hi * recipe.ctx_len)
        if span_hi - span_lo < recipe.n_copy:
            raise ValueError(f"Placement span {span_hi - span_lo} is shorter than n_copy={recipe.n_copy}.")
        positions = np.sort(rng.choice(span_hi - span_lo, size=recipe.n_copy, replace=False)) + span_lo

        content_ids = np.asarray(self.lexicon(recipe).content_ids)
        if recipe.unique_content:
            # Repeated words make "what came after X" ambiguous, which matters once n_copy is a sizeable
            # fraction of the lexicon (~n^2 / 2|V| repeated pairs with replacement).
            if recipe.n_copy > len(content_ids):
                raise ValueError(f"n_copy={recipe.n_copy} exceeds the content lexicon size {len(content_ids)}.")
            content = content_ids[rng.choice(len(content_ids), size=recipe.n_copy, replace=False)]
        else:
            content = content_ids[rng.integers(len(content_ids), size=recipe.n_copy)]
        return template_id, positions, content

    def build(self, row: Mapping[str, Any]) -> SyntheticSample:
        """Render a recipe row. Optional ``body_positions`` / ``content`` columns pin the layout explicitly
        (used for frozen test sets, so they do not depend on the RNG implementation)."""
        recipe = SelectiveCopyRecipe.from_mapping(row)
        rng = np.random.default_rng(recipe.seed)
        template_id, positions, content = self.sample_layout(recipe, rng)
        if row.get("body_positions") is not None:
            positions = np.asarray(row["body_positions"], dtype=np.int64)
        if row.get("content") is not None:
            content = np.asarray(row["content"], dtype=np.int64)

        lexicon = self.lexicon(recipe)
        noise_ids = np.asarray(lexicon.noise_ids)
        body = noise_ids[rng.integers(len(noise_ids), size=recipe.ctx_len)]
        body[positions] = content

        template = self.templates[recipe.fmt][recipe.template_split][template_id]
        prefix, suffix = list(template.prefix_ids), list(template.suffix_ids)
        answer = content.tolist() + [self.eos_id]
        input_ids = prefix + body.tolist() + suffix + answer

        answer_start = len(input_ids) - len(answer)
        labels = [IGNORE_INDEX] * answer_start + answer

        body_start = len(prefix)
        query_pos = body_start + recipe.ctx_len
        abs_positions = (positions + body_start).tolist()
        meta = {
            "task": self.name,
            "ctx_len": recipe.ctx_len,
            "n_copy": recipe.n_copy,
            "placement": recipe.placement,
            "fmt": recipe.fmt,
            "template_split": recipe.template_split,
            "template_id": template_id,
            "content": content.tolist(),
            "positions": abs_positions,
            "distances": [query_pos - p for p in abs_positions],
            "body_start": body_start,
            "query_pos": query_pos,
            "answer_start": answer_start,
            "lexicon": lexicon.fingerprint,
        }
        return SyntheticSample(input_ids=input_ids, labels=labels, meta=meta)

    def score(self, sample: SyntheticSample, pred_ids: Sequence[int]) -> Dict[str, Any]:
        content = sample.meta["content"]
        n_copy = len(content)
        if len(pred_ids) != n_copy + 1:
            raise ValueError(f"Expected {n_copy + 1} predictions (answer + eos), got {len(pred_ids)}.")
        item_correct = [int(p) == c for p, c in zip(pred_ids[:n_copy], content)]
        return {
            "token_acc": sum(item_correct) / n_copy,
            "exact": all(item_correct),
            "eos_correct": int(pred_ids[n_copy]) == self.eos_id,
            "item_correct": item_correct,
            "distances": sample.meta["distances"],
        }
