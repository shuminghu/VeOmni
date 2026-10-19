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

"""Needle in a haystack (upstream ``scripts/data/synthetic/niah.py``): 8 of the 13 standard RULER tasks.

Needles ``"One of the special magic numbers for <key> is: <value>."`` are hidden in a haystack of
- ``noise``: the repeated filler sentence, one per line;
- ``essay``: Paul Graham essays, needles inserted between sentences at depths drawn from 40 evenly spaced %;
- ``needle``: other needles (distractor keys), one per line.
"""

import random
import uuid
from typing import List, Set

from .base import RULER_FAMILY_REGISTRY, Rendered, RulerTask
from .resources import NOISE_SENTENCE


# Needle depths (% of the essay's sentences) upstream samples from: round(linspace(0, 100, 40)).
DEPTHS = [round(100 * i / 39) for i in range(40)]


def insert_at_depths(sentences: List[str], items: List[str], rng: random.Random) -> str:
    """Upstream essay insertion: split the sentence list at sampled depths and put one item in each gap."""
    cuts = [0] + sorted(int(len(sentences) * d / 100) for d in rng.sample(DEPTHS, len(items))) + [len(sentences)]
    parts = []
    for i in range(1, len(cuts)):
        parts.append(" ".join(sentences[cuts[i - 1] : cuts[i]]))
        if i - 1 < len(items):
            parts.append(items[i - 1])
    return " ".join(parts)


def essay_sentences(words: List[str], size: int, splitter) -> List[str]:
    """First ``size`` essay words (the corpus repeated if needed), split into sentences."""
    if size > len(words):
        words = words * (size // len(words) + 1)
    return splitter(" ".join(words[:size]).strip())


@RULER_FAMILY_REGISTRY.register("niah")
class NeedleInAHaystack(RulerTask):
    family = "niah"
    tokens_to_generate = 128
    template = (
        "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. "
        "I will quiz you about the {type_needle_v} afterwards.\n{context}\n"
        "What are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
    )
    answer_prefix = " The special magic {type_needle_v} for {query} mentioned in the provided text are"
    needle = "One of the special magic {type_needle_v} for {key} is: {value}."

    def __init__(
        self,
        name,
        tokenizer,
        resources,
        type_haystack: str = "essay",
        type_needle_k: str = "words",
        type_needle_v: str = "numbers",
        num_needle_k: int = 1,
        num_needle_v: int = 1,
        num_needle_q: int = 1,
    ):
        super().__init__(name, tokenizer, resources)
        if type_haystack not in ("noise", "essay", "needle"):
            raise ValueError(f"type_haystack must be noise / essay / needle, got {type_haystack!r}")
        for kind in (type_needle_k, type_needle_v):
            if kind not in ("numbers", "words", "uuids"):
                raise ValueError(f"needle type must be numbers / words / uuids, got {kind!r}")
        self.type_haystack = type_haystack
        self.type_needle_k = type_needle_k
        self.type_needle_v = type_needle_v
        self.num_needle_k = max(num_needle_k, num_needle_q)
        self.num_needle_v = num_needle_v
        self.num_needle_q = num_needle_q
        num_needles = self.num_needle_k * num_needle_v
        if type_haystack == "essay":
            if num_needles > len(DEPTHS):
                raise ValueError(f"essay haystack supports at most {len(DEPTHS)} needles")
            self.probe_size = 500  # words
        else:
            self.min_size = num_needles  # lines
            self.probe_size = 25

    def config(self):
        return {
            k: getattr(self, k)
            for k in (
                "type_haystack",
                "type_needle_k",
                "type_needle_v",
                "num_needle_k",
                "num_needle_v",
                "num_needle_q",
            )
        }

    def random_item(self, kind: str, rng: random.Random, exclude: Set[str] = frozenset()) -> str:
        while True:
            if kind == "numbers":
                item = str(rng.randint(10**6, 10**7 - 1))
            elif kind == "words":
                # Upstream: random.choice over all "adjective-noun" pairs (same distribution, without the 6M list).
                item = f"{rng.choice(self.resources.adjectives)}-{rng.choice(self.resources.nouns)}"
            else:
                item = str(uuid.UUID(int=rng.getrandbits(128), version=4))
            if item not in exclude:
                return item

    def render(self, size, rng, max_seq_len, index):
        keys, values, needles = [], [], []
        for _ in range(self.num_needle_k):
            keys.append(self.random_item(self.type_needle_k, rng, exclude=set(keys)))
            values.append([])
            for _ in range(self.num_needle_v):
                values[-1].append(self.random_item(self.type_needle_v, rng))
                needles.append(
                    self.needle.format(type_needle_v=self.type_needle_v, key=keys[-1], value=values[-1][-1])
                )
        rng.shuffle(needles)

        if self.type_haystack == "essay":
            _, splitter = self.resources.splitter
            context = insert_at_depths(essay_sentences(self.resources.essay_words, size, splitter), needles, rng)
        else:
            if self.type_haystack == "noise":
                lines = [NOISE_SENTENCE] * size
            else:  # distractor needles; keys never collide with the real ones (upstream does not check)
                real = set(keys)
                lines = [
                    self.needle.format(
                        type_needle_v=self.type_needle_v,
                        key=self.random_item(self.type_needle_k, rng, exclude=real),
                        value=self.random_item(self.type_needle_v, rng),
                    )
                    for _ in range(size)
                ]
            for position, needle in zip(sorted(rng.sample(range(size), len(needles)), reverse=True), needles):
                lines.insert(position, needle)
            context = "\n".join(lines)

        queried = rng.sample(range(self.num_needle_k), self.num_needle_q)
        queries = [keys[i] for i in queried]
        answers = [v for i in queried for v in values[i]]
        query = ", ".join(queries[:-1]) + ", and " + queries[-1] if len(queries) > 1 else queries[0]

        template, answer_prefix, type_needle_v = self.template, self.answer_prefix, self.type_needle_v
        if self.num_needle_q * self.num_needle_v == 1:  # upstream singularizes the question for a single answer
            template = template.replace("Some", "A").replace("are all", "is").replace("are", "is")
            answer_prefix = answer_prefix.replace("are", "is")
            type_needle_v = type_needle_v[:-1]
        text = template.format(type_needle_v=type_needle_v, context=context, query=query)
        return Rendered(
            input=text,
            answer_prefix=answer_prefix.format(type_needle_v=type_needle_v, query=query),
            outputs=answers,
            meta={"answer_char_positions": [text.find(a) for a in answers]},
        )
