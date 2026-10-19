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

"""Aggregation tasks: common words extraction (CWE) and frequent words extraction (FWE).

Upstream ``scripts/data/synthetic/{common_words_extraction,freq_words_extraction}.py``.
"""

import math
import random
import string
from functools import cached_property
from typing import List, Tuple

from .base import RULER_FAMILY_REGISTRY, Rendered, RulerTask


def riemann_zeta(alpha: float, terms: int = 1000) -> float:
    """zeta(alpha) for alpha > 1 (Euler-Maclaurin tail), replacing upstream's ``scipy.special.zeta``."""
    head = sum(k**-alpha for k in range(1, terms))
    return head + terms ** (1 - alpha) / (alpha - 1) + terms**-alpha / 2 + alpha * terms ** (-alpha - 1) / 12


@RULER_FAMILY_REGISTRY.register("common_words_extraction")
class CommonWordsExtraction(RulerTask):
    """A numbered word list where ``num_cw`` words repeat ``freq_cw`` times and the rest ``freq_ucw`` times.

    Below 4096 tokens upstream switches to repeats 6 / 1 and a smaller in-context example; we keep that.
    """

    family = "common_words_extraction"
    tokens_to_generate = 120
    template = (
        "Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones "
        "that appear most often.\n{context}\nQuestion: What are the 10 most common words in the above list?"
    )
    answer_prefix = " Answer: The top 10 words that appear most often in the list are:"
    probe_size = 200

    def __init__(
        self, name, tokenizer, resources, freq_cw: int = 30, freq_ucw: int = 3, num_cw: int = 10, num_fewshot: int = 1
    ):
        super().__init__(name, tokenizer, resources)
        self.freq_cw, self.freq_ucw, self.num_cw, self.num_fewshot = freq_cw, freq_ucw, num_cw, num_fewshot
        self.min_size = num_cw + 1

    def config(self):
        return {
            "freq_cw": self.freq_cw,
            "freq_ucw": self.freq_ucw,
            "num_cw": self.num_cw,
            "num_fewshot": self.num_fewshot,
        }

    @cached_property
    def words(self) -> List[str]:
        r = self.resources
        return sorted(set(r.nouns + r.adjectives + r.verbs))

    def example(
        self, rng: random.Random, num_words: int, common_repeats: int, uncommon_repeats: int
    ) -> Tuple[str, List[str]]:
        if num_words > len(self.words):
            # Upstream falls back to a larger list (english_words.json); only reached around >= 128k tokens.
            raise ValueError(f"CWE needs {num_words} distinct words, only {len(self.words)} available")
        chosen = rng.sample(self.words, num_words)
        common, uncommon = chosen[: self.num_cw], chosen[self.num_cw :]
        word_list = common * common_repeats + uncommon * uncommon_repeats
        rng.shuffle(word_list)
        return " ".join(f"{i + 1}. {w}" for i, w in enumerate(word_list)), common

    def render(self, size, rng, max_seq_len, index):
        if max_seq_len < 4096:
            shot_args, main_repeats = (20, 3, 1), (6, 1)
        else:
            shot_args, main_repeats = (40, 10, 3), (self.freq_cw, self.freq_ucw)
        shots = []
        for _ in range(self.num_fewshot):
            context, answer = self.example(rng, *shot_args)
            shots.append(
                self.template.format(context=context)
                + self.answer_prefix
                + " "
                + " ".join(f"{i + 1}. {w}" for i, w in enumerate(answer))
            )
        context, answer = self.example(rng, size, *main_repeats)
        return Rendered(
            input="\n".join(shots) + "\n" + self.template.format(context=context),
            answer_prefix=self.answer_prefix,
            outputs=answer,
        )


@RULER_FAMILY_REGISTRY.register("freq_words_extraction")
class FreqWordsExtraction(RulerTask):
    """Random 'coded' words with Zipf(alpha) counts; the top-ranked one is replaced by '...' noise and the
    answer is ranks 2-4."""

    family = "freq_words_extraction"
    tokens_to_generate = 50
    template = (
        "Read the following coded text and track the frequency of each coded word. Find the three most frequently "
        "appeared coded words. {context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. "
        "What are the three most frequently appeared words in the above coded text?"
    )
    answer_prefix = " Answer: According to the coded text above, the three most frequently appeared words are:"
    probe_size = 2000

    def __init__(self, name, tokenizer, resources, alpha: float = 2.0, coded_wordlen: int = 6, vocab_size: int = -1):
        super().__init__(name, tokenizer, resources)
        if alpha <= 1:
            raise ValueError("FWE needs alpha > 1")
        self.alpha, self.coded_wordlen, self.vocab_size = alpha, coded_wordlen, vocab_size
        self.zeta = riemann_zeta(alpha)
        # Smallest word count at which rank 5 still gets a word, so ranks 2-4 are all present and distinct-ish.
        self.min_size = math.ceil(self.zeta * 5**alpha)

    def config(self):
        return {"alpha": self.alpha, "coded_wordlen": self.coded_wordlen, "vocab_size": self.vocab_size}

    def render(self, size, rng, max_seq_len, index):
        vocab_size = self.budget(max_seq_len) // 50 if self.vocab_size == -1 else self.vocab_size
        vocab = set()
        while len(vocab) < vocab_size:
            vocab.add("".join(rng.choices(string.ascii_lowercase, k=self.coded_wordlen)))
        vocab = sorted(vocab)
        rng.shuffle(vocab)
        vocab[0] = "..."  # the most frequent "word" is noise
        words = [w for k, w in enumerate(vocab, start=1) for _ in range(int(size * k**-self.alpha / self.zeta))]
        rng.shuffle(words)
        return Rendered(
            input=self.template.format(context=" ".join(words)),
            answer_prefix=self.answer_prefix,
            outputs=vocab[1:4],
        )
