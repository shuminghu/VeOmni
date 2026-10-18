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

"""Natural-language harnesses that wrap a synthetic token body into a prompt.

A template is ``prefix`` + body + ``suffix`` + answer. Body items are space-prefixed single tokens, so:
- ``prefix`` must end with a character that does not merge with a following ``" x"`` token (e.g. ``":"``).
- ``suffix`` must start with a space so it never merges with the last body token.
Both rules are enforced at runtime by ``SyntheticTask.check_canonical``.

Templates are split into ``train`` and ``test`` (held-out paraphrases). Formats:

- ``nl``: 8 + 2 hand-written templates (v1). Every train prefix has exactly one colon, right before the body, and
  models learn "the answer starts after the first colon" (a held-out prefix with an earlier colon breaks them).
- ``nl2``: templates composed from part pools (intro x list marker x end marker x question), 6144 train / 72 test,
  disjoint pools. Intros often contain ``"Word: words"`` and list markers vary (``":"``, ``"="``, ``"->"``,
  brackets, bare words), so punctuation alone does not locate the body.
- ``raw``: no natural language at all; mirrors the classic toy setup.

Template ids index these lists, so a published format is frozen: add a new format instead of editing one.
"""

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


@dataclass(frozen=True)
class PromptTemplate:
    prefix: str
    suffix: str


@dataclass(frozen=True)
class TokenizedTemplate:
    prefix_ids: Tuple[int, ...]
    suffix_ids: Tuple[int, ...]

    @property
    def overhead(self) -> int:
        return len(self.prefix_ids) + len(self.suffix_ids)


# format -> split -> templates
TemplateSet = Dict[str, Dict[str, List[PromptTemplate]]]


@dataclass(frozen=True)
class TemplateParts:
    """Pools composed as ``prefix = intro + marker`` and ``suffix = closer + question`` (all combinations)."""

    intros: Sequence[str]
    markers: Sequence[str]
    closers: Sequence[str]
    questions: Sequence[str]

    def compose(self) -> List[PromptTemplate]:
        return [
            PromptTemplate(intro + marker, closer + question)
            for intro, marker, closer, question in itertools.product(
                self.intros, self.markers, self.closers, self.questions
            )
        ]


NL2_TRAIN_PARTS = TemplateParts(
    intros=(
        "Memory test. The log below is mostly filler. Remember every word that appears in it, in order.",
        "Note: most entries below are filler, but a few are real words.",
        "Task: read the sequence and keep track of the words hidden in it.",
        "Instructions: ignore the dots, remember the words, and keep their order.",
        "Here is a field notebook where most cells hold filler.",
        "Rules of the game: words are hidden among filler marks. Find them all.",
        "Copy task. Ignore the filler and keep the words.",
        "Warning: the record below is padded. Only the words matter.",
        "A student wrote a few words between lots of filler on a long tape.",
        "Context: this is a noisy transmission. Goal: recover the words in order.",
        "Below is a noisy transmission. Only a few words carry the message; the rest is static.",
        "Reminder: order matters. Hint: most of the text is filler.",
        "Find the words hidden in the filler below and repeat them in the same order.",
        "Step one: read everything. Step two: list the words you saw.",
        "The following record is padded with filler. Extract the words.",
        "Topic: memory. Format: words scattered in filler, then a question.",
    ),
    markers=(
        "\nLog:",
        "\nSequence:",
        " Here it is:",
        "\nInput =",
        "\nData ->",
        "\n[start]",
        "\nBEGIN",
        "\nText follows",
    ),
    closers=(" (end of log)", " [end]", " <end>", " -- end --", " ###", " (done)"),
    questions=(
        "\nQuestion: Which words appeared, in order?\nAnswer:",
        "\nList the words in the order they appeared:",
        "\nOutput:",
        "\nQ: What words were hidden, first to last?\nA:",
        "\nWords in order:",
        "\nThe words, in order, were",
        "\nRepeat them now ->",
        "\nAnswer =",
    ),
)

NL2_TEST_PARTS = TemplateParts(
    intros=(
        "Attention check. A long stretch of filler follows, with a few words scattered in it.",
        "Signal and noise: most of what follows is noise, a few words are signal.",
        "Exercise: spot the words. Difficulty: the filler is long.",
        "Please read carefully, since only some items are words.",
    ),
    markers=("\nList:", "\nStream:", "\n(begin)"),
    closers=(" (list over)", " | end"),
    questions=(
        "\nPlease repeat the scattered words in their original order:",
        "\nRecovered signal:",
        "\nIn order, the words are",
    ),
)


SELECTIVE_COPY_TEMPLATES: TemplateSet = {
    "nl": {
        "train": [
            PromptTemplate(
                "Memory test. The log below is mostly filler. Remember every word that appears in it, in order.\nLog:",
                " (end of log)\nQuestion: Which words appeared in the log, in order?\nAnswer:",
            ),
            PromptTemplate(
                "Read the following sequence carefully. Most entries are filler, but a few are words.\nSequence:",
                " (end of sequence)\nList the words from the sequence in the order they appeared:",
            ),
            PromptTemplate(
                "Below is a noisy transmission. Only a few words carry the message; the rest is static.\n"
                "Transmission:",
                " (transmission ends)\nThe message, word by word, was:",
            ),
            PromptTemplate(
                "Here is a field notebook where most cells hold filler.\nNotebook:",
                " (notebook ends)\nWrite down the recorded words in order.\nRecorded words:",
            ),
            PromptTemplate(
                "Copy task. Ignore the filler and keep the words.\nInput:",
                " (end of input)\nOutput:",
            ),
            PromptTemplate(
                "A student wrote a few words between lots of filler on a long tape.\nTape:",
                " (end of tape)\nQ: What words were written on the tape, from first to last?\nA:",
            ),
            PromptTemplate(
                "Find the words hidden in the filler below and repeat them in the same order.\nText:",
                " (end of text)\nWords in order:",
            ),
            PromptTemplate(
                "The following record is padded with filler. Extract the words.\nRecord:",
                " (record ends)\nExtracted words:",
            ),
        ],
        "test": [
            PromptTemplate(
                "Attention check. A long stretch of filler follows, with a few words scattered in it.\nList:",
                " (list over)\nPlease repeat the scattered words in their original order:",
            ),
            PromptTemplate(
                "Signal and noise: most of what follows is noise, a few words are signal.\nStream:",
                " (stream ends)\nRecovered signal:",
            ),
        ],
    },
    "nl2": {"train": NL2_TRAIN_PARTS.compose(), "test": NL2_TEST_PARTS.compose()},
    "raw": {
        "train": [PromptTemplate("", " ==>")],
        "test": [PromptTemplate("", " ==>")],
    },
}


def tokenize_templates(tokenizer: "PreTrainedTokenizer", templates: TemplateSet) -> Dict[str, Dict[str, list]]:
    return {
        fmt: {
            split: [
                TokenizedTemplate(
                    prefix_ids=tuple(tokenizer.encode(t.prefix, add_special_tokens=False)),
                    suffix_ids=tuple(tokenizer.encode(t.suffix, add_special_tokens=False)),
                )
                for t in split_templates
            ]
            for split, split_templates in splits.items()
        }
        for fmt, splits in templates.items()
    }
