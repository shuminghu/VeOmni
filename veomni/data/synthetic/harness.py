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

Templates are split into ``train`` and ``test`` (held-out paraphrases). The ``raw`` format has no natural
language at all and mirrors the classic toy setup.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple


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
