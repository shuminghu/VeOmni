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

"""Single-token word lexicons for synthetic tasks.

Synthetic sequences are assembled directly from token ids so that lengths and positions are exact. That
only works if every item is exactly one token and re-tokenizes to itself next to other items. We use
space-prefixed lowercase English words (``" maple"``), which byte-level BPE pre-tokenizers always split on.

Words come from ``assets/english_words.txt``: standalone words ranked by frequency in FineWeb-Edu
(``scripts/longctx/make_wordlist.py``). Each tokenizer keeps the ones that are single tokens, in rank order.
"""

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Tuple


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


WORDLIST_PATH = os.path.join(os.path.dirname(__file__), "assets", "english_words.txt")

# Stems of "n't" contractions, excluded from the word list. FineWeb-Edu mostly writes "doesn’t" (U+2019), which
# the ASCII-apostrophe word regex splits into "doesn". Dropped outright: standalone "won" / "don" are also mostly
# "won't" / "don't" in web text, and a lexicon word should read as a whole word.
CONTRACTION_STEMS = frozenset(
    "don doesn didn isn won wasn aren weren hasn haven hadn couldn wouldn shouldn mustn needn ain".split()
)


@dataclass(frozen=True)
class TokenLexicon:
    """Disjoint content / noise token-id sets used to build synthetic bodies."""

    content_ids: Tuple[int, ...]
    noise_ids: Tuple[int, ...]

    @property
    def fingerprint(self) -> str:
        payload = ",".join(map(str, self.content_ids)) + "|" + ",".join(map(str, self.noise_ids))
        return hashlib.sha1(payload.encode()).hexdigest()[:12]


@lru_cache(maxsize=None)
def load_wordlist(path: str = WORDLIST_PATH) -> Tuple[str, ...]:
    with open(path) as f:
        return tuple(line.strip() for line in f if line.strip())


def find_word_tokens(tokenizer: "PreTrainedTokenizer", wordlist_path: str = WORDLIST_PATH) -> Tuple[int, ...]:
    """Token ids of the words in ``wordlist_path`` whose space-prefixed form is a single token, in list order."""
    word_ids = []
    for word in load_wordlist(wordlist_path):
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) == 1:
            word_ids.append(ids[0])
    return tuple(word_ids)


_LEXICON_CACHE: Dict[tuple, TokenLexicon] = {}


def build_lexicon(
    tokenizer: "PreTrainedTokenizer",
    content_size: int = 2048,
    noise_size: int = 1,
    noise_token: str = " .",
    wordlist_path: str = WORDLIST_PATH,
) -> TokenLexicon:
    """Build (and cache) a lexicon.

    - ``content_ids``: the ``content_size`` most frequent single-token words.
    - ``noise_ids``: ``noise_token`` alone when ``noise_size == 1``; otherwise the next ``noise_size`` word
      tokens after the content words (disjoint from content, so content stays identifiable by identity).
    """
    key = (tokenizer.name_or_path, len(tokenizer), content_size, noise_size, noise_token, wordlist_path)
    if key in _LEXICON_CACHE:
        return _LEXICON_CACHE[key]

    words = find_word_tokens(tokenizer, wordlist_path)
    needed = content_size + (noise_size if noise_size > 1 else 0)
    if len(words) < needed:
        raise ValueError(f"Tokenizer has only {len(words)} single-token words, need {needed}.")

    content_ids = words[:content_size]
    if noise_size == 1:
        noise_ids = tuple(tokenizer.encode(noise_token, add_special_tokens=False))
        if len(noise_ids) != 1:
            raise ValueError(f"noise_token {noise_token!r} is not a single token: {noise_ids}")
        if noise_ids[0] in content_ids:
            raise ValueError(f"noise_token {noise_token!r} overlaps with the content lexicon.")
    else:
        noise_ids = words[content_size : content_size + noise_size]

    lexicon = TokenLexicon(content_ids=content_ids, noise_ids=noise_ids)
    _LEXICON_CACHE[key] = lexicon
    return lexicon
