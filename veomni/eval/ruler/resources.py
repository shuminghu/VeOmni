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

"""Raw material for RULER tasks: word lists, the essay haystack, QA datasets, sentence splitting.

- ``assets/{noun,adjective,verb}list.txt``: copied from ``wonderwords`` 3.0.1 (MIT), which upstream RULER reads.
- Essay haystack: Paul Graham essays (``PaulGrahamEssays.json``), QA: SQuAD v2 dev / HotpotQA dev distractor;
  both fetched by ``scripts/longctx/ruler/download_ruler_data.py``.
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Callable, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")

# RULER's filler sentence for the "noise" haystack.
NOISE_SENTENCE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."


@lru_cache(maxsize=None)
def load_word_asset(name: str) -> Tuple[str, ...]:
    with open(os.path.join(ASSET_DIR, f"{name}list.txt")) as f:
        return tuple(line.rstrip() for line in f)


def _regex_sent_tokenize(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def get_sentence_splitter(kind: str = "auto") -> Tuple[str, Callable[[str], List[str]]]:
    """``nltk`` punkt (what upstream uses) or a regex fallback; returns ``(name, fn)``."""
    if kind in ("auto", "nltk"):
        try:
            from nltk.tokenize import sent_tokenize

            sent_tokenize("Probe. Sentence.")
            return "nltk", sent_tokenize
        except (ImportError, LookupError) as e:
            if kind == "nltk":
                raise RuntimeError(
                    "nltk sentence splitter requested but unavailable (pip install nltk; nltk.download('punkt_tab'))"
                ) from e
            logger.warning(f"nltk punkt unavailable ({e}); falling back to a regex sentence splitter.")
    return "regex", _regex_sent_tokenize


@dataclass
class RulerResources:
    """Lazily loaded inputs shared by all tasks. Paths are only needed by the tasks that use them."""

    essay_path: Optional[str] = None
    qa_dir: Optional[str] = None
    sentence_splitter: str = "auto"

    @cached_property
    def nouns(self) -> Tuple[str, ...]:
        return load_word_asset("noun")

    @cached_property
    def adjectives(self) -> Tuple[str, ...]:
        return load_word_asset("adjective")

    @cached_property
    def verbs(self) -> Tuple[str, ...]:
        return load_word_asset("verb")

    @cached_property
    def splitter(self) -> Tuple[str, Callable[[str], List[str]]]:
        return get_sentence_splitter(self.sentence_splitter)

    @cached_property
    def essay_words(self) -> List[str]:
        """The essay corpus as whitespace-separated words (upstream: ``re.sub(r'\\s+', ' ', text).split(' ')``)."""
        if not self.essay_path:
            raise ValueError("This task needs the essay haystack: pass essay_path (PaulGrahamEssays.json).")
        with open(os.path.expanduser(self.essay_path)) as f:
            text = json.load(f)["text"]
        return re.sub(r"\s+", " ", text).split(" ")

    @cached_property
    def essay_fingerprint(self) -> str:
        return hashlib.sha1(" ".join(self.essay_words).encode()).hexdigest()[:12]

    def qa(self, dataset: str) -> Tuple[List[Dict], List[str]]:
        """``(qas, docs)`` as built by upstream ``read_squad`` / ``read_hotpotqa``."""
        cache = self.__dict__.setdefault("_qa_cache", {})
        if dataset not in cache:
            if not self.qa_dir:
                raise ValueError(f"QA task '{dataset}' needs qa_dir (with squad.json / hotpotqa.json).")
            path = os.path.join(os.path.expanduser(self.qa_dir), f"{dataset}.json")
            readers = {"squad": read_squad, "hotpotqa": read_hotpotqa}
            cache[dataset] = readers[dataset](path)
        return cache[dataset]


def read_squad(path: str) -> Tuple[List[Dict], List[str]]:
    with open(path) as f:
        data = json.load(f)
    docs = sorted({p["context"] for d in data["data"] for p in d["paragraphs"]})
    doc_index = {c: i for i, c in enumerate(docs)}
    qas = []
    for d in data["data"]:
        article_docs = [doc_index[p["context"]] for p in d["paragraphs"]]
        for p in d["paragraphs"]:
            own = doc_index[p["context"]]
            for qa in p["qas"]:
                if not qa["is_impossible"]:
                    qas.append(
                        {
                            "query": qa["question"],
                            "outputs": [a["text"] for a in qa["answers"]],
                            "context": [own],
                            "more_context": [i for i in article_docs if i != own],
                        }
                    )
    return qas, docs


def read_hotpotqa(path: str) -> Tuple[List[Dict], List[str]]:
    with open(path) as f:
        data = json.load(f)
    docs = sorted({f"{t}\n{''.join(p)}" for d in data for t, p in d["context"]})
    doc_index = {c: i for i, c in enumerate(docs)}
    qas = [
        {
            "query": d["question"],
            "outputs": [d["answer"]],
            "context": [doc_index[f"{t}\n{''.join(p)}"] for t, p in d["context"]],
        }
        for d in data
    ]
    return qas, docs
