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

"""Multi-document QA (upstream ``scripts/data/synthetic/qa.py``): gold paragraphs padded with distractors.

Sample ``i`` asks question ``order[i]`` of a fixed (seeded) permutation, so the same questions appear at every
length. Upstream asks question ``i``, so its first samples all come from the first articles (SQuAD: 'Normans').
SQuAD distractors come first from the same article, then from the rest of the dev set; HotpotQA uses its
distractor-setting paragraphs, then other questions' paragraphs.
"""

import random
from functools import cached_property
from typing import List

from .base import RULER_FAMILY_REGISTRY, Rendered, RulerTask


DOCUMENT_PROMPT = "Document {i}:\n{document}"


@RULER_FAMILY_REGISTRY.register("qa")
class MultiDocQA(RulerTask):
    family = "qa"
    tokens_to_generate = 32
    metric = "string_match_part"
    template = (
        "Answer the question based on the given documents. Only give me the answer and do not output any other "
        "words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given "
        "documents. Only give me the answer and do not output any other words.\n\nQuestion: {query}"
    )
    answer_prefix = " Answer:"
    probe_size = 10

    def __init__(self, name, tokenizer, resources, dataset: str = "squad"):
        super().__init__(name, tokenizer, resources)
        if dataset not in ("squad", "hotpotqa"):
            raise ValueError(f"dataset must be squad / hotpotqa, got {dataset!r}")
        self.dataset = dataset

    def config(self):
        return {"dataset": self.dataset}

    @cached_property
    def question_order(self) -> List[int]:
        order = list(range(len(self.resources.qa(self.dataset)[0])))
        random.Random(f"ruler/qa/{self.dataset}").shuffle(order)
        return order

    def render(self, size, rng, max_seq_len, index):
        qas, docs = self.resources.qa(self.dataset)
        qa_index = self.question_order[index % len(qas)]
        qa = qas[qa_index]
        gold, more = qa["context"], qa.get("more_context", [])
        num_docs = max(size, len(gold))  # gold paragraphs are never dropped
        if num_docs >= len(docs):
            chosen = (list(range(len(docs))) * (num_docs // len(docs) + 1))[:num_docs]
        elif num_docs - len(gold) > len(more):
            taken = set(gold + more)
            pool = [i for i in range(len(docs)) if i not in taken]
            chosen = gold + more + rng.sample(pool, num_docs - len(gold) - len(more))
        else:
            chosen = gold + rng.sample(more, num_docs - len(gold))
        rng.shuffle(chosen)
        context = "\n\n".join(DOCUMENT_PROMPT.format(i=i + 1, document=docs[d]) for i, d in enumerate(chosen))
        return Rendered(
            input=self.template.format(context=context, query=qa["query"]),
            answer_prefix=self.answer_prefix,
            outputs=qa["outputs"],
            meta={"qa_index": qa_index, "gold_docs": [chosen.index(g) + 1 for g in gold]},
        )
