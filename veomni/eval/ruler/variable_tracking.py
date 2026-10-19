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

"""Variable tracking (upstream ``scripts/data/synthetic/variable_tracking.py``): multi-hop tracing.

Chains ``VAR ABCDE = 12345``, ``VAR FGHIJ = VAR ABCDE``, ... are scattered in the haystack; the model must list
every variable holding the value. Each prompt starts with one solved in-context example (~500 tokens, 3-letter
names), as upstream always does.
"""

import heapq
import random
import string
from functools import cached_property
from typing import List, Tuple

from .base import RULER_FAMILY_REGISTRY, Rendered, RulerTask, largest_fitting
from .niah import essay_sentences, insert_at_depths
from .resources import NOISE_SENTENCE


ICL_MAX_TOKENS = 500


def interleave_chains(chains: List[List[str]], rng: random.Random) -> List[str]:
    """Random merge of the chains that keeps each chain's order (upstream ``shuffle_sublists_heap``)."""
    heap = [(rng.random(), i, 0) for i in range(len(chains))]
    heapq.heapify(heap)
    merged = []
    while heap:
        _, chain_idx, elem_idx = heapq.heappop(heap)
        merged.append(chains[chain_idx][elem_idx])
        if elem_idx + 1 < len(chains[chain_idx]):
            heapq.heappush(heap, (rng.random(), chain_idx, elem_idx + 1))
    return merged


@RULER_FAMILY_REGISTRY.register("variable_tracking")
class VariableTracking(RulerTask):
    family = "variable_tracking"
    tokens_to_generate = 30
    template = (
        "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\n"
        "Question: Find all variables that are assigned the value {query} in the text above."
    )
    answer_prefix = (
        " Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are "
        "assigned the value {query}, they are: "
    )

    def __init__(
        self, name, tokenizer, resources, type_haystack: str = "noise", num_chains: int = 1, num_hops: int = 4
    ):
        super().__init__(name, tokenizer, resources)
        if type_haystack not in ("noise", "essay"):
            raise ValueError(f"type_haystack must be noise / essay, got {type_haystack!r}")
        self.type_haystack = type_haystack
        self.num_chains = num_chains
        self.num_hops = num_hops
        self.min_size = (num_hops + 1) * num_chains if type_haystack == "noise" else 1
        self.probe_size = 500 if type_haystack == "essay" else 50

    def config(self):
        return {"type_haystack": self.type_haystack, "num_chains": self.num_chains, "num_hops": self.num_hops}

    def chains(self, rng: random.Random, name_len: int, num_hops: int) -> Tuple[List[List[str]], List[List[str]]]:
        names: List[str] = []
        while len(names) < (num_hops + 1) * self.num_chains:
            name = "".join(rng.choices(string.ascii_uppercase, k=name_len))
            if name not in names:
                names.append(name)
        all_vars, all_chains = [], []
        for c in range(self.num_chains):
            chain_vars = names[c * (num_hops + 1) : (c + 1) * (num_hops + 1)]
            chain = [f"VAR {chain_vars[0]} = {rng.randint(10000, 99998)}"]
            chain += [f"VAR {chain_vars[j + 1]} = VAR {chain_vars[j]} " for j in range(num_hops)]
            all_vars.append(chain_vars)
            all_chains.append(chain)
        return all_vars, all_chains

    def render_chain_prompt(self, size: int, rng: random.Random, name_len: int, num_hops: int):
        """One full prompt (template + answer prefix) for a chain set; returns (input, answer_prefix, vars)."""
        chain_vars, chains = self.chains(rng, name_len, num_hops)
        value = chains[0][0].split("=")[-1].strip()
        if self.type_haystack == "essay":
            _, splitter = self.resources.splitter
            items = [line.strip() + "." for line in interleave_chains(chains, rng)]
            context = insert_at_depths(essay_sentences(self.resources.essay_words, size, splitter), items, rng)
        else:
            lines = [NOISE_SENTENCE] * size
            for chain in chains:
                for j, position in enumerate(sorted(rng.sample(range(len(lines)), len(chain)))):
                    lines.insert(position + j, chain[j])
            context = "\n".join(lines)
        context = context.replace(". \n", ".\n")
        text = self.template.format(context=context, query=value)
        prefix = self.answer_prefix.format(num_v=num_hops + 1, query=value)
        return text, prefix, chain_vars[0]

    @cached_property
    def icl_size(self) -> int:
        """Haystack size of the in-context example (upstream: fit to 500 tokens, no generation budget)."""
        hops = min(10, self.num_hops)

        def count(size: int) -> int:
            text, prefix, _ = self.render_chain_prompt(size, random.Random("ruler/vt_icl"), 3, hops)
            return self.num_tokens(text + prefix)

        return largest_fitting(count, ICL_MAX_TOKENS, self.min_size, 10)

    def render(self, size, rng, max_seq_len, index):
        icl_text, icl_prefix, icl_vars = self.render_chain_prompt(self.icl_size, rng, 3, min(10, self.num_hops))
        icl = icl_text + icl_prefix + " " + " ".join(icl_vars) + "\n"
        text, prefix, chain_vars = self.render_chain_prompt(size, rng, 5, self.num_hops)
        text = icl + "\n" + text
        start = len(icl) + 1
        return Rendered(
            input=text,
            answer_prefix=prefix,
            outputs=chain_vars,
            meta={"answer_char_positions": [text.find(f"VAR {v} =", start) for v in chain_vars]},
        )
