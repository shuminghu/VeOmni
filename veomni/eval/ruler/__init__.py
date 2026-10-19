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

"""RULER long-context benchmark (Hsieh et al., 2024, "RULER: What's the Real Context Size of Your Long-Context
Language Models?"), ported from NVIDIA/RULER for prompt generation and scoring.

    resources = RulerResources(essay_path=".../PaulGrahamEssays.json", qa_dir=".../raw")
    task = build_ruler_task("niah_multikey_2", tokenizer, resources)
    sample = task.generate(index=0, max_seq_len=8192)        # prompt = sample.input + sample.answer_prefix
    score = METRICS[sample.metric](postprocess_pred(generated_text), sample.outputs)

Prepare / evaluate with ``scripts/longctx/ruler/{prepare_ruler,eval_ruler}.py``.
"""

from . import aggregation, niah, qa, variable_tracking  # noqa: F401  (register the task families)
from .base import (
    RULER_CATEGORIES,
    RULER_FAMILY_REGISTRY,
    RULER_TASKS,
    RulerSample,
    RulerTask,
    build_ruler_task,
    largest_fitting,
)
from .metrics import METRICS, postprocess_pred, string_match_all, string_match_part
from .resources import RulerResources


__all__ = [
    "METRICS",
    "RULER_CATEGORIES",
    "RULER_FAMILY_REGISTRY",
    "RULER_TASKS",
    "RulerResources",
    "RulerSample",
    "RulerTask",
    "build_ruler_task",
    "largest_fitting",
    "postprocess_pred",
    "string_match_all",
    "string_match_part",
]
