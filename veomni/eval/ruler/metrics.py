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

"""RULER scoring (``scripts/eval/synthetic/constants.py`` upstream), per sample in [0, 1].

Upstream reports the mean over samples x 100; aggregation is left to the caller.
"""

import re
from typing import Callable, Dict, Sequence


_CONTROL_CHARS = re.compile(r"[\x00-\x1f]")


def postprocess_pred(text: str) -> str:
    """Upstream ``postprocess_pred``: strip, and turn control characters into newlines."""
    return _CONTROL_CHARS.sub("\n", text.strip()).strip()


def string_match_all(pred: str, refs: Sequence[str]) -> float:
    """Fraction of references that occur (case-insensitive) in the prediction."""
    pred = pred.lower()
    return sum(ref.lower() in pred for ref in refs) / len(refs)


def string_match_part(pred: str, refs: Sequence[str]) -> float:
    """1 if any reference occurs (case-insensitive) in the prediction (QA: any accepted answer)."""
    pred = pred.lower()
    return float(any(ref.lower() in pred for ref in refs))


METRICS: Dict[str, Callable[[str, Sequence[str]], float]] = {
    "string_match_all": string_match_all,
    "string_match_part": string_match_part,
}
