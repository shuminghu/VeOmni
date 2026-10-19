# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Qwen3-Hybrid (KDA + NoPE full attention) modeling, written directly against ``VeomniOp`` handles."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from veomni.models.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY

from .configuration_qwen3_hybrid import Qwen3HybridConfig
from .modeling_qwen3_hybrid import Qwen3HybridForCausalLM, Qwen3HybridModel


# ``qwen3_hybrid`` is not a transformers model type: also register it on the Auto classes, so its HF checkpoints
# load outside the trainer (e.g. ``scripts/longctx/eval_longctx.py``) once ``veomni`` is imported.
AutoConfig.register(Qwen3HybridConfig.model_type, Qwen3HybridConfig, exist_ok=True)
AutoModel.register(Qwen3HybridConfig, Qwen3HybridModel, exist_ok=True)
AutoModelForCausalLM.register(Qwen3HybridConfig, Qwen3HybridForCausalLM, exist_ok=True)


@MODEL_CONFIG_REGISTRY.register("qwen3_hybrid")
def register_qwen3_hybrid_config():
    return Qwen3HybridConfig


@MODELING_REGISTRY.register("qwen3_hybrid")
def register_qwen3_hybrid_modeling(architecture: str):
    if "ForCausalLM" in architecture:
        return Qwen3HybridForCausalLM
    if "Model" in architecture:
        return Qwen3HybridModel
    return Qwen3HybridForCausalLM
