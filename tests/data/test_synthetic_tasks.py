import os
import re
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from tools import hf_local_or_remote
from transformers import AutoTokenizer

from veomni.data.data_transform import process_pretrain_mix_example
from veomni.data.synthetic import build_lexicon, get_synthetic_task
from veomni.data.synthetic.harness import NL2_TEST_PARTS, NL2_TRAIN_PARTS, SELECTIVE_COPY_TEMPLATES
from veomni.data.synthetic.lexicon import CONTRACTION_STEMS, load_wordlist
from veomni.data.synthetic.selective_copy import PLACEMENT_SPANS
from veomni.utils.constants import IGNORE_INDEX


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(hf_local_or_remote("Qwen/Qwen3-0.6B-Base"))


@pytest.fixture(scope="module")
def task(tokenizer):
    return get_synthetic_task("selective_copy", tokenizer)


def _recipe(**overrides):
    recipe = {"task": "selective_copy", "seed": 0, "ctx_len": 64, "n_copy": 8}
    recipe.update(overrides)
    return recipe


@pytest.mark.parametrize("noise_size", [1, 16])
def test_lexicon_single_token_and_disjoint(tokenizer, noise_size):
    lexicon = build_lexicon(tokenizer, content_size=512, noise_size=noise_size)
    assert len(lexicon.content_ids) == len(set(lexicon.content_ids)) == 512
    assert len(lexicon.noise_ids) == noise_size
    assert not set(lexicon.content_ids) & set(lexicon.noise_ids)
    for token_id in lexicon.content_ids[:64] + lexicon.noise_ids:
        assert tokenizer.encode(tokenizer.decode([token_id]), add_special_tokens=False) == [token_id]


def test_wordlist_has_no_contraction_stems():
    assert not CONTRACTION_STEMS & set(load_wordlist())


@pytest.mark.parametrize("fmt", ["nl", "nl2", "raw"])
@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("noise_size", [1, 16])
def test_all_templates_are_canonical(task, fmt, split, noise_size):
    num_templates = len(task.templates[fmt][split])
    for template_id in range(num_templates):
        for seed in range(4 if num_templates <= 16 else 1):
            sample = task.build(
                _recipe(seed=seed, fmt=fmt, template_split=split, template_id=template_id, noise_size=noise_size)
            )
            task.check_canonical(sample)


def test_nl2_body_start_not_marked_by_punctuation():
    """Guards the v1 shortcut: in ``nl`` the only prefix colon sits right before the body."""
    prefixes = [t.prefix for t in SELECTIVE_COPY_TEMPLATES["nl2"]["train"]]
    word_after_inner_colon = re.compile(r": [A-Za-z]")
    assert sum(bool(word_after_inner_colon.search(p)) for p in prefixes) / len(prefixes) > 0.4
    assert sum(p.endswith(":") for p in prefixes) / len(prefixes) < 0.5
    train_parts, test_parts = NL2_TRAIN_PARTS, NL2_TEST_PARTS
    for name in ("intros", "markers", "closers", "questions"):
        assert not set(getattr(train_parts, name)) & set(getattr(test_parts, name)), name


@pytest.mark.parametrize("ctx_len,n_copy", [(16, 1), (64, 8), (1000, 64)])
def test_layout_and_labels(task, tokenizer, ctx_len, n_copy):
    sample = task.build(_recipe(seed=3, ctx_len=ctx_len, n_copy=n_copy))
    meta = sample.meta
    lexicon = build_lexicon(tokenizer)

    assert len(sample.input_ids) == len(sample.labels) <= ctx_len + task.max_overhead(n_copy)
    assert meta["query_pos"] - meta["body_start"] == ctx_len

    body = sample.input_ids[meta["body_start"] : meta["query_pos"]]
    content_positions = {p - meta["body_start"] for p in meta["positions"]}
    assert len(content_positions) == n_copy
    for i, token_id in enumerate(body):
        if i in content_positions:
            assert token_id in lexicon.content_ids
        else:
            assert token_id in lexicon.noise_ids
    assert [sample.input_ids[p] for p in meta["positions"]] == meta["content"]

    answer = meta["content"] + [tokenizer.eos_token_id]
    assert sample.label_positions == list(range(meta["answer_start"], len(sample.input_ids)))
    assert [sample.labels[i] for i in sample.label_positions] == answer
    assert sample.input_ids[meta["answer_start"] :] == answer
    assert all(label == IGNORE_INDEX for label in sample.labels[: meta["answer_start"]])
    assert meta["distances"] == [meta["query_pos"] - p for p in meta["positions"]]


def test_deterministic_given_seed(task):
    a, b = task.build(_recipe(seed=11)), task.build(_recipe(seed=11))
    c = task.build(_recipe(seed=12))
    assert a.input_ids == b.input_ids and a.meta == b.meta
    assert a.input_ids != c.input_ids


def test_pinned_layout_overrides_seed(task, tokenizer):
    content = list(build_lexicon(tokenizer).content_ids[:3])
    sample = task.build(
        _recipe(seed=5, ctx_len=20, n_copy=3, template_id=0, body_positions=[0, 7, 19], content=content)
    )
    body_start = sample.meta["body_start"]
    assert sample.meta["positions"] == [body_start, body_start + 7, body_start + 19]
    assert sample.meta["content"] == content


@pytest.mark.parametrize("placement", sorted(PLACEMENT_SPANS))
def test_placement_span(task, placement):
    lo, hi = PLACEMENT_SPANS[placement]
    ctx_len = 400
    for seed in range(8):
        meta = task.build(_recipe(seed=seed, ctx_len=ctx_len, n_copy=16, placement=placement)).meta
        body_positions = [p - meta["body_start"] for p in meta["positions"]]
        assert int(lo * ctx_len) <= min(body_positions) and max(body_positions) < int(hi * ctx_len)


def test_score(task, tokenizer):
    sample = task.build(_recipe(n_copy=4))
    perfect = [sample.labels[i] for i in sample.label_positions]
    result = task.score(sample, perfect)
    assert result["token_acc"] == 1.0 and result["exact"] and result["eos_correct"]

    wrong_first = [perfect[1]] + perfect[1:]
    result = task.score(sample, wrong_first)
    assert result["token_acc"] == 0.75 and not result["exact"]


def test_pretrain_mix_transform(tokenizer, task):
    recipe = _recipe(ctx_len=100, n_copy=8)
    (sample,) = process_pretrain_mix_example(recipe, tokenizer=tokenizer, max_seq_len=512)
    expected = task.build(recipe)
    assert sample["input_ids"].tolist() == expected.input_ids
    assert sample["labels"].tolist() == expected.labels
    assert sample["attention_mask"].dtype == torch.long and int(sample["attention_mask"].sum()) == len(
        expected.input_ids
    )

    text_samples = process_pretrain_mix_example({"text": "hello world " * 50}, tokenizer=tokenizer, max_seq_len=32)
    assert len(text_samples) > 1
    assert all(torch.equal(s["input_ids"], s["labels"]) for s in text_samples)

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        process_pretrain_mix_example(recipe, tokenizer=tokenizer, max_seq_len=64)
