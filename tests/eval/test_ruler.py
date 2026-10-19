import json
import os
import re
import sys
from collections import Counter


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tools import hf_local_or_remote
from transformers import AutoTokenizer

from veomni.eval.ruler import (
    RULER_TASKS,
    RulerResources,
    RulerSample,
    build_ruler_task,
    largest_fitting,
    postprocess_pred,
    string_match_all,
    string_match_part,
)
from veomni.eval.ruler.aggregation import riemann_zeta


MAX_SEQ_LEN = 2048


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(hf_local_or_remote("Qwen/Qwen3-0.6B-Base"))


@pytest.fixture(scope="module")
def resources(tmp_path_factory):
    """Stand-ins for the downloaded essay / QA files, so the tests need no network data."""
    root = tmp_path_factory.mktemp("ruler_raw")
    sentences = [
        f"Essay sentence number {i} talks about startups and {w}." for i, w in enumerate(["ideas", "money"] * 400)
    ]
    with open(root / "PaulGrahamEssays.json", "w") as f:
        json.dump({"text": "  ".join(sentences)}, f)
    squad = {"data": []}
    for a in range(30):
        paragraphs = []
        for p in range(3):
            context = f"Article {a} paragraph {p}. " + "Filler text about the topic. " * 20 + f"The code is A{a}P{p}."
            qas = [
                {
                    "question": f"What is the code of article {a} paragraph {p}?",
                    "is_impossible": False,
                    "answers": [{"text": f"A{a}P{p}"}],
                }
            ]
            paragraphs.append({"context": context, "qas": qas})
        squad["data"].append({"paragraphs": paragraphs})
    with open(root / "squad.json", "w") as f:
        json.dump(squad, f)
    hotpot = [
        {
            "question": f"Question {q}?",
            "answer": f"answer{q}",
            "context": [
                [f"Title {q}-{i}", [f"Paragraph {i} of question {q} says answer{q}. " * 3]] for i in range(10)
            ],
        }
        for q in range(30)
    ]
    with open(root / "hotpotqa.json", "w") as f:
        json.dump(hotpot, f)
    return RulerResources(essay_path=str(root / "PaulGrahamEssays.json"), qa_dir=str(root), sentence_splitter="regex")


@pytest.mark.parametrize("name", list(RULER_TASKS))
def test_standard_tasks_fit_length_budget(tokenizer, resources, name):
    task = build_ruler_task(name, tokenizer, resources)
    samples = [task.generate(i, MAX_SEQ_LEN) for i in range(4)]
    for s in samples:
        num_tokens = len(tokenizer.encode(s.prompt, add_special_tokens=False))
        assert s.length == num_tokens + task.tokens_to_generate <= MAX_SEQ_LEN
        # Filled close to the budget: only QA (whole documents) is coarse.
        assert s.length >= MAX_SEQ_LEN * (0.5 if task.family == "qa" else 0.85)
        assert s.outputs and all(isinstance(o, str) and o for o in s.outputs)
        assert RulerSample.from_dict(json.loads(json.dumps(s.to_dict()))) == s


@pytest.mark.parametrize("name", ["niah_multikey_2", "vt", "cwe", "fwe", "qa_1"])
def test_generation_is_deterministic_per_sample(tokenizer, resources, name):
    a = build_ruler_task(name, tokenizer, resources)
    b = build_ruler_task(name, tokenizer, resources)
    assert a.generate(3, MAX_SEQ_LEN).to_dict() == b.generate(3, MAX_SEQ_LEN).to_dict()
    assert a.generate(3, MAX_SEQ_LEN).input != a.generate(4, MAX_SEQ_LEN).input
    assert a.generate(3, MAX_SEQ_LEN, seed=1).input != a.generate(3, MAX_SEQ_LEN, seed=2).input


@pytest.mark.parametrize("name", [n for n in RULER_TASKS if n.startswith("niah")])
def test_niah_needles_and_answer_positions(tokenizer, resources, name):
    task = build_ruler_task(name, tokenizer, resources)
    sample = task.generate(0, MAX_SEQ_LEN)
    expected_needles = task.num_needle_k * task.num_needle_v
    if task.type_haystack == "needle":
        assert sample.input.count("One of the special magic") == sample.meta["haystack_size"] + expected_needles
    else:
        assert sample.input.count("One of the special magic") == expected_needles
    single = task.num_needle_q * task.num_needle_v == 1
    assert sample.input.startswith("A special magic" if single else "Some special magic")
    assert sample.answer_prefix.endswith("mentioned in the provided text " + ("is" if single else "are"))
    ids = tokenizer.encode(sample.input, add_special_tokens=False)
    for answer, pos in zip(sample.outputs, sample.meta["answer_token_positions"]):
        assert f"is: {answer}." in sample.input
        window = tokenizer.decode(ids[pos : pos + len(tokenizer.encode(answer, add_special_tokens=False)) + 1])
        assert answer[:3] in window


def test_niah_distractor_keys_never_match_query(tokenizer, resources):
    task = build_ruler_task("niah_multikey_2", tokenizer, resources)
    for i in range(4):
        sample = task.generate(i, MAX_SEQ_LEN)
        key = re.search(r"special magic number for (\S+) mentioned", sample.answer_prefix).group(1)
        assert sample.input.count(f"for {key} is:") == 1


def test_variable_tracking_chain_and_icl(tokenizer, resources):
    task = build_ruler_task("vt", tokenizer, resources)
    sample = task.generate(0, MAX_SEQ_LEN)
    names = sample.outputs
    assert len(names) == task.num_hops + 1 and all(re.fullmatch(r"[A-Z]{5}", n) for n in names)
    value = re.search(r"assigned the value (\d+)", sample.answer_prefix).group(1)
    assert f"VAR {names[0]} = {value}" in sample.input
    for prev, cur in zip(names, names[1:]):
        assert f"VAR {cur} = VAR {prev} " in sample.input
    # One solved in-context example (3-letter names) precedes the query prompt.
    assert sample.input.count("Memorize and track") == 2
    icl = sample.input.split("\n\nMemorize and track")[0]
    assert re.search(r"they are:  [A-Z]{3}( [A-Z]{3}){%d}$" % task.num_hops, icl)


@pytest.mark.parametrize("max_seq_len", [2048, 4096])
def test_cwe_common_words_are_most_frequent(tokenizer, resources, max_seq_len):
    task = build_ruler_task("cwe", tokenizer, resources)
    sample = task.generate(0, max_seq_len)
    query_list = sample.input.split("\n")[-2]  # the query's own numbered list (after the in-context example)
    counts = Counter(re.findall(r"\d+\. (.+?)(?= \d+\. |$)", query_list))
    common_repeats = 6 if max_seq_len < 4096 else task.freq_cw
    assert len(sample.outputs) == task.num_cw
    assert all(counts[w] == common_repeats for w in sample.outputs)
    assert max(c for w, c in counts.items() if w not in sample.outputs) < common_repeats


def test_fwe_answer_is_ranks_two_to_four(tokenizer, resources):
    task = build_ruler_task("fwe", tokenizer, resources)
    sample = task.generate(0, MAX_SEQ_LEN)
    context = sample.input.split("coded words. ")[1].split("\nQuestion:")[0]
    counts = Counter(context.split())
    ranked = [w for w, _ in counts.most_common()]
    assert ranked[0] == "..."
    assert set(ranked[1:4]) == set(sample.outputs)
    assert counts[sample.outputs[-1]] > counts[ranked[4]]


@pytest.mark.parametrize("name", ["qa_1", "qa_2"])
def test_qa_contains_gold_documents(tokenizer, resources, name):
    task = build_ruler_task(name, tokenizer, resources)
    sample = task.generate(0, MAX_SEQ_LEN)
    for doc_id in sample.meta["gold_docs"]:
        assert f"Document {doc_id}:\n" in sample.input
    assert any(o in sample.input for o in sample.outputs)
    assert sample.metric == "string_match_part"


def test_qa_too_long_gold_documents_raise(tokenizer, resources):
    task = build_ruler_task("qa_2", tokenizer, resources)  # 10 paragraphs ~ 400 tokens
    with pytest.raises(ValueError):
        task.fitted_size(256, seed=42)


def test_custom_task_from_family(tokenizer, resources):
    task = build_ruler_task("niah:niah_noise_uuid", tokenizer, resources, type_haystack="noise", type_needle_v="uuids")
    sample = task.generate(0, 1024)
    assert sample.task == "niah_noise_uuid"
    assert re.fullmatch(r"[0-9a-f-]{36}", sample.outputs[0])


def test_largest_fitting():
    def count(n):
        return 3 * n + 7

    for budget in (10, 100, 1000, 12345):
        for guess in (1, 5, 50, 10**6):
            size = largest_fitting(count, budget, 1, guess)
            assert count(size) <= budget < count(size + 1)
    with pytest.raises(ValueError):
        largest_fitting(count, 5, 1, 1)


def test_metrics():
    assert postprocess_pred("  ans\x07wer \n") == "ans\nwer"
    assert string_match_all("The numbers are 123 and 456.", ["123", "456"]) == 1.0
    assert string_match_all("The number is 123.", ["123", "456"]) == 0.5
    assert string_match_part("PARIS, France", ["paris", "Lyon"]) == 1.0
    assert string_match_part("Lyon", ["Paris"]) == 0.0


def test_riemann_zeta():
    assert riemann_zeta(2.0) == pytest.approx(3.14159265358979**2 / 6, rel=1e-10)
