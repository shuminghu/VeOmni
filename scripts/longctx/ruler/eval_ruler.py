"""RULER eval of an HF checkpoint: greedy generation + upstream string-match scoring, data parallel over torchrun.

    torchrun --nproc_per_node 8 scripts/longctx/ruler/eval_ruler.py --model_path <hf_ckpt> \
        --tokenizer_path ~/models/Qwen3-0.6B-Base --data_dir ~/datasets/ruler/qwen3 --out_dir <run>/eval/ruler_step<N>

Prompts come from ``prepare_ruler.py``; the model continues ``input + answer_prefix`` (upstream ``base`` template,
greedy, no stop words, ``tokens_to_generate`` new tokens or eos). Writes ``results.json`` (score per task x length
in %, averages, NIAH accuracy vs needle distance from the question) and ``predictions.jsonl``.
"""

import argparse
import glob
import json
import math
import os
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, GenerationConfig

from veomni.eval.ruler import METRICS, RULER_CATEGORIES, RULER_TASKS, RulerSample, postprocess_pred
from veomni.eval.utils import add_qat_args, gather, init_dist, load_causal_lm, ternary_from_args, token_budget_batches


def load_samples(data_dir: str, tasks, lengths, limit) -> List[RulerSample]:
    samples = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*", "L*.jsonl"))):
        task, length = os.path.basename(os.path.dirname(path)), int(os.path.basename(path)[1:-6])
        if (tasks and task not in tasks) or (lengths and length not in lengths):
            continue
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        samples.extend(RulerSample.from_dict(r) for r in rows[:limit])
    return samples


@torch.no_grad()
def generate(model, tokenizer, prompts: List[List[int]], max_new_tokens: int, device) -> List[str]:
    """Greedy continuation of left-padded prompts; returns decoded new text, cut at eos."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(p) for p in prompts)
    input_ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, p in enumerate(prompts):
        input_ids[i, max_len - len(p) :] = torch.as_tensor(p)
        attention_mask[i, max_len - len(p) :] = 1
    # use_cache explicitly: some checkpoints (early qwen3_hybrid) ship use_cache=false, and generate would then
    # silently recompute the whole prefix at every step.
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    out = model.generate(
        input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), generation_config=config
    )
    texts = []
    for row in out[:, max_len:].tolist():
        if tokenizer.eos_token_id in row:
            row = row[: row.index(tokenizer.eos_token_id)]
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return texts


def score_sample(sample: RulerSample, raw_pred: str, stop: List[str]) -> dict:
    pred = raw_pred
    for s in stop:
        pred = pred.split(s)[0]
    pred = postprocess_pred(pred)
    hits = [ref.lower() in pred.lower() for ref in sample.outputs]
    return {
        "task": sample.task,
        "max_seq_len": sample.max_seq_len,
        "index": sample.index,
        "prompt_tokens": sample.length - sample.tokens_to_generate,
        "score": METRICS[sample.metric](pred, sample.outputs),
        "hits": hits,
        "outputs": sample.outputs,
        "pred": pred,
        "answer_token_positions": sample.meta.get("answer_token_positions"),
    }


def aggregate(records: List[dict]) -> dict:
    by_cell = defaultdict(list)
    for r in records:
        by_cell[(r["task"], r["max_seq_len"])].append(r["score"])
    tasks = [t for t in RULER_TASKS if any(k[0] == t for k in by_cell)] + sorted(
        {k[0] for k in by_cell} - set(RULER_TASKS)
    )
    lengths = sorted({k[1] for k in by_cell})
    scores = {
        t: {str(L): round(100 * float(np.mean(by_cell[(t, L)])), 2) for L in lengths if (t, L) in by_cell}
        for t in tasks
    }
    counts = {t: {str(L): len(by_cell[(t, L)]) for L in lengths if (t, L) in by_cell} for t in tasks}

    def mean_over(names, L):
        vals = [scores[t][str(L)] for t in names if str(L) in scores.get(t, {})]
        return round(float(np.mean(vals)), 2) if vals else None

    average = {str(L): mean_over(tasks, L) for L in lengths}
    categories = {c: {str(L): mean_over(names, L) for L in lengths} for c, names in RULER_CATEGORIES.items()}

    # NIAH: per-answer hit rate vs distance (tokens) from the answer to the end of the prompt, log2 bins.
    # Separates "the needle is further back than the training length" from "the prompt is long".
    by_distance = defaultdict(lambda: defaultdict(list))
    for r in records:
        if not r["task"].startswith("niah") or not r["answer_token_positions"]:
            continue
        for hit, pos in zip(r["hits"], r["answer_token_positions"]):
            distance = max(1, r["prompt_tokens"] - pos)
            by_distance[str(r["max_seq_len"])][2 ** int(math.floor(math.log2(distance)))].append(hit)
    distance_summary = {
        L: {str(b): {"acc": round(100 * float(np.mean(v)), 2), "n": len(v)} for b, v in sorted(bins.items())}
        for L, bins in sorted(by_distance.items(), key=lambda kv: int(kv[0]))
    }
    return {
        "scores": scores,
        "counts": counts,
        "average": average,
        "categories": categories,
        "niah_by_distance": distance_summary,
    }


def print_table(results: dict) -> None:
    lengths = sorted({int(L) for row in results["scores"].values() for L in row})
    print(f"{'task':<18}" + "".join(f"{L:>8}" for L in lengths))
    for task, row in list(results["scores"].items()) + [("AVERAGE", results["average"])]:
        cells = [row.get(str(L)) for L in lengths]
        print(f"{task:<18}" + "".join(f"{c:>8.1f}" if c is not None else f"{'-':>8}" for c in cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--lengths", nargs="+", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Samples per (task, length).")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--max_tokens", type=int, default=131072, help="Prompt-token budget per generate batch.")
    parser.add_argument("--max_batch_size", type=int, default=64)
    parser.add_argument(
        "--stop", nargs="*", default=[], help="Cut predictions at these strings before scoring (upstream: none)."
    )
    add_qat_args(parser)
    args = parser.parse_args()

    rank, world_size, device = init_dist()
    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(args.tokenizer_path or args.model_path))
    model = load_causal_lm(args.model_path, device, args.attn_implementation, ternary_from_args(args))
    max_pos = getattr(model.config, "max_position_embeddings", None)

    samples = load_samples(os.path.expanduser(args.data_dir), args.tasks, args.lengths, args.limit)
    if max_pos is not None:
        samples = [s for s in samples if s.max_seq_len <= max_pos]
    # Interleave by length so every rank gets the same mix of long and short prompts.
    samples.sort(key=lambda s: (s.length, s.task, s.index))
    local = samples[rank::world_size]
    if rank == 0:
        print(f"{len(samples)} prompts, {len(local)} on rank 0")

    start = time.time()
    records = []
    by_budget: Dict[int, List[int]] = defaultdict(list)  # batches share max_new_tokens
    for i, s in enumerate(local):
        by_budget[s.tokens_to_generate].append(i)
    for budget, idxs in sorted(by_budget.items()):
        prompts = [tokenizer.encode(local[i].prompt, add_special_tokens=False) for i in idxs]
        for batch in token_budget_batches([len(p) for p in prompts], args.max_tokens):
            for chunk_start in range(0, len(batch), args.max_batch_size):
                chunk = batch[chunk_start : chunk_start + args.max_batch_size]
                preds = generate(model, tokenizer, [prompts[j] for j in chunk], budget, device)
                records.extend(score_sample(local[idxs[j]], p, args.stop) for j, p in zip(chunk, preds))
    elapsed = time.time() - start
    records = gather(records, world_size)

    if rank == 0:
        results = {"model_path": args.model_path, "args": vars(args), "seconds": round(elapsed, 1)}
        results.update(aggregate(records))
        out_dir = os.path.expanduser(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        manifest = os.path.join(os.path.expanduser(args.data_dir), "manifest.json")
        if os.path.exists(manifest):
            with open(manifest) as f:
                m = json.load(f)
            results["data"] = {k: m.get(k) for k in ("tokenizer", "seed", "sentence_splitter", "essay_fingerprint")}
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        with open(os.path.join(out_dir, "predictions.jsonl"), "w") as f:
            for r in sorted(records, key=lambda r: (r["task"], r["max_seq_len"], r["index"])):
                f.write(json.dumps(r) + "\n")
        print_table(results)
        print(f"generation took {elapsed:.0f}s")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
