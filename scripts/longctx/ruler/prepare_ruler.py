"""Generate RULER prompts for one tokenizer, one jsonl per (task, length).

    python scripts/longctx/ruler/prepare_ruler.py --tokenizer ~/models/Qwen3-0.6B-Base \
        --raw_dir ~/datasets/ruler/raw --out_dir ~/datasets/ruler/qwen3

Writes ``<out_dir>/<task>/L<length>.jsonl`` (``RulerSample`` rows) and ``<out_dir>/manifest.json``. Lengths count
prompt + ``tokens_to_generate`` tokens, as upstream's ``max_seq_length``. Raw data: ``download_ruler_data.py``.
(task, length) pairs that cannot be built (e.g. HotpotQA's 10 gold/distractor paragraphs exceed 1k tokens) are
skipped and listed in the manifest.
"""

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Optional

import numpy as np


_WORKER: Dict[str, object] = {}


def _init_worker(tokenizer_path: str, essay_path: Optional[str], qa_dir: Optional[str], splitter: str) -> None:
    from transformers import AutoTokenizer

    from veomni.eval.ruler import RulerResources

    _WORKER["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_path)
    _WORKER["resources"] = RulerResources(essay_path=essay_path, qa_dir=qa_dir, sentence_splitter=splitter)


def _build(task_name: str, max_seq_len: int, num_samples: int, seed: int, out_path: str) -> dict:
    from veomni.eval.ruler import build_ruler_task

    task = build_ruler_task(task_name, _WORKER["tokenizer"], _WORKER["resources"])
    info = {"task": task_name, "max_seq_len": max_seq_len, "family": task.family, "config": task.config()}
    start = time.time()
    try:
        task.fitted_size(max_seq_len, seed)
    except ValueError as e:
        return {**info, "skipped": str(e)}
    rows, failed = [], []
    for index in range(num_samples):
        try:
            rows.append(task.generate(index, max_seq_len, seed).to_dict())
        except ValueError as e:  # this sample alone cannot fit (e.g. very long gold QA paragraphs)
            failed.append({"index": index, "error": str(e)})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, out_path)
    lengths = [r["length"] for r in rows]
    return {
        **info,
        "num_samples": len(rows),
        "failed": failed,
        "length_mean": float(np.mean(lengths)),
        "length_min": int(min(lengths)),
        "haystack_size": int(np.median([r["meta"]["haystack_size"] for r in rows])),
        "seconds": round(time.time() - start, 1),
    }


def main():
    from veomni.eval.ruler import RULER_TASKS, RulerResources

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--raw_dir", default="~/datasets/ruler/raw", help="PaulGrahamEssays.json, squad/hotpotqa.json")
    parser.add_argument("--tasks", nargs="+", default=list(RULER_TASKS))
    parser.add_argument("--lengths", nargs="+", type=int, default=[1024, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--num_samples", type=int, default=100, help="Per (task, length); upstream uses 500.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sentence_splitter", default="nltk", choices=["auto", "nltk", "regex"])
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tokenizer = os.path.expanduser(args.tokenizer)
    out_dir = os.path.expanduser(args.out_dir)
    raw_dir = os.path.expanduser(args.raw_dir)
    essay_path = os.path.join(raw_dir, "PaulGrahamEssays.json")
    resources = RulerResources(essay_path=essay_path, qa_dir=raw_dir, sentence_splitter=args.sentence_splitter)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {"sets": {}}
    if manifest.get("seed", args.seed) != args.seed or manifest.get("tokenizer", tokenizer) != tokenizer:
        raise ValueError(f"{out_dir} was built with another seed / tokenizer; use a new --out_dir")

    jobs = []
    for task in args.tasks:
        for length in args.lengths:
            out_path = os.path.join(out_dir, task, f"L{length}.jsonl")
            if os.path.exists(out_path) and not args.overwrite:
                continue
            jobs.append((task, length, args.num_samples, args.seed, out_path))
    print(f"{len(jobs)} (task, length) sets to build with {args.workers} workers")

    with ProcessPoolExecutor(
        args.workers, initializer=_init_worker, initargs=(tokenizer, essay_path, raw_dir, args.sentence_splitter)
    ) as pool:
        futures = [pool.submit(_build, *job) for job in sorted(jobs, key=lambda j: -j[1])]
        for future in as_completed(futures):
            info = future.result()
            key = f"{info['task']}/L{info['max_seq_len']}"
            manifest["sets"][key] = info
            if "skipped" in info:
                print(f"{key:<24} SKIPPED: {info['skipped']}")
            else:
                print(
                    f"{key:<24} n={info['num_samples']} len={info['length_mean']:.0f} (min {info['length_min']}) "
                    f"size={info['haystack_size']} failed={len(info['failed'])} {info['seconds']}s"
                )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
    ).stdout.strip()
    manifest.update(
        tokenizer=tokenizer,
        seed=args.seed,
        sentence_splitter=resources.splitter[0],
        essay_fingerprint=resources.essay_fingerprint,
        git_commit=commit,
    )
    manifest["sets"] = dict(sorted(manifest["sets"].items()))
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
