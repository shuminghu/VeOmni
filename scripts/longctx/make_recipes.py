"""Generate selective-copy recipe parquets (train stream or frozen test grid).

A recipe row is a few ints/strings; ``veomni.data.synthetic`` renders it into tokens at load time.

Train (random lengths up to the training sequence length, seeds 0..N-1; n_copy log-uniform up to 512)::

    python scripts/longctx/make_recipes.py train --tokenizer ~/models/Qwen3-0.6B-Base \
        --max_seq_len 4096 --num_samples 4000000 --out_dir ~/datasets/synthetic/selective_copy/train_T4096_cap512

Test (fixed grid, layout pinned explicitly in the rows; seeds from 10**9)::

    python scripts/longctx/make_recipes.py test --tokenizer ~/models/Qwen3-0.6B-Base \
        --out_dir ~/datasets/synthetic/selective_copy/test
"""

import argparse
import itertools
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer

from veomni.data.synthetic import build_lexicon, get_synthetic_task
from veomni.data.synthetic.selective_copy import SelectiveCopyRecipe


TASK = "selective_copy"
TEST_SEED_BASE = 10**9


def _write_shards(rows: dict, out_dir: str, num_shards: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    table = pa.table(rows)
    bounds = np.linspace(0, table.num_rows, num_shards + 1).astype(int)
    for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        pq.write_table(table.slice(lo, hi - lo), os.path.join(out_dir, f"{i:05d}.parquet"))


def _write_meta(out_dir: str, args: argparse.Namespace, tokenizer, extra: dict) -> None:
    lexicon = build_lexicon(tokenizer, content_size=args.content_size, noise_size=args.noise_size)
    meta = {"args": vars(args), "tokenizer": tokenizer.name_or_path, "lexicon": lexicon.fingerprint, **extra}
    # Not .json: the dataset loader would pick a json file up as data next to the parquet shards.
    with open(os.path.join(out_dir, "recipe_meta.yaml"), "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


def _min_ctx_len(n_copy, args):
    """Shortest body that keeps the content fraction ``n_copy / ctx_len`` at most ``max_content_frac``."""
    return np.maximum(args.min_ctx_len, np.ceil(np.asarray(n_copy) / args.max_content_frac).astype(np.int64))


def _draw_n_copy(rng: np.random.Generator, dist: str, max_n_copy: int, size: int) -> np.ndarray:
    """``uniform``: U{1..max}. ``loguniform``: floor(exp(U[0, log(max + 1)))), i.e. every octave of n_copy gets
    the same share of samples, so small n stays well covered while the range extends to large n."""
    if dist == "uniform":
        return rng.integers(1, max_n_copy + 1, size=size)
    if dist == "loguniform":
        return (
            np.floor(np.exp(rng.uniform(0.0, np.log(max_n_copy + 1), size=size))).astype(np.int64).clip(1, max_n_copy)
        )
    raise ValueError(f"Unknown n_copy distribution {dist!r}.")


def make_train(args, tokenizer) -> None:
    """n_copy ~ ``n_copy_dist`` over {1..max_n_copy};
    ctx_len ~ log-uniform [max(min_ctx_len, n_copy / max_content_frac), T - overhead(n_copy)]."""
    task = get_synthetic_task(TASK, tokenizer)
    rng = np.random.default_rng(args.seed)
    n = args.num_samples

    n_copy = _draw_n_copy(rng, args.n_copy_dist, args.max_n_copy, n)
    lo = _min_ctx_len(n_copy, args)
    overhead = np.array([task.max_overhead(int(k), args.fmt, "train") for k in range(args.max_n_copy + 1)])
    hi = args.max_seq_len - overhead[n_copy]
    if (hi < lo).any():
        raise ValueError("max_seq_len too small for the requested n_copy / min_ctx_len.")
    ctx_len = np.floor(np.exp(rng.uniform(np.log(lo), np.log(hi + 1)))).astype(np.int64).clip(lo, hi)

    rows = {
        "task": [TASK] * n,
        "seed": np.arange(args.seed_base, args.seed_base + n, dtype=np.int64),
        "ctx_len": ctx_len,
        "n_copy": n_copy.astype(np.int64),
        "placement": ["uniform"] * n,
        "template_split": ["train"] * n,
        "template_id": np.full(n, -1, dtype=np.int64),
        "fmt": [args.fmt] * n,
        "content_size": np.full(n, args.content_size, dtype=np.int64),
        "noise_size": np.full(n, args.noise_size, dtype=np.int64),
        "unique_content": np.full(n, args.unique_content, dtype=bool),
    }
    _write_shards(rows, args.out_dir, args.num_shards)
    mean_len = float(np.mean(ctx_len + overhead[n_copy]))
    mean_n_copy = float(np.mean(n_copy))
    _write_meta(
        args.out_dir,
        args,
        tokenizer,
        {
            "num_rows": n,
            "mean_tokens_per_sample": mean_len,
            "mean_n_copy": mean_n_copy,
            "approx_total_tokens": mean_len * n,
        },
    )
    print(
        f"wrote {n} train recipes, ~{mean_len:.0f} tokens/sample, mean n_copy {mean_n_copy:.1f}, "
        f"~{mean_len * n / 1e9:.2f}B tokens total"
    )


def make_test(args, tokenizer) -> None:
    """Full grid; each row pins ``body_positions`` / ``content`` so the set never depends on the RNG version."""
    task = get_synthetic_task(TASK, tokenizer)
    columns = {
        k: []
        for k in (
            "task",
            "seed",
            "ctx_len",
            "n_copy",
            "placement",
            "template_split",
            "template_id",
            "fmt",
            "content_size",
            "noise_size",
            "unique_content",
            "body_positions",
            "content",
            "cell",
        )
    }
    # Seeds are consecutive over cells, outer axes first: appending a value to an outer axis (noise size, template
    # split) adds cells at the end and leaves the seeds of existing cells unchanged.
    # Cells denser than training allows (n_copy / ctx_len > max_content_frac) are skipped.
    grid = list(
        itertools.product(args.noise_sizes, args.template_splits, args.ctx_lens, args.n_copies, args.placements)
    )
    cells = [c for c in grid if c[2] >= _min_ctx_len(c[3], args)]
    skipped = sorted({(c[2], c[3]) for c in grid if c not in cells})
    if skipped:
        print(f"skipping (ctx_len, n_copy) cells denser than max_content_frac={args.max_content_frac}: {skipped}")
    seed = TEST_SEED_BASE
    for noise_size, template_split, ctx_len, n_copy, placement in cells:
        cell = f"L{ctx_len}_n{n_copy}_{placement}_{template_split}" + (f"_k{noise_size}" if noise_size > 1 else "")
        for _ in range(args.num_per_cell):
            recipe = SelectiveCopyRecipe(
                seed=seed,
                ctx_len=ctx_len,
                n_copy=n_copy,
                placement=placement,
                template_split=template_split,
                fmt=args.fmt,
                content_size=args.content_size,
                noise_size=noise_size,
                unique_content=args.unique_content,
            )
            template_id, positions, content = task.sample_layout(recipe, np.random.default_rng(seed))
            for key, value in (
                ("task", TASK),
                ("seed", seed),
                ("ctx_len", ctx_len),
                ("n_copy", n_copy),
                ("placement", placement),
                ("template_split", template_split),
                ("template_id", template_id),
                ("fmt", args.fmt),
                ("content_size", args.content_size),
                ("noise_size", noise_size),
                ("unique_content", args.unique_content),
                ("body_positions", positions.tolist()),
                ("content", content.tolist()),
                ("cell", cell),
            ):
                columns[key].append(value)
            seed += 1
    _write_shards(columns, args.out_dir, 1)
    _write_meta(args.out_dir, args, tokenizer, {"num_rows": len(columns["seed"]), "num_cells": len(cells)})
    print(f"wrote {len(columns['seed'])} test recipes in {len(cells)} cells")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["train", "test"])
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fmt", default="nl2", choices=["nl", "nl2", "raw"], help="Template set (harness.py).")
    parser.add_argument("--content_size", type=int, default=2048)
    parser.add_argument("--noise_size", type=int, default=1, help="Train only; test uses --noise_sizes.")
    parser.add_argument(
        "--unique_content",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the n_copy content words without replacement (pre-capacity-axis recipes: --no-unique_content).",
    )
    parser.add_argument(
        "--max_content_frac",
        type=float,
        default=0.25,
        help="Max n_copy / ctx_len: lower bound on train ctx_len, and test cells above it are skipped.",
    )
    # train
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--num_samples", type=int, default=4_000_000)
    parser.add_argument("--min_ctx_len", type=int, default=64)
    parser.add_argument("--max_n_copy", type=int, default=512)
    parser.add_argument(
        "--n_copy_dist",
        default="loguniform",
        choices=["uniform", "loguniform"],
        help="Pre-capacity-axis recipes: --n_copy_dist uniform --max_n_copy 64.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for drawing lengths.")
    parser.add_argument("--seed_base", type=int, default=0, help="First per-sample seed.")
    parser.add_argument("--num_shards", type=int, default=64)
    # test
    parser.add_argument("--ctx_lens", type=int, nargs="+", default=[2**k for k in range(8, 16)])
    parser.add_argument("--n_copies", type=int, nargs="+", default=[8, 32, 64, 128, 256, 512])
    parser.add_argument("--placements", nargs="+", default=["uniform", "early"])
    # "train" (templates seen in training) is the primary metric; "test" (held-out templates) checks prompt transfer.
    parser.add_argument("--template_splits", nargs="+", default=["test", "train"])
    parser.add_argument("--num_per_cell", type=int, default=128)
    parser.add_argument(
        "--noise_sizes", type=int, nargs="+", default=[1], help="1: filler is ' .'; k>1: k filler words."
    )
    args = parser.parse_args()

    args.out_dir = os.path.expanduser(args.out_dir)
    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(args.tokenizer))
    (make_train if args.split == "train" else make_test)(args, tokenizer)


if __name__ == "__main__":
    main()
