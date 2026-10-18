"""Split FineWeb-Edu into train / heldout parquet shards by a stable hash of the document id.

Long documents are scarce (only ~3.4k docs >= 32k tokens in sample-10BT), so they are held out at a higher
rate than short ones: ``--long_pct`` percent of docs with ``token_count >= --long_threshold`` and
``--base_pct`` percent of the rest. The heldout split feeds the positional-NLL eval sets.

    python scripts/longctx/split_fineweb.py --src ~/datasets/fineweb-edu/sample/10BT \
        --dst ~/datasets/fineweb-edu-longctx
"""

import argparse
import glob
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


KEEP_COLUMNS = ["text", "id", "token_count"]


def hash_bucket(doc_id: str, mod: int = 100) -> int:
    return int.from_bytes(hashlib.md5(doc_id.encode()).digest()[:8], "little") % mod


def heldout_mask(table: pa.Table, base_pct: int, long_pct: int, long_threshold: int) -> np.ndarray:
    buckets = np.fromiter((hash_bucket(i) for i in table.column("id").to_pylist()), dtype=np.int64)
    is_long = pc.greater_equal(table.column("token_count"), long_threshold).to_numpy(zero_copy_only=False)
    return buckets < np.where(is_long, long_pct, base_pct)


def split_file(path: str, dst: str, base_pct: int, long_pct: int, long_threshold: int) -> dict:
    table = pq.read_table(path, columns=KEEP_COLUMNS)
    mask = heldout_mask(table, base_pct, long_pct, long_threshold)
    name = os.path.basename(path)
    stats = {"file": name}
    for split, split_mask in (("train", ~mask), ("heldout", mask)):
        part = table.filter(pa.array(split_mask))
        pq.write_table(part, os.path.join(dst, split, name), compression="zstd", row_group_size=8192)
        stats[split] = {"docs": part.num_rows, "gpt2_tokens": int(pc.sum(part.column("token_count")).as_py() or 0)}
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--base_pct", type=int, default=1)
    parser.add_argument("--long_pct", type=int, default=25)
    parser.add_argument("--long_threshold", type=int, default=16384, help="In the dataset's GPT-2 token_count.")
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.src), "*.parquet")))
    dst = os.path.expanduser(args.dst)
    for split in ("train", "heldout"):
        os.makedirs(os.path.join(dst, split), exist_ok=True)

    with ProcessPoolExecutor(args.workers) as pool:
        stats = list(
            pool.map(
                split_file,
                files,
                [dst] * len(files),
                [args.base_pct] * len(files),
                [args.long_pct] * len(files),
                [args.long_threshold] * len(files),
            )
        )

    summary = {
        "args": vars(args),
        "files": stats,
        "total": {
            split: {k: sum(s[split][k] for s in stats) for k in ("docs", "gpt2_tokens")}
            for split in ("train", "heldout")
        },
    }
    with open(os.path.join(dst, "split_meta.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["total"], indent=2))


if __name__ == "__main__":
    main()
