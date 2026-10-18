"""Build fixed positional-NLL eval sets from the FineWeb-Edu heldout split.

For each length L, picks up to N heldout docs with at least L model-tokenizer tokens (deterministic order by
doc-id hash), truncates them to exactly L tokens (eos is not appended) and writes ``<out_dir>/L<L>.parquet``
with columns ``id`` and ``input_ids``.

    python scripts/longctx/make_nll_evalset.py --heldout ~/datasets/fineweb-edu-longctx/heldout \
        --tokenizer ~/models/Qwen3-0.6B-Base --out_dir ~/datasets/fineweb-edu-longctx/nll_eval_qwen3
"""

import argparse
import glob
import hashlib
import json
import os

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--num_docs", type=int, nargs="+", default=[256, 256, 256, 256, 128])
    parser.add_argument(
        "--min_gpt2_ratio", type=float, default=0.7, help="Prefilter docs by GPT-2 token_count >= ratio * min(L)."
    )
    args = parser.parse_args()
    assert len(args.lengths) == len(args.num_docs)

    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(args.tokenizer))
    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.heldout), "*.parquet")))
    table = pa.concat_tables(pq.read_table(f, columns=["id", "text", "token_count"]) for f in files)
    table = table.filter(pc.greater_equal(table.column("token_count"), int(args.min_gpt2_ratio * min(args.lengths))))
    ids, texts = table.column("id").to_pylist(), table.column("text").to_pylist()
    order = sorted(range(len(ids)), key=lambda i: hashlib.md5(ids[i].encode()).hexdigest())
    print(f"{len(ids)} candidate docs; tokenizing")
    token_ids = tokenizer(texts, add_special_tokens=False)["input_ids"]

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    summary = {"args": vars(args), "tokenizer": tokenizer.name_or_path, "sets": {}}
    for length, num_docs in zip(args.lengths, args.num_docs):
        chosen = [i for i in order if len(token_ids[i]) >= length][:num_docs]
        pq.write_table(
            pa.table(
                {
                    "id": [ids[i] for i in chosen],
                    "input_ids": pa.array([token_ids[i][:length] for i in chosen], type=pa.list_(pa.int32())),
                }
            ),
            os.path.join(out_dir, f"L{length}.parquet"),
        )
        available = sum(len(t) >= length for t in token_ids)
        summary["sets"][length] = {"docs": len(chosen), "available": available}
        print(f"L={length}: {len(chosen)} docs (available {available})")

    with open(os.path.join(out_dir, "evalset_meta.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
