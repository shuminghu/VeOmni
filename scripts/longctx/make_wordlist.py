"""Build the frequency-ranked English word list used by synthetic-task lexicons.

Counts standalone lowercase words in a slice of FineWeb-Edu. Output is tokenizer-agnostic; each tokenizer
keeps the words that are single tokens (see ``veomni/data/synthetic/lexicon.py``).

    python scripts/longctx/make_wordlist.py --parquet ~/datasets/fineweb-edu/sample/10BT/000_00000.parquet \
        --out veomni/data/synthetic/assets/english_words.txt
"""

import argparse
import re
from collections import Counter

import pyarrow.parquet as pq

from veomni.data.synthetic.lexicon import CONTRACTION_STEMS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_docs", type=int, default=100_000)
    parser.add_argument("--top_k", type=int, default=8000)
    parser.add_argument("--min_chars", type=int, default=3)
    parser.add_argument("--max_chars", type=int, default=10)
    args = parser.parse_args()

    # Whole lowercase words delimited by whitespace / punctuation (excludes fragments of capitalized or mixed tokens).
    word_re = re.compile(r"(?<![A-Za-z'\-])([a-z]{%d,%d})(?![A-Za-z'\-])" % (args.min_chars, args.max_chars))
    counts = Counter()
    pf = pq.ParquetFile(args.parquet)
    seen = 0
    for batch in pf.iter_batches(batch_size=2048, columns=["text"]):
        for text in batch.column(0).to_pylist():
            counts.update(word_re.findall(text))
        seen += batch.num_rows
        if seen >= args.num_docs:
            break

    for stem in CONTRACTION_STEMS:
        counts.pop(stem, None)
    with open(args.out, "w") as f:
        for word, _ in counts.most_common(args.top_k):
            f.write(word + "\n")
    print(f"{seen} docs, {len(counts)} distinct words, wrote top {args.top_k} to {args.out}")


if __name__ == "__main__":
    main()
