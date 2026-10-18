"""Dump scalar history from an offline ``.wandb`` run file, without syncing it.

The file is a leveldb-style log (32 KiB blocks of crc/len/type-framed chunks) after a 7-byte W&B header, each
record a ``wandb_internal_pb2.Record``.

    python scripts/longctx/read_wandb_offline.py ~/runs/longctx/a0_pilot_1B/wandb/offline-run-*/run-*.wandb \
        --keys loss --every 100
"""

import argparse
import glob
import json
import re
import struct
from typing import Dict, Iterator, List

from wandb.proto import wandb_internal_pb2 as pb


HEADER_SIZE = 7
BLOCK_SIZE = 32768
CHUNK_HEADER = struct.Struct("<IHB")  # crc32, length, type
FULL, FIRST, MIDDLE, LAST = 1, 2, 3, 4


def iter_records(path: str) -> Iterator[bytes]:
    with open(path, "rb") as f:
        data = f.read()
    pos, pending = HEADER_SIZE, b""
    while pos + CHUNK_HEADER.size <= len(data):
        left_in_block = BLOCK_SIZE - pos % BLOCK_SIZE
        if left_in_block < CHUNK_HEADER.size:  # block trailer padding
            pos += left_in_block
            continue
        _, length, kind = CHUNK_HEADER.unpack_from(data, pos)
        pos += CHUNK_HEADER.size
        chunk, pos = data[pos : pos + length], pos + length
        if kind == FULL:
            yield chunk
        elif kind == FIRST:
            pending = chunk
        elif kind == MIDDLE:
            pending += chunk
        elif kind == LAST:
            yield pending + chunk
            pending = b""


def read_history(path: str) -> List[Dict[str, float]]:
    rows = []
    for raw in iter_records(path):
        record = pb.Record()
        record.ParseFromString(raw)
        if record.WhichOneof("record_type") == "history":
            rows.append({i.key or "/".join(i.nested_key): json.loads(i.value_json) for i in record.history.item})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path or glob of the run-*.wandb file.")
    parser.add_argument("--keys", default=".", help="Regex selecting metric keys.")
    parser.add_argument("--every", type=int, default=1, help="Print every N-th step (plus the last).")
    args = parser.parse_args()

    rows = read_history(glob.glob(args.path)[0])
    keys = sorted({k for r in rows for k in r if re.search(args.keys, k) and not k.startswith("_")})
    print("step\t" + "\t".join(keys))
    for i, row in enumerate(rows):
        step = row.get("_step", i)
        if step % args.every == 0 or i == len(rows) - 1:
            print(f"{step}\t" + "\t".join(f"{row[k]:.4g}" if k in row else "" for k in keys))


if __name__ == "__main__":
    main()
