"""Long-context eval for an HF checkpoint: selective-copy accuracy grid + positional NLL on FineWeb-Edu.

    torchrun --nproc_per_node 8 scripts/longctx/eval_longctx.py --model_path <hf_ckpt> \
        --sc_test_dir ~/datasets/synthetic/selective_copy/test \
        --nll_dir ~/datasets/fineweb-edu-longctx/nll_eval_qwen3 \
        --out_dir <hf_ckpt>/longctx_eval

Writes ``results.json`` (aggregates) and ``sc_samples.jsonl`` (per-sample selective-copy records).

Logits are never materialized for the whole sequence (64k x 151k vocab would not fit): the eval runs the
backbone once and applies the LM head only where needed, in chunks.
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer

from veomni.data.synthetic import get_synthetic_task
from veomni.eval.utils import add_qat_args, gather, init_dist, load_causal_lm, ternary_from_args, token_budget_batches


# ---------------------------------------------------------------- model helpers


class CausalLMScorer:
    """Backbone forward + chunked LM head, so long sequences never materialize full logits."""

    def __init__(
        self,
        model_path: str,
        attn_implementation: str,
        device: torch.device,
        head_tokens: int = 16384,
        ternary: Optional[dict] = None,
    ):
        self.model = load_causal_lm(model_path, device, attn_implementation, ternary)
        self.backbone = self.model.base_model
        self.lm_head = self.model.get_output_embeddings()
        self.softcap = getattr(self.model.config, "final_logit_softcapping", None)
        self.device = device
        self.head_tokens = head_tokens  # tokens per LM-head chunk (bounds the fp32 logits buffer)

    @torch.no_grad()
    def hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state

    @torch.no_grad()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden).float()
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits

    @torch.no_grad()
    def token_nll(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """NLL of ``input_ids[:, 1:]`` given the prefix; shape ``[B, L-1]``."""
        batch_size, seq_len = input_ids.shape
        chunk = max(1, self.head_tokens // batch_size)
        out = []
        for start in range(0, seq_len - 1, chunk):
            end = min(start + chunk, seq_len - 1)
            logits = self.logits(hidden[:, start:end])
            targets = input_ids[:, start + 1 : end + 1]
            out.append(F.cross_entropy(logits.transpose(1, 2), targets, reduction="none"))
        return torch.cat(out, dim=1)


def pad_batch(seqs: Sequence[Sequence[int]], pad_id: int, device: torch.device):
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = torch.as_tensor(s)
        attention_mask[i, : len(s)] = 1
    return input_ids.to(device), attention_mask.to(device)


# ---------------------------------------------------------------- helpers


def log2_bin(x: int) -> int:
    return int(math.floor(math.log2(max(x, 1))))


# ---------------------------------------------------------------- selective copy


def eval_selective_copy(scorer, tokenizer, test_dir, rank, world_size, max_tokens, max_ctx_len, limit_per_cell):
    task = get_synthetic_task("selective_copy", tokenizer)
    rows = pq.read_table(sorted(glob.glob(os.path.join(test_dir, "*.parquet")))).to_pylist()
    rows = [r for r in rows if max_ctx_len is None or r["ctx_len"] <= max_ctx_len]
    if limit_per_cell:
        per_cell = defaultdict(int)
        kept = []
        for r in rows:
            if per_cell[r["cell"]] < limit_per_cell:
                per_cell[r["cell"]] += 1
                kept.append(r)
        rows = kept
    rows = rows[rank::world_size]
    samples = [task.build(r) for r in rows]

    records = []
    for batch in token_budget_batches([len(s.input_ids) for s in samples], max_tokens):
        input_ids, attention_mask = pad_batch(
            [samples[i].input_ids for i in batch], tokenizer.eos_token_id, scorer.device
        )
        hidden = scorer.hidden_states(input_ids, attention_mask)
        for row_idx, i in enumerate(batch):
            sample, label_pos = samples[i], samples[i].label_positions
            logits = scorer.logits(hidden[row_idx, [p - 1 for p in label_pos]])
            targets = torch.as_tensor([sample.labels[p] for p in label_pos], device=logits.device)
            result = task.score(sample, logits.argmax(-1).tolist())
            answer_nll = F.cross_entropy(logits[:-1], targets[:-1]).item()  # content items only
            records.append(
                {
                    "cell": rows[i]["cell"],
                    **{k: sample.meta[k] for k in ("ctx_len", "n_copy", "placement", "template_split", "template_id")},
                    "noise_size": int(rows[i].get("noise_size") or 1),
                    "seq_len": len(sample.input_ids),
                    "answer_nll": answer_nll,
                    **result,
                }
            )
    return records


def aggregate_selective_copy(records: list) -> dict:
    cells = defaultdict(list)
    for r in records:
        cells[r["cell"]].append(r)
    cell_summary = []

    def cell_order(kv):
        first = kv[1][0]
        return first["noise_size"], first["ctx_len"], first["n_copy"], kv[0]

    for cell, rs in sorted(cells.items(), key=cell_order):
        first = rs[0]
        cell_summary.append(
            {
                "cell": cell,
                **{k: first[k] for k in ("ctx_len", "n_copy", "placement", "template_split", "noise_size")},
                "n": len(rs),
                "token_acc": float(np.mean([r["token_acc"] for r in rs])),
                # Teacher forcing lets items k>0 be found by induction from item k-1; item 0 is pure retrieval.
                "first_item_acc": float(np.mean([r["item_correct"][0] for r in rs])),
                "exact": float(np.mean([r["exact"] for r in rs])),
                "eos_acc": float(np.mean([r["eos_correct"] for r in rs])),
                "answer_nll": float(np.mean([r["answer_nll"] for r in rs])),
            }
        )

    # Item accuracy vs distance to the query (log2 bins), per (ctx_len, n_copy), uniform placement only.
    by_distance = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["placement"] != "uniform":
            continue
        key = f"L{r['ctx_len']}_n{r['n_copy']}" + (f"_k{r['noise_size']}" if r["noise_size"] > 1 else "")
        for correct, distance in zip(r["item_correct"], r["distances"]):
            by_distance[key][log2_bin(distance)].append(correct)
    distance_summary = {
        key: {str(2**b): {"acc": float(np.mean(v)), "n": len(v)} for b, v in sorted(bins.items())}
        for key, bins in by_distance.items()
    }
    return {"cells": cell_summary, "by_distance": distance_summary}


# ---------------------------------------------------------------- positional NLL


def eval_position_nll(scorer, nll_dir, rank, world_size, max_tokens, max_ctx_len, linear_bin):
    results = {}
    for path in sorted(glob.glob(os.path.join(nll_dir, "L*.parquet")), key=lambda p: int(os.path.basename(p)[1:-8])):
        seq_len = int(os.path.basename(path)[1:-8])
        if max_ctx_len is not None and seq_len > max_ctx_len:
            continue
        docs = pq.read_table(path).column("input_ids").to_pylist()
        local = list(range(rank, len(docs), world_size))
        batch_size = max(1, max_tokens // seq_len)
        per_doc = []  # [num_docs, seq_len - 1] float32 on cpu
        for start in range(0, len(local), batch_size):
            batch = [docs[i] for i in local[start : start + batch_size]]
            input_ids = torch.as_tensor(batch, dtype=torch.long, device=scorer.device)
            hidden = scorer.hidden_states(input_ids, torch.ones_like(input_ids))
            per_doc.extend(scorer.token_nll(hidden, input_ids).float().cpu().numpy())
        per_doc = np.stack(gather(per_doc, world_size))  # target position t = index + 1

        positions = np.arange(1, seq_len)
        log_bins = np.floor(np.log2(positions)).astype(int)
        log_summary = []
        for b in range(log_bins.max() + 1):
            doc_means = per_doc[:, log_bins == b].mean(1)
            log_summary.append(
                {
                    "start": 2**b,
                    "end": min(2 ** (b + 1), seq_len),
                    "nll": float(doc_means.mean()),
                    "ci95": float(1.96 * doc_means.std(ddof=1) / math.sqrt(len(doc_means))),
                }
            )
        num_linear = (seq_len - 1) // linear_bin
        linear = per_doc[:, : num_linear * linear_bin].reshape(len(per_doc), num_linear, linear_bin).mean(2).mean(0)
        results[str(seq_len)] = {
            "num_docs": len(per_doc),
            "mean_nll": float(per_doc.mean()),
            "log2_bins": log_summary,
            "linear_bin": linear_bin,
            "linear_nll": linear.tolist(),
        }
    return results


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--sc_test_dir", default=None)
    parser.add_argument("--nll_dir", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--max_tokens", type=int, default=65536, help="Token budget per forward batch.")
    parser.add_argument("--max_ctx_len", type=int, default=None, help="Skip SC cells / NLL sets longer than this.")
    parser.add_argument("--limit_per_cell", type=int, default=None)
    parser.add_argument("--linear_bin", type=int, default=256)
    add_qat_args(parser)
    args = parser.parse_args()

    rank, world_size, device = init_dist()
    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(args.tokenizer_path or args.model_path))
    scorer = CausalLMScorer(
        os.path.expanduser(args.model_path), args.attn_implementation, device, ternary=ternary_from_args(args)
    )
    results: Dict[str, object] = {"model_path": args.model_path, "args": vars(args)}

    if args.sc_test_dir:
        records = eval_selective_copy(
            scorer,
            tokenizer,
            os.path.expanduser(args.sc_test_dir),
            rank,
            world_size,
            args.max_tokens,
            args.max_ctx_len,
            args.limit_per_cell,
        )
        records = gather(records, world_size)
        results["selective_copy"] = aggregate_selective_copy(records)
    if args.nll_dir:
        results["position_nll"] = eval_position_nll(
            scorer,
            os.path.expanduser(args.nll_dir),
            rank,
            world_size,
            args.max_tokens,
            args.max_ctx_len,
            args.linear_bin,
        )

    if rank == 0:
        out_dir = os.path.expanduser(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        if args.sc_test_dir:
            with open(os.path.join(out_dir, "sc_samples.jsonl"), "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
        for cell in results.get("selective_copy", {}).get("cells", []):
            print(
                f"{cell['cell']:<36} acc={cell['token_acc']:.3f} first={cell['first_item_acc']:.3f} "
                f"exact={cell['exact']:.3f} n={cell['n']}"
            )
        for seq_len, r in results.get("position_nll", {}).items():
            curve = " ".join(f"{b['start']}:{b['nll']:.2f}" for b in r["log2_bins"])
            print(f"NLL L={seq_len} ({r['num_docs']} docs): {curve}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
