"""Compare long-context eval results across checkpoints.

    python scripts/longctx/plot_longctx.py --results a/results.json b/results.json --labels A B \
        --train_len 4096 --out longctx.png
"""

import argparse
import json
from collections import defaultdict

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pick_nll_set(lengths, max_pos):
    """Shortest NLL eval set covering ``max_pos`` positions (longest set when there is no limit)."""
    if not lengths:
        return None
    covering = [n for n in lengths if max_pos and n >= max_pos]
    return min(covering) if covering else max(lengths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--train_len", type=int, default=None, help="Draw a marker at the training length.")
    parser.add_argument(
        "--nll_max_pos",
        type=int,
        default=None,
        help="Plot NLL only for positions below this (default: --train_len; 0 = no limit).",
    )
    parser.add_argument("--placement", default="uniform")
    parser.add_argument("--noise_size", type=int, default=1)
    parser.add_argument(
        "--metric", default="token_acc", choices=["token_acc", "first_item_acc", "exact", "answer_nll"]
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    labels = args.labels or args.results
    results = [json.load(open(p)) for p in args.results]

    n_copies = sorted({c["n_copy"] for r in results for c in r.get("selective_copy", {}).get("cells", [])})
    nll_max_pos = args.train_len if args.nll_max_pos is None else args.nll_max_pos
    nll_len = pick_nll_set({int(k) for r in results for k in r.get("position_nll", {})}, nll_max_pos)
    num_panels = len(n_copies) + (nll_len is not None)
    fig, axes = plt.subplots(1, num_panels, figsize=(4.2 * num_panels, 3.6), squeeze=False)
    axes = axes[0]

    for ax, n_copy in zip(axes, n_copies):
        for res, label in zip(results, labels):
            curve = defaultdict(list)
            for c in res.get("selective_copy", {}).get("cells", []):
                if (
                    c["n_copy"] == n_copy
                    and c["placement"] == args.placement
                    and c.get("noise_size", 1) == args.noise_size
                ):
                    curve[c["ctx_len"]].append(c[args.metric])
            xs = sorted(curve)
            ax.plot(xs, [sum(curve[x]) / len(curve[x]) for x in xs], marker="o", label=label)
        ax.set(
            xscale="log",
            xlabel="body length L",
            ylabel=args.metric,
            title=f"selective copy, n_copy={n_copy}, noise k={args.noise_size}",
        )
        if args.train_len:
            ax.axvline(args.train_len, color="gray", linestyle="--", linewidth=1)

    if nll_len is not None:
        ax = axes[-1]
        for res, label in zip(results, labels):
            bins = res.get("position_nll", {}).get(str(nll_len), {}).get("log2_bins", [])
            bins = [b for b in bins if b["start"] >= 16 and (not nll_max_pos or b["end"] <= nll_max_pos)]
            ax.errorbar(
                [b["start"] for b in bins],
                [b["nll"] for b in bins],
                yerr=[b["ci95"] for b in bins],
                marker="o",
                capsize=2,
                label=label,
            )
        ax.set(
            xscale="log",
            xlabel="token position (log2 bin start)",
            ylabel="NLL",
            title=f"FineWeb-Edu NLL vs position (L={nll_len})",
        )
        if args.train_len and not (nll_max_pos and nll_max_pos <= args.train_len):
            ax.axvline(args.train_len, color="gray", linestyle="--", linewidth=1)

    # One shared legend above all panels: every panel plots the same runs in the same colors.
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(legend_labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
