"""Compare long-context eval results across checkpoints.

    python scripts/longctx/plot_longctx.py --results a/results.json b/results.json --labels A B \
        --train_len 4096 --out longctx.png
"""

import argparse
import json
import math
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


def _short(n: int) -> str:
    return f"{n // 1024}k" if n >= 1024 and n % 1024 == 0 else str(n)


def _log_ticks(ax, ticks, metric):
    """Explicit major ticks at the tested values (matplotlib's minor log labels overlap on short ranges)."""
    ax.set_xticks(ticks, [_short(t) for t in ticks])
    ax.minorticks_off()
    if metric != "answer_nll":
        ax.set_ylim(-0.03, 1.03)


def sc_curve(result, args, x_key, **fixed):
    """Mean SC ``args.metric`` vs ``x_key`` over cells matching ``fixed`` and the placement/noise/template filters."""
    fixed = {
        "placement": args.placement,
        "noise_size": args.noise_size,
        "template_split": args.template_split,
        **fixed,
    }
    curve = defaultdict(list)
    for c in result.get("selective_copy", {}).get("cells", []):
        if all(c.get(k, 1 if k == "noise_size" else None) == v for k, v in fixed.items()):
            curve[c[x_key]].append(c[args.metric])
    xs = sorted(curve)
    return xs, [sum(curve[x]) / len(curve[x]) for x in xs]


def min_ctx_len(n_copy, cells):
    return min((c["ctx_len"] for c in cells if c["n_copy"] == n_copy), default=0)


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
        "--template_split", default="train", help="train: primary metric; test: held-out prompt templates."
    )
    parser.add_argument(
        "--metric", default="token_acc", choices=["token_acc", "first_item_acc", "exact", "answer_nll"]
    )
    parser.add_argument("--n_copies", type=int, nargs="+", default=None, help="Per-n_copy panels (default: all).")
    parser.add_argument(
        "--capacity_len",
        type=int,
        default=None,
        help="Body length of the metric-vs-n_copy panel (default: largest tested L <= --train_len).",
    )
    parser.add_argument("--ncols", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    labels = args.labels or args.results
    results = [json.load(open(p)) for p in args.results]

    cells = [c for r in results for c in r.get("selective_copy", {}).get("cells", [])]
    all_n_copies = sorted({c["n_copy"] for c in cells})
    n_copies = args.n_copies or all_n_copies
    ctx_lens = sorted({c["ctx_len"] for c in cells})
    capacity_len = args.capacity_len or max(
        (L for L in ctx_lens if not args.train_len or L <= args.train_len), default=None
    )
    show_capacity = len(all_n_copies) > 1 and capacity_len is not None
    nll_max_pos = args.train_len if args.nll_max_pos is None else args.nll_max_pos
    nll_len = pick_nll_set({int(k) for r in results for k in r.get("position_nll", {})}, nll_max_pos)
    num_panels = len(n_copies) + show_capacity + (nll_len is not None)
    ncols = min(args.ncols, num_panels)
    nrows = math.ceil(num_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
    axes = list(axes.flat)
    for ax in axes[num_panels:]:
        ax.set_visible(False)
    panels = iter(axes)

    for n_copy in n_copies:
        ax = next(panels)
        for res, label in zip(results, labels):
            ax.plot(*sc_curve(res, args, "ctx_len", n_copy=n_copy), marker="o", label=label)
        ax.set(
            xscale="log",
            xlabel="body length L",
            ylabel=args.metric,
            title=f"SC n_copy={n_copy}, k={args.noise_size}, {args.template_split} templates",
        )
        _log_ticks(ax, [L for L in ctx_lens if L >= min_ctx_len(n_copy, cells)], args.metric)
        if args.train_len:
            ax.axvline(args.train_len, color="gray", linestyle="--", linewidth=1)

    if show_capacity:
        # Capacity axis: how many items fit, at a fixed (in-distribution) distance.
        ax = next(panels)
        for res, label in zip(results, labels):
            ax.plot(*sc_curve(res, args, "n_copy", ctx_len=capacity_len), marker="o", label=label)
        ax.set(
            xscale="log",
            xlabel="n_copy",
            ylabel=args.metric,
            title=f"SC capacity, L={capacity_len}, {args.template_split} templates",
        )
        _log_ticks(ax, all_n_copies, args.metric)

    if nll_len is not None:
        ax = next(panels)
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
