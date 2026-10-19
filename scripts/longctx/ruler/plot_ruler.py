"""Compare RULER results across checkpoints: score vs length per task, the average, and NIAH vs needle distance.

    python scripts/longctx/ruler/plot_ruler.py --results a/results.json b/results.json --labels A B \
        --train_len 4096 --out ruler.png
"""

import argparse
import json
import math

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Categorical slots in fixed order (one per model; never cycled).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5bd6", "#b25c2a"]
INK, MUTED, GRID = "#2b2b29", "#73726c", "#e6e5e0"


def _short(n: int) -> str:
    return f"{n // 1024}k" if n >= 1024 and n % 1024 == 0 else str(n)


def _style(ax, title, xticks, train_len):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xticks, [_short(t) for t in xticks], fontsize=7, color=MUTED)
    ax.minorticks_off()
    ax.set_ylim(-3, 103)
    ax.tick_params(axis="y", labelsize=7, colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if train_len:
        ax.axvline(train_len, color=MUTED, linestyle=":", linewidth=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--train_len", type=int, default=None, help="Dotted marker at the training length.")
    parser.add_argument("--ncols", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if len(args.results) > len(SERIES_COLORS):
        raise ValueError(f"at most {len(SERIES_COLORS)} result files per figure; use small multiples")
    results = [json.load(open(p)) for p in args.results]
    labels = args.labels or [p.split("/")[-2] for p in args.results]

    tasks = list(dict.fromkeys(t for r in results for t in r["scores"]))
    panels = ["average"] + tasks + ["niah_by_distance"]
    nrows = math.ceil(len(panels) / args.ncols)
    fig, axes = plt.subplots(nrows, args.ncols, figsize=(2.9 * args.ncols, 2.4 * nrows), squeeze=False)
    xticks = sorted({int(L) for r in results for row in r["scores"].values() for L in row})

    for ax, panel in zip(axes.flat, panels):
        for color, label, r in zip(SERIES_COLORS, labels, results):
            if panel == "niah_by_distance":
                # Pool all lengths: hit rate of each needle vs its distance from the end of the prompt.
                pooled = {}
                for bins in r.get("niah_by_distance", {}).values():
                    for b, v in bins.items():
                        hits, n = pooled.get(int(b), (0.0, 0))
                        pooled[int(b)] = (hits + v["acc"] * v["n"], n + v["n"])
                xs = sorted(b for b in pooled if b >= 64)
                ys = [pooled[b][0] / pooled[b][1] for b in xs]
            else:
                row = r["average"] if panel == "average" else r["scores"].get(panel, {})
                xs = sorted(int(L) for L, v in row.items() if v is not None)
                ys = [row[str(L)] for L in xs]
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4, label=label)
        if panel == "niah_by_distance":
            dist_ticks = [2**k for k in range(6, 17, 2)]
            _style(ax, "NIAH hit rate vs needle distance", dist_ticks, args.train_len)
            ax.set_xlabel("tokens from needle to question", fontsize=7, color=MUTED)
        else:
            _style(ax, "Average (all tasks)" if panel == "average" else panel, xticks, args.train_len)
    for ax in axes.flat[len(panels) :]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("score (%)", fontsize=8, color=MUTED)

    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=min(len(labels), 6), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / nrows))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
