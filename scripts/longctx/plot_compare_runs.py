"""Side-by-side figures for several long-context runs (eval_longctx results + training curves).

Each run is ``--run LABEL RUN_DIR COLOR LINESTYLE MARKER``; ``RUN_DIR`` holds ``eval/step<N>/results.json`` and a
``wandb/*run-*/run-*.wandb`` file (online or offline). Encode one factor with color and another with line style /
marker fill, e.g. architecture x precision:

    python scripts/longctx/plot_compare_runs.py --out_dir fineweb_1B_syn_select_copy_attn_kda_tern_plots \
        --run "Qwen3-S bf16" ~/runs/longctx/qat_pair_1B/bf16 "#2a78d6" solid o \
        --run "Qwen3-S ternary" ~/runs/longctx/qat_pair_1B/ternary "#2a78d6" dashed o \
        --run "Hybrid bf16" ~/runs/longctx/hybrid_pair_1B/bf16 "#eb6834" solid s \
        --run "Hybrid ternary" ~/runs/longctx/hybrid_pair_1B/ternary "#eb6834" dashed s

Selective-copy panels use one template split and placement (default: train templates, uniform placement).
LINESTYLE is a matplotlib name (``solid``, ``dashed``, ``dotted``); non-solid styles draw hollow markers so precision is readable without color.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_wandb_offline import read_history  # noqa: E402


INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


@dataclass
class Run:
    label: str
    path: str
    color: str
    linestyle: str
    marker: str

    def style(self) -> dict:
        hollow = self.linestyle not in ("-", "solid")
        return dict(
            color=self.color,
            linestyle=self.linestyle,
            marker=self.marker,
            markerfacecolor=SURFACE if hollow else self.color,
            markeredgecolor=self.color,
            markeredgewidth=1.5,
            label=self.label,
        )

    def results(self, step: int) -> Optional[dict]:
        p = os.path.join(self.path, "eval", f"step{step}", "results.json")
        return json.load(open(p)) if os.path.exists(p) else None

    def history(self) -> List[Dict[str, float]]:
        files = sorted(glob.glob(os.path.join(self.path, "wandb", "*run-*", "run-*.wandb")))
        return read_history(files[-1]) if files else []


def short(n: int) -> str:
    return f"{n // 1024}k" if n >= 1024 and n % 1024 == 0 else str(n)


def log2_axis(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([short(t) for t in ticks])
    ax.minorticks_off()


def mark_train_len(ax, train_len):
    if train_len:
        ax.axvline(train_len, color=INK_2, linewidth=1, linestyle=":", zorder=0)


def sc_cells(result, split, placement):
    return [
        c
        for c in result["selective_copy"]["cells"]
        if c["template_split"] == split and c.get("placement", placement) == placement
    ]


def legend_below(fig, runs):
    handles = [plt.Line2D([], [], **r.style()) for r in runs]
    fig.legend(handles=handles, loc="lower center", ncol=len(runs), bbox_to_anchor=(0.5, -0.04))


# ---------------------------------------------------------------- figures


def fig_nll(runs, step, train_len, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    longest = None
    for r in runs:
        res = r.results(step)
        if res is None:
            continue
        nll = res["position_nll"]
        longest = max(nll, key=int)
        bins = nll[longest]["log2_bins"]
        x = [bn["start"] for bn in bins]
        y = np.array([bn["nll"] for bn in bins])
        ci = np.array([bn.get("ci95", 0.0) for bn in bins])
        a.plot(x, y, **r.style())
        a.fill_between(x, y - ci, y + ci, color=r.color, alpha=0.08, linewidth=0)
        lengths = sorted(nll, key=int)
        b.plot([int(L) for L in lengths], [nll[L]["mean_nll"] for L in lengths], **r.style())
    a.set_title(f"NLL by position (L={short(int(longest))} set, log2 bins, 95% CI)")
    a.set_xlabel("token position")
    a.set_ylabel("NLL (nats)")
    log2_axis(a, [2**k for k in range(0, 17, 2)])
    mark_train_len(a, train_len)
    b.set_title("Mean NLL over the whole sequence")
    b.set_xlabel("eval length L")
    lengths = sorted({int(L) for r in runs if r.results(step) for L in r.results(step)["position_nll"]})
    log2_axis(b, lengths)
    mark_train_len(b, train_len)
    legend_below(fig, runs)
    fig.suptitle(f"Held-out FineWeb-Edu NLL, step {step} (dotted: training length)", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def fig_sc_by_len(runs, step, train_len, split, placement, metric, n_copies, out):
    ncols = 3
    nrows = (len(n_copies) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows), sharey=True, squeeze=False)
    all_lens = set()
    for ax, n in zip(axes.flat, n_copies):
        for r in runs:
            res = r.results(step)
            if res is None:
                continue
            cells = sorted(
                (c for c in sc_cells(res, split, placement) if c["n_copy"] == n), key=lambda c: c["ctx_len"]
            )
            all_lens.update(c["ctx_len"] for c in cells)
            ax.plot([c["ctx_len"] for c in cells], [c[metric] for c in cells], **r.style())
        ax.set_title(f"n_copy = {n}")
        ax.set_ylim(-0.03, 1.03)
        mark_train_len(ax, train_len)
    for ax in axes.flat[len(n_copies) :]:
        ax.set_visible(False)
    for ax in axes.flat[: len(n_copies)]:
        log2_axis(ax, sorted(all_lens))
        ax.tick_params(axis="x", labelrotation=45)
    for ax in axes[:, 0]:
        ax.set_ylabel(metric.replace("_", " "))
    for ax in axes[-1]:
        ax.set_xlabel("context length L")
    legend_below(fig, runs)
    fig.suptitle(
        f"Selective copy {metric.replace('_', ' ')} vs length, step {step} ({split} templates, {placement})",
        color=INK,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def fig_sc_capacity(runs, step, train_len, split, placement, lengths, out):
    fig, axes = plt.subplots(1, len(lengths), figsize=(3.0 * len(lengths), 3.4), sharey=True)
    for ax, L in zip(np.atleast_1d(axes), lengths):
        for r in runs:
            res = r.results(step)
            if res is None:
                continue
            cells = sorted(
                (c for c in sc_cells(res, split, placement) if c["ctx_len"] == L), key=lambda c: c["n_copy"]
            )
            ax.plot([c["n_copy"] for c in cells], [c["exact"] for c in cells], **r.style())
        ax.set_title(f"L = {short(L)}" + (" (> T)" if train_len and L > train_len else ""))
        log2_axis(ax, [8, 32, 128, 512])
        ax.set_xlabel("n_copy (words to copy)")
        ax.set_ylim(-0.03, 1.03)
    np.atleast_1d(axes)[0].set_ylabel("exact match")
    legend_below(fig, runs)
    fig.suptitle(f"Copy capacity, step {step} ({split} templates, {placement})", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def fig_templates(runs, step, train_len, placement, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, split in zip(axes, ("train", "test")):
        lens = set()
        for r in runs:
            res = r.results(step)
            if res is None:
                continue
            by_len = defaultdict(list)
            for c in sc_cells(res, split, placement):
                by_len[c["ctx_len"]].append(c["exact"])
            xs = sorted(by_len)
            lens.update(xs)
            ax.plot(xs, [np.mean(by_len[x]) for x in xs], **r.style())
        ax.set_title(f"{'training' if split == 'train' else 'held-out'} templates")
        log2_axis(ax, sorted(lens))
        ax.set_xlabel("context length L")
        ax.set_ylim(-0.03, 1.03)
        mark_train_len(ax, train_len)
    axes[0].set_ylabel("exact match (mean over n_copy)")
    legend_below(fig, runs)
    fig.suptitle(f"Selective copy: prompt templates seen vs unseen in training, step {step}", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def fig_learning(runs, steps, train_len, split, placement, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for r in runs:
        rows = [(s, r.results(s)) for s in steps]
        rows = [(s, res) for s, res in rows if res is not None]
        xs = [s for s, _ in rows]
        nll = [res["position_nll"][str(train_len)]["mean_nll"] for _, res in rows]

        def sc(res, inside):
            v = [c["exact"] for c in sc_cells(res, split, placement) if (c["ctx_len"] <= train_len) == inside]
            return float(np.mean(v)) if v else float("nan")

        axes[0].plot(xs, nll, **r.style())
        axes[1].plot(xs, [sc(res, True) for _, res in rows], **r.style())
        axes[2].plot(xs, [sc(res, False) for _, res in rows], **r.style())
    axes[0].set_title(f"FineWeb NLL, L = {short(train_len)}")
    axes[0].set_ylabel("NLL (nats)")
    axes[1].set_title(f"SC exact, L ≤ T ({short(train_len)})")
    axes[2].set_title("SC exact, L > T (8k–32k)")
    for ax in axes:
        ax.set_xlabel("training step")
        ax.set_xticks(steps)
    for ax in axes[1:]:
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("exact match (mean over cells)")
    legend_below(fig, runs)
    fig.suptitle(f"Learning curves over eval checkpoints ({split} templates, {placement})", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def smooth(y, window):
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def fig_train_loss(runs, out, window=10):
    channels = [("fineweb_edu", "FineWeb-Edu channel loss"), ("selective_copy", "Selective-copy channel loss")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for r in runs:
        hist = r.history()
        for ax, (chan, _) in zip(axes, channels):
            key = next((k for row in hist for k in row if k.startswith("channel_loss/") and k.endswith(chan)), None)
            pts = [(row["_step"], row[key]) for row in hist if key and key in row]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ys = smooth(ys, window)
            ax.plot(xs[len(xs) - len(ys) :], ys, **{**r.style(), "marker": None})
    for ax, (_, title) in zip(axes, channels):
        ax.set_title(title + f" (moving avg of {window} logs)")
        ax.set_xlabel("training step")
    # FineWeb spans ~3.2-6 nats (linear reads best); selective copy falls 8 -> 0.003 (needs log).
    axes[0].set_ylim(top=6)
    axes[0].set_ylabel("loss (nats)")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("loss (nats, log scale)")
    legend_below(fig, runs)
    fig.suptitle("Training loss by data channel", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def throughput_table(runs, skip_steps=50) -> str:
    lines = ["| run | tokens/s per GPU (median) | logged MFU (median) |", "|---|---|---|"]
    for r in runs:
        hist = [row for row in r.history() if row.get("_step", 0) > skip_steps]
        tps = [row["tokens_per_second(M)"] * 1e6 / 8 for row in hist if "tokens_per_second(M)" in row]
        mfu = [row["mfu"] for row in hist if row.get("mfu")]
        lines.append(
            f"| {r.label} | {np.median(tps):,.0f} | " + (f"{np.median(mfu):.3f}" if mfu else "– (not logged)") + " |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", nargs=5, action="append", required=True, metavar=("LABEL", "DIR", "COLOR", "LS", "MARKER"))
    p.add_argument("--out_dir", required=True)
    p.add_argument("--step", type=int, default=1908)
    p.add_argument("--steps", type=int, nargs="+", default=[500, 1000, 1500, 1908])
    p.add_argument("--train_len", type=int, default=4096)
    p.add_argument("--split", default="train")
    p.add_argument("--placement", default="uniform")
    p.add_argument("--n_copies", type=int, nargs="+", default=[8, 32, 64, 128, 256, 512])
    p.add_argument("--capacity_lens", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    args = p.parse_args()

    runs = [Run(label, os.path.expanduser(d), c, ls, m) for label, d, c, ls, m in args.run]
    os.makedirs(args.out_dir, exist_ok=True)
    o = lambda name: os.path.join(args.out_dir, name)  # noqa: E731
    fig_nll(runs, args.step, args.train_len, o("nll.png"))
    for metric in ("exact", "first_item_acc"):
        fig_sc_by_len(
            runs,
            args.step,
            args.train_len,
            args.split,
            args.placement,
            metric,
            args.n_copies,
            o(f"sc_{metric}_by_len.png"),
        )
    fig_sc_capacity(
        runs, args.step, args.train_len, args.split, args.placement, args.capacity_lens, o("sc_capacity.png")
    )
    fig_templates(runs, args.step, args.train_len, args.placement, o("sc_templates.png"))
    fig_learning(runs, args.steps, args.train_len, args.split, args.placement, o("learning_curves.png"))
    fig_train_loss(runs, o("train_loss.png"))
    print(throughput_table(runs))


if __name__ == "__main__":
    main()
