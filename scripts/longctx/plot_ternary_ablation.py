"""Figures for a ternary-QAT placement ablation: per-arm cost / recall summary, ternary-code dynamics, outliers.

Each arm is ``--arm LABEL RUN_DIR RULER_RESULTS TERNARY_SHARE`` (share of the fully ternary model's ternary params,
0..1; 0 for the bf16 reference). ``--bf16`` names the bf16 arm's label (NLL cost is measured against it) and
``--dynamics`` names the arm whose ``qat/*`` history is plotted per projection type. Reads ``eval/step<N>/results.json``
and the local ``wandb/*run-*/run-*.wandb`` history of each run.

    python scripts/longctx/plot_ternary_ablation.py --out_dir <dir> --bf16 "bf16" --dynamics "all ternary" \\
        --arm "bf16" ~/runs/longctx/hybrid_pair_1B/bf16 ~/runs/longctx/ruler/hyb_bf16_s1908/results.json 0 \\
        --arm "all ternary" ~/runs/longctx/ternary_ablation/ref ~/runs/longctx/ruler/tabl_ref_s1908/results.json 1 ...
"""

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_compare_runs import INK, INK_2  # noqa: E402  (shared rcParams / tokens)
from read_wandb_offline import read_history  # noqa: E402


# Categorical slots of the reference palette, in fixed order (one per arm); the bf16 arm is neutral ink.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#52514e"


@dataclass
class Arm:
    label: str
    path: str
    ruler: str
    share: float

    def results(self, step: int = 1908) -> dict:
        return json.load(open(os.path.join(self.path, "eval", f"step{step}", "results.json")))

    def ruler_results(self) -> dict:
        return json.load(open(os.path.expanduser(self.ruler)))

    def history(self) -> List[Dict[str, float]]:
        return read_history(sorted(glob.glob(os.path.join(self.path, "wandb", "*run-*", "run-*.wandb")))[-1])


def sc_exact(result, keep):
    cells = [
        c
        for c in result["selective_copy"]["cells"]
        if c["template_split"] == "train" and c["placement"] == "uniform" and keep(c)
    ]
    return float(np.mean([c["exact"] for c in cells]))


def ruler_avg(result, keep):
    return float(np.mean([v for L, v in result["average"].items() if v is not None and keep(int(L))]))


def summary(arms, bf16_label, train_len=4096):
    base = next(a for a in arms if a.label == bf16_label).results()["position_nll"][str(train_len)]["mean_nll"]
    rows = []
    for a in arms:
        res, ru = a.results(), a.ruler_results()
        rows.append(
            dict(
                label=a.label,
                share=a.share,
                nll=res["position_nll"][str(train_len)]["mean_nll"],
                cost=res["position_nll"][str(train_len)]["mean_nll"] - base,
                sc_in=sc_exact(res, lambda c: c["ctx_len"] <= train_len),
                sc_out=sc_exact(res, lambda c: c["ctx_len"] > train_len),
                ruler_in=ruler_avg(ru, lambda L: L <= train_len),
                ruler_out=ruler_avg(ru, lambda L: L > train_len),
                single1_32k=ru["scores"]["niah_single_1"].get("32768"),
            )
        )
    return rows


def fig_summary(rows, colors, out):
    panels = [
        ("cost", "NLL cost vs bf16 at L = 4k (nats)"),
        ("sc_out", "Selective copy exact, L > T (8k–32k)"),
        ("ruler_in", "RULER 13-task avg, L ≤ T (%)"),
        ("ruler_out", "RULER 13-task avg, L > T (%)"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.8))
    y = np.arange(len(rows))
    for ax, (key, title) in zip(axes, panels):
        vals = [r[key] for r in rows]
        ax.barh(y, vals, color=[colors[r["label"]] for r in rows], height=0.6)
        for yi, v in zip(y, vals):
            ax.text(
                v,
                yi,
                f" {v:.3f}" if key == "cost" else f" {v:.2f}" if key.startswith("sc") else f" {v:.1f}",
                va="center",
                ha="left",
                fontsize=8,
                color=INK_2,
            )
        ax.set_yticks(y)
        ax.set_yticklabels([r["label"] for r in rows] if ax is axes[0] else [], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.3 if max(vals) > 0 else 1)
    fig.suptitle("Ternary placement ablation (Qwen3-S-Hybrid, 1B tokens, step 1908)", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_dynamics(arm, out):
    hist = arm.history()
    flips = sorted({k.split("/", 2)[2] for r in hist for k in r if k.startswith("qat/flip_rate/")})
    kinds = [k for k in flips if k != "all" and not k.startswith("layer_")]
    groups = {
        "KDA q/k/v/o": [
            k
            for k in kinds
            if k.startswith("linear_attn.") and k.split(".")[1] in ("q_proj", "k_proj", "v_proj", "o_proj")
        ],
        "KDA gates (f, g, β)": [
            k for k in kinds if k.startswith("linear_attn.") and k.split(".")[1] in ("f_proj", "g_proj", "b_proj")
        ],
        "attention q/k/v/o": [k for k in kinds if k.startswith("self_attn.")],
        "MLP": [k for k in kinds if k.startswith("mlp.")],
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    steps = [r["_step"] for r in hist if "qat/flip_rate/all" in r]
    for i, (name, members) in enumerate(groups.items()):
        for j, k in enumerate(members):
            pts = [(r["_step"], r[f"qat/flip_rate/{k}"] * 100) for r in hist if f"qat/flip_rate/{k}" in r]
            if pts:
                xs, ys = zip(*pts)
                axes[0].plot(xs, ys, color=SLOTS[i], linewidth=1.2, alpha=0.9, label=name if j == 0 else None)
    pts = [(r["_step"], r["qat/flip_rate/all"] * 100) for r in hist if "qat/flip_rate/all" in r]
    xs, ys = zip(*pts)
    axes[0].plot(xs, ys, color=INK, linewidth=2.2, label="all")
    lr = [(r["_step"], r["training/lr"]) for r in hist if "training/lr" in r]
    lx, ly = zip(*lr)
    scale = max(ys) / max(ly)
    axes[0].plot(lx, [v * scale for v in ly], color=INK_2, linestyle="dotted", linewidth=1.5, label="Muon LR (scaled)")
    axes[0].set_title(f"Flip rate per projection ({arm.label})")
    axes[0].set_ylabel("% of ternary codes changed per step")
    axes[0].set_xlabel("training step")
    axes[0].legend(fontsize=8, ncol=2, loc="upper right")

    for i, (name, members) in enumerate(groups.items()):
        for j, k in enumerate(members):
            pts = [(r["_step"], r[f"qat/zero_frac/{k}"]) for r in hist if f"qat/zero_frac/{k}" in r]
            if pts:
                xs2, ys2 = zip(*pts)
                axes[1].plot(xs2, ys2, color=SLOTS[i], linewidth=1.2, alpha=0.9, label=name if j == 0 else None)
    axes[1].set_title("Zero fraction (TWN threshold 0.7)")
    axes[1].set_xlabel("training step")
    axes[1].legend(fontsize=8)

    final = {
        k: next(r[f"qat/flip_rate/{k}"] for r in reversed(hist) if f"qat/flip_rate/{k}" in r) * 100 for k in kinds
    }
    order = sorted(kinds, key=lambda k: final[k])
    color_of = {k: SLOTS[i] for i, (_, ms) in enumerate(groups.items()) for k in ms}
    axes[2].barh(range(len(order)), [final[k] for k in order], color=[color_of[k] for k in order], height=0.65)
    axes[2].set_yticks(range(len(order)))
    axes[2].set_yticklabels(order, fontsize=8)
    axes[2].set_title(f"Flip rate at the end (step {steps[-1]})")
    axes[2].set_xlabel("% per step")
    axes[2].grid(axis="y", visible=False)
    fig.suptitle("Ternary-code dynamics", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_outliers(arms, colors, out, num_layers=12):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for a in arms:
        hist = a.history()
        last = [r for r in hist if "act_absmax/layer_00" in r][-1]
        xs = list(range(num_layers))
        absmax = [last[f"act_absmax/layer_{i:02d}"] for i in xs]
        rms = [last[f"act_rms/layer_{i:02d}"] for i in xs]
        style = dict(
            color=colors[a.label], linewidth=2 if a.share in (0, 1) else 1.4, marker="o", markersize=4, label=a.label
        )
        axes[0].plot(xs, absmax, **style)
        axes[1].plot(xs, [m / s for m, s in zip(absmax, rms)], **style)
        pts = [(r["_step"], r["act_absmax/layer_06"]) for r in hist if "act_absmax/layer_06" in r]
        axes[2].plot(*zip(*pts), **{**style, "marker": None})
    axes[0].set_title("Max |activation| per layer (step 1900)")
    axes[0].set_yscale("log")
    axes[1].set_title("Max / RMS per layer")
    axes[2].set_title("Max |activation| at layer 6 over training")
    axes[2].set_xlabel("training step")
    for ax in axes[:2]:
        ax.set_xlabel("decoder layer")
        ax.set_xticks(range(0, num_layers, 2))
    axes[0].legend(fontsize=8)
    fig.suptitle("Residual-stream outliers: bf16 vs ternary placements", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", nargs=4, action="append", required=True, metavar=("LABEL", "DIR", "RULER", "SHARE"))
    p.add_argument("--bf16", required=True, help="Label of the bf16 arm (NLL cost reference).")
    p.add_argument("--dynamics", required=True, help="Label of the arm whose qat/* history is plotted.")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    arms = [Arm(label, os.path.expanduser(d), r, float(s)) for label, d, r, s in args.arm]
    colors = {}
    slot = 0
    for a in arms:
        if a.label == args.bf16:
            colors[a.label] = NEUTRAL
        else:
            colors[a.label] = SLOTS[slot]
            slot += 1
    os.makedirs(args.out_dir, exist_ok=True)
    rows = summary(arms, args.bf16)
    fig_summary(rows, colors, os.path.join(args.out_dir, "ternary_ablation_summary.png"))
    fig_dynamics(next(a for a in arms if a.label == args.dynamics), os.path.join(args.out_dir, "ternary_dynamics.png"))
    fig_outliers(arms, colors, os.path.join(args.out_dir, "ternary_outliers.png"))
    print(
        "| arm | ternary share | NLL@4k | cost vs bf16 | cost / share | SC exact ≤T | SC exact >T | RULER ≤T | RULER >T | single_1 @32k |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        per = f"{r['cost'] / r['share']:.3f}" if r["share"] else "–"
        print(
            f"| {r['label']} | {r['share'] * 100:.1f}% | {r['nll']:.3f} | {r['cost']:+.3f} | {per} | {r['sc_in']:.2f} | "
            f"{r['sc_out']:.2f} | {r['ruler_in']:.1f} | {r['ruler_out']:.1f} | {r['single1_32k']:.0f} |"
        )


if __name__ == "__main__":
    main()
