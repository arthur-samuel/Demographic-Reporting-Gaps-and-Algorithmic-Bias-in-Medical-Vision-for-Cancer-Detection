"""
Regenerates the data figures (Figures 3-6) of the manuscript from the CSVs in ../data
and the meta-analysis outputs. Run meta_analysis.py first.

Figures 1 and 2 are schematic (study design, evidence base) and are not data driven.

Usage:  python make_figures.py [datadir] [resultsdir] [figdir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data"
RES = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "results"
FIG = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MAROON, GOLD, BLUE, GREEN, RED2 = "#500000", "#B8860B", "#2E5A87", "#2E7D5B", "#8B2A2A"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure3():
    a = pd.read_csv(DATA / "trackA_audits.csv").dropna(subset=["figure3_label"])
    order = ["academic", "fda692", "fda1012", "ce", "derm", "datasets"]
    colors = {"academic": GOLD, "fda692": MAROON, "fda1012": RED2, "ce": GOLD, "derm": BLUE, "datasets": GREEN}
    a["o"] = a.figure3_group.map({g: i for i, g in enumerate(order)})
    a = a.sort_values("o", kind="stable").iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y = np.arange(len(a))
    ax.barh(y, a.value_pct, color=[colors[g] for g in a.figure3_group])
    for yi, (v, g) in zip(y, zip(a.value_pct, a.figure3_group)):
        ax.text(v + 0.5, yi, "0% to 25% by year" if g == "academic" else f"{v:.1f}%", va="center", fontsize=6.5)
    ax.set_yticks(y, a.figure3_label, fontsize=6.5)
    ax.set_xlim(0, 55)
    ax.set_xlabel("Percentage of devices, studies or images")
    ax.grid(axis="x", alpha=0.3)
    save(fig, "figure3_trackA_reporting")


def figure4():
    d = pd.read_csv(DATA / "nguyen2024_false_positive_or.csv").iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for i, r in d.iterrows():
        c = MAROON if r.odds_ratio > 1 else BLUE
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=c)
        ax.plot(r.odds_ratio, i, "s", color=c)
        ax.text(7.5, i, f"{r.odds_ratio} ({r.ci_lo} to {r.ci_hi})", va="center", fontsize=6.5)
    ax.axvline(1, ls="--", color="grey", lw=0.8)
    split = (d.score == "Risk score").sum() - 0.5
    ax.axhline(split, color="lightgrey", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
    ax.set_xlim(0.4, 7)
    ax.set_yticks(range(len(d)), [f"{s}: {c}" for s, c in zip(d.score, d.comparison)], fontsize=6.5)
    ax.set_xlabel("Odds ratio of a false positive AI output (log scale)")
    save(fig, "figure4_nguyen_false_positive_or")


def figure5():
    comp = pd.read_csv(RES / "comparison_level_effects.csv")
    tab = pd.read_csv(RES / "table5_meta_analysis.csv").set_index("analysis")
    rows = [(r.comparison_label, r.d_auc, r.ci_lo, r.ci_hi, False) for r in comp.itertuples()]
    for name, lab in [("Primary, study level, RE", "Pooled, study level (k = 3), random effects"),
                      ("All five comparisons, RE", "All comparisons (k = 5), random effects")]:
        t = tab.loc[name]
        rows.append((lab, t.estimate, t.ci_lo, t.ci_hi, True))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    n = len(rows)
    for i, (lab, e, lo, hi, pooled) in enumerate(rows):
        y = n - 1 - i
        c = MAROON if pooled else BLUE
        ax.plot([lo, hi], [y, y], color=c)
        ax.plot(e, y, "D" if pooled else "s", color=c, ms=6 if pooled else 4)
        ax.text(0.33, y, f"{e:+.3f} ({lo:+.3f} to {hi:+.3f})", va="center", fontsize=6.5)
    ax.axhline(1.5, color="lightgrey", lw=0.8)
    ax.axvline(0, ls="--", color="grey", lw=0.8)
    ax.set_yticks(range(n), [r[0] for r in rows][::-1], fontsize=6.5)
    ax.set_xlim(-0.2, 0.32)
    ax.set_xlabel("Difference in AUC, majority subgroup minus minority subgroup")
    p = tab.loc["Primary, study level, RE"]
    ax.set_title(f"Study level pool: {p.estimate:+.3f} (95% CI {p.ci_lo:.3f} to {p.ci_hi:+.3f}); "
                 f"I² = {p.I2:.0f}%; τ² = {p.tau2:.4f}", fontsize=7)
    save(fig, "figure5_forest_auc_difference")


def figure6():
    m = pd.read_csv(DATA / "mechanism_evidence.csv")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [1.3, 1]})
    A = m[m.panel == "A"].reset_index(drop=True)
    for i, r in A.iterrows():
        y = len(A) - 1 - i
        if r.auc_lo == r.auc_hi:
            axA.plot(r.auc_lo, y, "o", color=MAROON if r.setting == "image model" else "grey")
        else:
            axA.plot([r.auc_lo, r.auc_hi], [y, y], color=MAROON, lw=4, solid_capstyle="round")
    axA.axvline(0.5, ls="--", color="grey", lw=0.8)
    axA.axhline(2.5, color="lightgrey", lw=0.8)
    axA.set_yticks(range(len(A)), A.label[::-1], fontsize=6.5)
    axA.set_xlim(0.45, 1.0)
    axA.set_xlabel("AUC for self reported race")
    axA.set_title("A. Race is recoverable from images\n(Gichoya et al., 2022)", fontsize=7)

    B = m[m.panel == "B"]
    labels = list(dict.fromkeys(B.label))
    x = np.arange(len(labels))
    std = [B[(B.label == l) & (B.setting.str.startswith("standard"))].auc_lo.iloc[0] for l in labels]
    site = [B[(B.label == l) & (B.setting.str.startswith("preserved"))].auc_lo.iloc[0] for l in labels]
    w = 0.38
    axB.bar(x - w / 2, std, w, color=MAROON, label="Standard cross validation")
    axB.bar(x + w / 2, site, w, color=GOLD, label="Preserved site cross validation")
    for xi, v in zip(np.r_[x - w / 2, x + w / 2], std + site):
        axB.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=6)
    axB.axhline(0.5, ls="--", color="grey", lw=0.8)
    axB.set_ylim(0.4, 1.05)
    axB.set_xticks(x, [l.replace(" (", "\n(") for l in labels], fontsize=6)
    axB.set_ylabel("AUROC for genetic ancestry")
    axB.legend(fontsize=6, loc="upper left", frameon=False)
    axB.set_title("B. Ancestry signal rides on site signatures\n(Howard et al., 2021)", fontsize=7)
    fig.tight_layout()
    save(fig, "figure6_mechanism_evidence")


if __name__ == "__main__":
    figure3(); figure4(); figure5(); figure6()
    print("Figures written to", FIG)
