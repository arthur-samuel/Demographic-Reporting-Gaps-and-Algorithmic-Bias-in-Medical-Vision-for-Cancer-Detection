"""
Track B meta-analysis: difference in AUC (majority minus minority).

Reproduces Table 5 of "Demographic Reporting Gaps and Algorithmic Bias in
Medical Vision for Cancer Detection" (Ameyaw, Agbalekpor, Adu-Afari, Boateng).

Method
  dAUC_i      = AUC_majority - AUC_minority                  (Eq. 2)
  SE(AUC)     = (upper - lower) / 3.92
  SE(dAUC_i)  = sqrt(SE_maj^2 + SE_min^2)                    (Eq. 3)
  Pooling     : DerSimonian-Laird random effects             (Eq. 4)
  I^2         = max(0, (Q - (k-1)) / Q) * 100                (Eq. 5)

The three Daneshjou 2022 models share one image set, so the primary
analysis collapses them to a single study-level estimate (mean dAUC,
mean SE: assumes complete correlation, conservative). A sensitivity
analysis enters all five comparisons separately.

Usage:  python meta_analysis.py [path/to/trackB_pooled_comparisons.csv] [outdir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "trackB_pooled_comparisons.csv"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "results"
Z = 1.959964


def comparison_effects(df):
    df = df.copy()
    df["se_maj"] = (df.auc_majority_hi - df.auc_majority_lo) / 3.92
    df["se_min"] = (df.auc_minority_hi - df.auc_minority_lo) / 3.92
    df["d_auc"] = df.auc_majority - df.auc_minority
    df["se_d"] = np.sqrt(df.se_maj**2 + df.se_min**2)
    df["ci_lo"] = df.d_auc - Z * df.se_d
    df["ci_hi"] = df.d_auc + Z * df.se_d
    return df


def study_level(df):
    """Collapse multi-model studies: mean dAUC and mean SE (complete correlation)."""
    g = df.groupby("study_id", sort=False)
    out = g.agg(modality=("modality", "first"), d_auc=("d_auc", "mean"), se_d=("se_d", "mean")).reset_index()
    out["ci_lo"] = out.d_auc - Z * out.se_d
    out["ci_hi"] = out.d_auc + Z * out.se_d
    return out


def pool(theta, se):
    theta, v = np.asarray(theta, float), np.asarray(se, float) ** 2
    k = len(theta)
    w = 1 / v
    fe = np.sum(w * theta) / np.sum(w)
    se_fe = np.sqrt(1 / np.sum(w))
    res = dict(k=k, fe=fe, fe_lo=fe - Z * se_fe, fe_hi=fe + Z * se_fe)
    if k == 1:
        res.update(re=theta[0], re_lo=theta[0] - Z * np.sqrt(v[0]), re_hi=theta[0] + Z * np.sqrt(v[0]),
                   se_re=np.sqrt(v[0]), Q=np.nan, Q_p=np.nan, I2=np.nan, tau2=np.nan, p=2 * stats.norm.sf(abs(theta[0] / np.sqrt(v[0]))))
        return res
    Q = np.sum(w * (theta - fe) ** 2)
    C = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / C)
    ws = 1 / (v + tau2)
    re = np.sum(ws * theta) / np.sum(ws)
    se_re = np.sqrt(1 / np.sum(ws))
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    res.update(re=re, se_re=se_re, re_lo=re - Z * se_re, re_hi=re + Z * se_re, Q=Q,
               Q_p=stats.chi2.sf(Q, k - 1), I2=I2, tau2=tau2, p=2 * stats.norm.sf(abs(re / se_re)))
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    comp = comparison_effects(pd.read_csv(DATA))
    study = study_level(comp)

    rows = []
    def add(name, r):
        rows.append(dict(analysis=name, k=r["k"], estimate=r["re"], ci_lo=r["re_lo"], ci_hi=r["re_hi"],
                         I2=r["I2"], tau2=r["tau2"], Q=r["Q"], Q_p=r["Q_p"], p=r["p"]))

    prim = pool(study.d_auc, study.se_d)
    add("Primary, study level, RE", prim)
    rows.append(dict(analysis="Primary, study level, FE", k=prim["k"], estimate=prim["fe"],
                     ci_lo=prim["fe_lo"], ci_hi=prim["fe_hi"]))
    add("All five comparisons, RE", pool(comp.d_auc, comp.se_d))

    mammo = study[study.modality == "mammography_risk"]
    derm = study[study.modality == "dermatology"]
    r_m, r_d = pool(mammo.d_auc, mammo.se_d), pool(derm.d_auc, derm.se_d)
    add("Mammography risk models only", r_m)
    add("Dermatology, study level", r_d)
    for sid in study.study_id:
        sub = study[study.study_id != sid]
        r = pool(sub.d_auc, sub.se_d)
        add(f"Leave out {sid}, RE", r)

    # test for subgroup difference (mammography vs dermatology)
    diff = r_d["re"] - r_m["re"]
    se_diff = np.sqrt(r_d["se_re"] ** 2 + r_m["se_re"] ** 2)
    z = diff / se_diff
    p = 2 * stats.norm.sf(abs(z))

    table = pd.DataFrame(rows)
    comp.to_csv(OUT / "comparison_level_effects.csv", index=False)
    study.to_csv(OUT / "study_level_effects.csv", index=False)
    table.to_csv(OUT / "table5_meta_analysis.csv", index=False)

    pd.set_option("display.width", 160)
    print("Comparison-level effects")
    print(comp[["comparison_label", "d_auc", "se_d", "ci_lo", "ci_hi"]].round(3).to_string(index=False))
    print("\nTable 5")
    print(table.round(4).to_string(index=False))
    print(f"\nSubgroup difference (dermatology - mammography): {diff:.3f} AUC, z = {z:.2f}, p = {p:.3f}")
    with open(OUT / "subgroup_difference.txt", "w") as f:
        f.write(f"dermatology minus mammography: {diff:.4f} (SE {se_diff:.4f}), z = {z:.3f}, p = {p:.4f}\n")


if __name__ == "__main__":
    main()
