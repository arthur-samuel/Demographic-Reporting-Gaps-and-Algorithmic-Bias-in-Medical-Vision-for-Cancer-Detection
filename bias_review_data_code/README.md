# Data and code: Demographic Reporting Gaps and Algorithmic Bias in Medical Vision for Cancer Detection

Ameyaw SA, Agbalekpor D, Adu-Afari D, Boateng J. Search date 24 September 2026.

Every value below was taken from a figure printed in the cited publications, or computed directly from such a figure.

## Layout

```
data/
  trackB_pooled_comparisons.csv   5 comparisons entering the meta-analysis (subgroup AUCs + 95% CIs)
  trackB_all_studies.csv          Table 4: all 10 subgroup-performance studies and how each was used
  trackA_audits.csv               Table 3 / Figure 3: published reporting audits (FDA, CE, academic, datasets)
  nguyen2024_false_positive_or.csv Figure 4: false-positive odds ratios (Nguyen et al. 2024)
  mechanism_evidence.csv          Figure 6: race-from-image AUCs (Gichoya 2022), ancestry AUROCs (Howard 2021)
scripts/
  meta_analysis.py                DerSimonian-Laird pooling -> Table 5, subgroup test
  make_figures.py                 Figures 3-6 (PNG 300 dpi + PDF)
results/                          outputs of meta_analysis.py
figures/                          outputs of make_figures.py
```

## Run

```bash
pip install numpy scipy pandas matplotlib
python scripts/meta_analysis.py
python scripts/make_figures.py
```

## Methods implemented (manuscript section 2.6)

- dAUC = AUC_majority - AUC_minority (positive = worse in minority group)
- SE(AUC) = (upper - lower) / 3.92; SE(dAUC) = sqrt(SE_maj^2 + SE_min^2) (independent patients)
- DerSimonian-Laird random effects; I^2 per Higgins & Thompson (2002)
- The three Daneshjou 2022 models share one image set. The primary analysis collapses them to one study-level estimate (mean dAUC, mean SE; complete correlation, conservative). The sensitivity analysis enters all five separately.
- Modality subgroup difference: z-test on the two subgroup random-effects estimates.

## Reproduced results (match Table 5)

| Analysis | k | Estimate (95% CI) | I^2 | tau^2 |
|---|---|---|---|---|
| Primary, study level, RE | 3 | 0.020 (-0.071 to 0.111) | 45% | 0.0030 |
| Primary, study level, FE | 3 | 0.008 (-0.053 to 0.069) | | |
| All five comparisons, RE | 5 | 0.057 (-0.018 to 0.133) | 50% | 0.0036 |
| Mammography risk only | 2 | -0.024 (-0.093 to 0.046) | 0% | 0 |
| Dermatology, study level | 1 | 0.117 (-0.012 to 0.246) | | |
| Leave out Yala 2019 | 2 | 0.034 (-0.109 to 0.177) | 73% | 0.0078 |
| Leave out Gastounioti 2022 | 2 | 0.066 (-0.047 to 0.180) | 23% | 0.0016 |
| Leave out Daneshjou 2022 | 2 | -0.024 (-0.093 to 0.046) | 0% | 0 |

Subgroup difference (dermatology - mammography) = 0.141, z = 1.88, p = 0.06. Primary RE p = 0.66; Q = 3.66 (df 2, p = 0.16).

## Notes for verification before submission

- Miyawaki 2025 reports a 0-25% range by year. Figure 3 draws the bar at the 25% upper bound.
- Mehta 2025 training-data-source value (6.7%) = 100 - 93.3% not reported.
- The "Hand radiograph (DHA)" range in mechanism_evidence.csv (0.87-0.90) was read off Figure 6 and is not stated in the manuscript text. Check it against Gichoya et al. 2022 before submission.
- The Yala 2019 minority CI (0.57-0.87) is asymmetric. The symmetric SE approximation is noted as a limitation (section 4.4).
- Figures 1 and 2 are schematics and are not generated from data.
