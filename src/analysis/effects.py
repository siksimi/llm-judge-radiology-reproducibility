"""
effects.py — secondary: temperature/prompt effects + ranking stability + drift.
  - temperature: mean score per T and within-report run SD per T (C2)
  - prompt: mean per variant, report-ranking stability (Kendall's W, mean Spearman) (C3)
  - drift: per-variant Bland-Altman vs v1 (mean difference, 95% LoA)
Target: primary outcome.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _common import load_cfg, load_scores, primary_item


def kendall_w(matrix: np.ndarray):
    """matrix: rows=items(reports), cols=raters(variants). returns W."""
    m = matrix.shape[1]; n = matrix.shape[0]
    if m < 2 or n < 2:
        return None
    ranks = np.apply_along_axis(lambda c: pd.Series(c).rank().values, 0, matrix)
    Ri = ranks.sum(axis=1)
    S = np.sum((Ri - Ri.mean()) ** 2)
    denom = m ** 2 * (n ** 3 - n)
    return round(float(12 * S / denom), 3) if denom else None


def temperature_effects(df, item):
    c2 = df[df.condition == "C2"]
    rows = []
    for (judge, T), g in c2.groupby(["judge_id", "temperature"]):
        within_sd = g.groupby("report_id")[item].std().mean()
        rows.append({"judge_id": judge, "temperature": T,
                     "mean_score": round(g[item].mean(), 3),
                     "within_report_run_sd": round(float(within_sd), 3)})
    return pd.DataFrame(rows)


def prompt_effects(df, item):
    c3 = df[df.condition == "C3"]
    rows = []
    for judge, g in c3.groupby("judge_id"):
        means = g.groupby("variant")[item].mean()
        # report x variant means -> ranking stability
        piv = g.groupby(["report_id", "variant"])[item].mean().unstack("variant").dropna()
        W = kendall_w(piv.values) if len(piv) > 1 else None
        rhos = []
        for a, b in itertools.combinations(piv.columns, 2):
            if len(piv) > 1:
                r = spearmanr(piv[a], piv[b]).statistic
                if not np.isnan(r):
                    rhos.append(r)
        rows.append({"judge_id": judge,
                     "variant_mean_spread": round(float(means.max() - means.min()), 3),
                     "kendall_W": W,
                     "mean_pairwise_spearman": round(float(np.mean(rhos)), 3) if rhos else None})
    return pd.DataFrame(rows)


def drift_vs_baseline(df, item, baseline="v1"):
    c3 = df[df.condition == "C3"]
    rows = []
    for judge, g in c3.groupby("judge_id"):
        piv = g.groupby(["report_id", "variant"])[item].mean().unstack("variant")
        if baseline not in piv.columns:
            continue
        for v in piv.columns:
            if v == baseline:
                continue
            diff = (piv[v] - piv[baseline]).dropna()
            if len(diff) < 2:
                continue
            md, sd = diff.mean(), diff.std(ddof=1)
            rows.append({"judge_id": judge, "variant": v,
                         "mean_diff": round(float(md), 3),
                         "loa_low": round(float(md - 1.96 * sd), 3),
                         "loa_high": round(float(md + 1.96 * sd), 3)})
    return pd.DataFrame(rows)


def run(cfg=None):
    cfg = cfg or load_cfg()
    df = load_scores(cfg)
    item = primary_item(cfg)
    return {"temperature": temperature_effects(df, item),
            "prompt": prompt_effects(df, item),
            "drift": drift_vs_baseline(df, item)}


if __name__ == "__main__":
    out = run()
    for name, t in out.items():
        print(f"\n=== {name} ===")
        print(t.to_string(index=False))
        t.to_csv(f"outputs/tables/effects_{name}.csv", index=False)
    print("\n[saved] outputs/tables/effects_*.csv")
