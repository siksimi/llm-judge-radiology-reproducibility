"""
reliability.py — primary outcome: intra-judge reproducibility (C1).
  - ICC(2,1)=ICC(A,1), ICC(2,k)=ICC(A,k)  (two-way random, absolute agreement) + 95% CI
  - quadratic weighted kappa (mean over run pairs)
  - CV (mean of per-report coefficients of variation)
pingouin 0.6 labels: ICC(A,1)/ICC(A,k); CI column = 'CI95'.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.metrics import cohen_kappa_score

from _common import load_cfg, load_scores, primary_item, ITEMS

ICC_MAP = {"ICC(A,1)": "ICC2_1", "ICC(A,k)": "ICC2_k"}  # absolute-agreement, two-way random


def _icc(df_long, item):
    d = df_long[["report_id", "run", item]].dropna()
    if d["run"].nunique() < 2 or d["report_id"].nunique() < 2:
        return None
    try:
        res = pg.intraclass_corr(data=d, targets="report_id", raters="run",
                                 ratings=item, nan_policy="omit").set_index("Type")
    except Exception:
        return None
    out = {}
    for typ, key in ICC_MAP.items():
        if typ in res.index:
            row = res.loc[typ]
            ci = row["CI95"]
            out[key] = round(float(row["ICC"]), 3)
            out[key + "_CI"] = [round(float(ci[0]), 3), round(float(ci[1]), 3)]
    return out


def _weighted_kappa_runs(df_long, item):
    wide = df_long.pivot_table(index="report_id", columns="run", values=item).dropna()
    runs = list(wide.columns)
    if len(runs) < 2 or len(wide) < 2:
        return None
    ks = []
    for a, b in itertools.combinations(runs, 2):
        try:
            ks.append(cohen_kappa_score(wide[a], wide[b], weights="quadratic"))
        except Exception:
            pass
    return round(float(np.nanmean(ks)), 3) if ks else None


def _cv(df_long, item):
    g = df_long.groupby("report_id")[item]
    cv = g.std(ddof=1) / g.mean().replace(0, np.nan)
    cv = cv.replace([np.inf, -np.inf], np.nan).dropna()
    return round(float(cv.mean()), 3) if len(cv) else None


def run(cfg=None):
    cfg = cfg or load_cfg()
    df = load_scores(cfg)
    c1 = df[df["condition"] == "C1"].copy()
    prim = primary_item(cfg)
    rows = []
    for judge, g in c1.groupby("judge_id"):
        for item in ITEMS:
            icc = _icc(g, item)
            if icc is None:
                continue
            rows.append({"judge_id": judge, "item": item, "is_primary": item == prim,
                         **icc, "weighted_kappa": _weighted_kappa_runs(g, item),
                         "CV": _cv(g, item),
                         "n_reports": g["report_id"].nunique(), "n_runs": g["run"].nunique()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = run()
    cols = ["judge_id", "item", "is_primary", "ICC2_1", "ICC2_1_CI",
            "ICC2_k", "weighted_kappa", "CV", "n_reports", "n_runs"]
    cols = [c for c in cols if c in res.columns]
    print(res[cols].to_string(index=False))
    res.to_csv("outputs/tables/reliability.csv", index=False)
    print("\n[saved] outputs/tables/reliability.csv")
