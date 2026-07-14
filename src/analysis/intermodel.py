"""
intermodel.py — secondary: inter-model agreement.
Report-level score per judge = rounded mean of repeats -> pairwise quadratic weighted kappa + % agreement.
Target: primary outcome (uses C1).
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from _common import load_cfg, load_scores, primary_item


def run(cfg=None):
    cfg = cfg or load_cfg()
    df = load_scores(cfg)
    item = primary_item(cfg)
    c1 = df[df.condition == "C1"]
    piv = c1.groupby(["report_id", "judge_id"])[item].mean().round().unstack("judge_id").dropna()
    judges = list(piv.columns)
    rows = []
    for a, b in itertools.combinations(judges, 2):
        try:
            wk = cohen_kappa_score(piv[a], piv[b], weights="quadratic")
        except Exception:
            wk = np.nan
        agree = float((piv[a] == piv[b]).mean())
        rows.append({"judge_a": a, "judge_b": b,
                     "weighted_kappa": round(float(wk), 3),
                     "pct_agreement": round(100 * agree, 1),
                     "n_reports": len(piv)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = run()
    print(res.to_string(index=False))
    res.to_csv("outputs/tables/intermodel_agreement.csv", index=False)
    print("\n[saved] outputs/tables/intermodel_agreement.csv")
