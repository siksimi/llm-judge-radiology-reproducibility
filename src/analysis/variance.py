"""
variance.py — secondary: variance-source decomposition (report / run / temperature / prompt).
Mixed-effects model (crossed random effects) estimating each source's percentage variance contribution.
statsmodels MixedLM + vc_formula. vcomp is scaled by residual -> actual variance = vcomp*scale.
Target: primary outcome.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')  # suppress MixedLM small-sample convergence warnings (irrelevant at full scale)

from _common import load_cfg, load_scores, primary_item


def _components(d: pd.DataFrame, item: str, factors: list):
    d = d.dropna(subset=[item]).rename(columns={item: "score"}).copy()
    d["dummy"] = 1
    factors = [f for f in factors if d[f].nunique() > 1]
    if not factors or len(d) < 10:
        return None
    vc = {f: f"0 + C({f})" for f in factors}
    try:
        mf = smf.mixedlm("score ~ 1", d, groups=d["dummy"], vc_formula=vc).fit(reml=True, method="lbfgs")
    except Exception:
        return None
    scale = float(mf.scale)
    names = list(mf.model.exog_vc.names)
    comps = {names[i]: float(mf.vcomp[i]) * scale for i in range(len(names))}
    comps["residual"] = scale
    total = sum(comps.values()) or np.nan
    out = {}
    for k, v in comps.items():
        out[f"{k}_var"] = round(v, 4)
        out[f"{k}_pct"] = round(100 * v / total, 1)
    return out


def run(cfg=None):
    cfg = cfg or load_cfg()
    df = load_scores(cfg)
    item = primary_item(cfg)
    rows = []
    for judge, g in df.groupby("judge_id"):
        res = _components(g, item, ["report_id", "run", "temperature", "variant"])
        if res:
            rows.append({"judge_id": judge, **res})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = run()
    print(res.to_string(index=False))
    res.to_csv("outputs/tables/variance_components.csv", index=False)
    print("\n[saved] outputs/tables/variance_components.csv")
