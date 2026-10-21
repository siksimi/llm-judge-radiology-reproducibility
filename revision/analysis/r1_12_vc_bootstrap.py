"""r1_12_vc_bootstrap.py - report-level bootstrap CIs for the condition-specific variance-component
percentages of R1-12 (same models as reanalysis_phase2_mixed.py). Added at R1-20 (figure uncertainty),
after the point estimates had been reported; descriptive.
Resampling: reports with replacement; each drawn copy becomes its own cluster. Percentile 95% CIs.
Usage: python3 r1_12_vc_bootstrap.py <scores.parquet> <out.csv> <B> <seed> [job_index]
(each job uses its own stream: seed + job_index)
"""
import sys, warnings, time
import numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
src, out, B, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
df = pd.read_parquet(src); y = "overall_global_quality"
LAB = {"claude-opus-4-8": "Claude Opus 4.8", "gpt-5.5": "GPT-5.5", "gemini-2.5-pro": "Gemini 2.5 Pro", "qwen-3.5": "Qwen 3.5"}


def three(d):
    m = smf.mixedlm(f"{y} ~ C(cond)", d, groups="rid", re_formula="1", vc_formula={"rxc": "0 + C(cond)"})
    r = m.fit(reml=True, method=["lbfgs", "powell"])
    a, b, c = float(r.cov_re.iloc[0, 0]), float(r.vcomp[0]), float(r.scale)
    return a, b, c, bool(r.converged)


def two(d):
    cc = d.groupby(["rid", "cond"])[y].first().reset_index()
    r = smf.mixedlm(f"{y} ~ C(cond)", cc, groups="rid").fit(reml=True)
    return float(r.cov_re.iloc[0, 0]), float(r.scale), 0.0, bool(r.converged)


jobs = [("gemini-2.5-pro", "C2", "temperature", "C2_temperature"), ("qwen-3.5", "C2", "temperature", "C2_temperature")] + \
       [(j, "C3", "variant", "C3_prompt") for j in LAB]
rows = []
sel = [int(sys.argv[5])] if len(sys.argv) > 5 else range(len(jobs))
for ji in sel:
    j, cond, col, lab = jobs[ji]; rng = np.random.default_rng(seed + ji)
    d0 = df[(df.judge_id == j) & (df.condition == cond)].dropna(subset=[y]).copy()
    d0["cond"] = d0[col].astype(str); d0["rid"] = d0.report_id
    det = d0.groupby(["report_id", "cond"])[y].std().max() == 0
    f = two if det else three
    est = f(d0); tot = sum(est[:3]); pe = [100 * v / tot for v in est[:3]]
    groups = {k: g for k, g in d0.groupby("report_id")}; ids = np.array(list(groups))
    bs = []; nconv = 0; t0 = time.time()
    for b in range(B):
        draw = rng.choice(ids, size=len(ids), replace=True)
        parts = []
        for k, rid in enumerate(draw):
            g = groups[rid].copy(); g["rid"] = f"{rid}#{k}"; parts.append(g)
        db = pd.concat(parts, ignore_index=True)
        try:
            v = f(db); s = sum(v[:3]); bs.append([100 * x / s for x in v[:3]]); nconv += v[3]
        except Exception:
            pass
    bs = np.array(bs)
    for i, name in enumerate(["report", "report_x_condition", "residual_run"]):
        lo, hi = (np.percentile(bs[:, i], [2.5, 97.5]) if len(bs) else (np.nan, np.nan))
        rows.append(dict(judge=LAB[j], condition=lab, component=name, pct=pe[i], lo=lo, hi=hi,
                         B_ok=len(bs), B_converged=int(nconv), deterministic_2comp=det))
    print(LAB[j], lab, f"{time.time()-t0:.0f}s", [round(x, 1) for x in pe], flush=True)
pd.DataFrame(rows).to_csv(out, index=False)
print("seed", seed, "B", B)
