"""make_figures_R1.py - revised Figures 1-5 (R1-20).
Inputs: analysis/out CSVs from the revision analyses (bootstrap CIs, B = 2,000).
Outputs: <out>/Figure{1..5}.pdf (vector) and .tif (600 dpi, LZW).
Usage: python3 make_figures_R1.py <csv_dir> <out_dir> <r1_12_vc_bootstrap.csv>
Figure 4 = variance components with bootstrap CIs (final); Figure4_stacked_unused = earlier stacked-bar version.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
})
INK2 = "#52514e"; GRID = "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"           # categorical slots 1-3 (validated all-pairs)
SEQ = ["#cde2fb", "#3987e5", "#104281"]                  # ordinal blue ramp
JUDGES = ["Claude Opus 4.8", "GPT-5.5", "Gemini 2.5 Pro", "Qwen 3.5"]
MM = 1 / 25.4


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tif", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.5); ax.set_axisbelow(True)


def panel(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def errpts(ax, x, est, lo, hi, color, marker, label):
    est, lo, hi = map(np.asarray, (est, lo, hi))
    ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt=marker, ms=4.5, color=color, mfc=color,
                mec="white", mew=0.6, ecolor=color, elinewidth=1, capsize=2.5, label=label, zorder=3)


# ---------------- Figure 1 ----------------
def fig1():
    r = pd.read_csv(CSV / "r1_c1_reliability_by_subset.csv")
    r = r[r.item == "overall_global_quality"]
    d = pd.read_csv(CSV / "r1_11_absolute_disagreement.csv"); d = d[d.set == "A_all100"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(174 * MM, 62 * MM), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.45})
    x = np.arange(4)
    for off, sub, col, mk, lab in [(-0.12, "A_all100", S1, "o", "Full set (n = 100)"),
                                   (0.12, "original30", S2, "s", "Intended originals (n = 30)")]:
        q = r[r.subset == sub].set_index("judge").loc[JUDGES]
        lo = q.ICC_A1_lo.where(q.ICC_A1_lo.notna(), q.ICC_A1); hi = q.ICC_A1_hi.where(q.ICC_A1_hi.notna(), q.ICC_A1)
        errpts(a, x + off, q.ICC_A1.clip(upper=1), lo.clip(upper=1), hi.clip(upper=1), col, mk, lab)
    a.set_xticks(x); a.set_xticklabels([j.replace(" ", "\n", 1) for j in JUDGES])
    a.set_ylim(0, 1.02); a.set_yticks(np.arange(0, 1.01, 0.2)); a.set_ylabel("ICC[2,1], overall global quality (95% CI)")
    grid(a); a.legend(loc="lower left", frameon=False); panel(a, "a")
    # b: distribution of run-pair absolute differences (full set)
    p1 = d[d.metric == "p_diff_ge1"].set_index("judge").estimate.loc[JUDGES].values
    p2 = d[d.metric == "p_diff_ge2"].set_index("judge").estimate.loc[JUDGES].values
    mad = d[d.metric == "mean_abs_pair_diff"].set_index("judge").loc[JUDGES]
    parts = [(1 - p1) * 100, (p1 - p2) * 100, p2 * 100]
    labs = ["Identical", "Differ by 1 point", "Differ by ≥2 points"]
    y = np.arange(4)[::-1]; left = np.zeros(4)
    for v, c, l in zip(parts, SEQ, labs):
        b.barh(y, v, left=left, color=c, edgecolor="white", linewidth=1.0, height=0.6, label=l)
        for yi, li, vi in zip(y, left, v):
            if vi >= 6:
                b.text(li + vi / 2, yi, f"{vi:.1f}", ha="center", va="center", fontsize=7,
                       color="white" if c != SEQ[0] else "#0b0b0b")
        left += v
    b.set_yticks(y); b.set_yticklabels(JUDGES); b.set_xlim(0, 100)
    b.set_xlabel("Run pairs within the same report (%)")
    for yi, (_, m) in zip(y, mad.iterrows()):
        b.text(101.5, yi, f"{m.estimate:.3f}\n({m.lo:.3f}–{m.hi:.3f})" if m.estimate > 0 else "0.000",
               va="center", fontsize=6.5, color=INK2)
    b.text(101.5, 3.62, "Mean |difference|\n(95% CI)", fontsize=6.5, color=INK2, va="bottom")
    b.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3, frameon=False, handlelength=1.2)
    b.spines["left"].set_visible(False); b.tick_params(axis="y", length=0); panel(b, "b")
    save(fig, "Figure1")


# ---------------- Figure 2 ----------------
def fig2():
    t = pd.read_csv(CSV / "r1_10_temperature.csv"); t = t[t.set == "A_all100"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(174 * MM, 62 * MM), gridspec_kw={"wspace": 0.35})
    for j, col, mk, dx in [("Gemini 2.5 Pro", S1, "o", -0.012), ("Qwen 3.5", S2, "s", 0.012)]:
        q = t[t.judge == j].sort_values("temperature"); xx = q.temperature + dx
        for ax, est, lo, hi in [(a, q.within_SD, q.within_SD_lo, q.within_SD_hi), (b, q["mean"], q.mean_lo, q.mean_hi)]:
            ax.plot(xx, est, color=col, lw=1.2, zorder=2)
            errpts(ax, xx, est, lo, hi, col, mk, j)
    for ax in (a, b):
        ax.set_xticks([0, 0.3, 0.7, 1.0]); ax.set_xlim(-0.06, 1.06); ax.set_xlabel("Temperature"); grid(ax)
    a.set_ylabel("Within-report SD across runs (95% CI)"); a.set_ylim(-0.02, 0.7)
    b.set_ylabel("Mean overall global quality (95% CI)"); b.set_ylim(1, 5); b.set_yticks([1, 2, 3, 4, 5])
    a.legend(loc="upper left", frameon=False)
    panel(a, "a"); panel(b, "b"); save(fig, "Figure2")


# ---------------- Figure 3 ----------------
def fig3():
    p = pd.read_csv(CSV / "r1_9_r2_3_prompt.csv"); p = p[p.set == "A_all100"]
    get = lambda m: p[p.metric == m].set_index("judge").loc[JUDGES]
    fig, (a, b) = plt.subplots(1, 2, figsize=(174 * MM, 62 * MM), gridspec_kw={"wspace": 0.35})
    x = np.arange(4)
    for off, m, col, mk, lab in [(-0.12, "range_of_prompt_grand_means", S1, "o", "Range of prompt-level grand means"),
                                 (0.12, "mean_within_report_range", S2, "s", "Mean within-report range")]:
        q = get(m); errpts(a, x + off, q.estimate, q.lo, q.hi, col, mk, lab)
    a.set_ylim(0, 1.0); a.set_ylabel("Points on the 1–5 scale (95% CI)")
    a.legend(loc="upper left", frameon=False)
    q = get("kendall_W_tie_corrected"); errpts(b, x, q.estimate, q.lo, q.hi, S1, "o", None)
    b.axhline(0.90, ls="--", color=INK2, lw=0.7); b.set_ylim(0, 1.02); b.set_yticks(np.arange(0, 1.01, 0.2))
    b.set_ylabel("Kendall's W, tie-corrected (95% CI)")
    for ax in (a, b):
        ax.set_xticks(x); ax.set_xticklabels([j.replace(" ", "\n", 1) for j in JUDGES]); ax.set_xlim(-0.5, 3.5); grid(ax)
    panel(a, "a"); panel(b, "b"); save(fig, "Figure3")


# ---------------- Figure 4 ----------------
def fig4():
    v = pd.read_csv(CSV / "r1_12_mixed_variance_components.csv")
    cond = {"C2_temperature": "C2, temperature", "C3_prompt": "C3, prompt"}
    v["lab"] = v.judge + " · " + v.condition.map(cond)
    order = [("Gemini 2.5 Pro", "C2_temperature"), ("Qwen 3.5", "C2_temperature")] + [(j, "C3_prompt") for j in JUDGES]
    v = v.set_index(["judge", "condition"]).loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(120 * MM, 62 * MM))
    y = np.arange(len(v))[::-1]; left = np.zeros(len(v))
    comps = [("pct_report", "Report", S1), ("pct_report_x_condition", "Report × condition", S2),
             ("pct_residual_run", "Run-to-run (residual)", S3)]
    for c, l, col in comps:
        ax.barh(y, v[c], left=left, color=col, edgecolor="white", linewidth=1.0, height=0.62, label=l)
        for yi, li, vi in zip(y, left, v[c]):
            if vi >= 5:
                ax.text(li + vi / 2, yi, f"{vi:.1f}", ha="center", va="center", fontsize=7, color="white")
        left += v[c].values
    ax.set_yticks(y); ax.set_yticklabels(v.lab); ax.set_xlim(0, 100)
    ax.set_xlabel("Share of modeled variance (%)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=3, frameon=False, handlelength=1.2)
    save(fig, "Figure4_stacked_unused")


# ---------------- Figure 5 ----------------
def fig5():
    m = pd.read_csv(CSV / "r1_13_intermodel.csv")
    m = m[(m.set == "A_all100") & (m.metric == "kappa_rounded_means")]
    n = len(JUDGES); K = np.full((n, n), np.nan); T = [["" for _ in range(n)] for _ in range(n)]
    for _, r in m.iterrows():
        i, j = JUDGES.index(r.judge_a), JUDGES.index(r.judge_b)
        i, j = max(i, j), min(i, j)
        K[i, j] = r.estimate; T[i][j] = f"{r.estimate:.2f}\n({r.lo:.2f}–{r.hi:.2f})"
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    fig, ax = plt.subplots(figsize=(100 * MM, 82 * MM))
    im = ax.imshow(np.ma.masked_invalid(K), cmap=cmap, vmin=0.5, vmax=1.0)
    for i in range(n):
        for j in range(n):
            if T[i][j]:
                ax.text(j, i, T[i][j], ha="center", va="center", fontsize=7,
                        color="white" if K[i, j] > 0.70 else "#0b0b0b")
    ax.set_xticks(range(n)); ax.set_xticklabels([j.replace(" ", "\n", 1) for j in JUDGES])
    ax.set_yticks(range(n)); ax.set_yticklabels(JUDGES)
    ax.set_xlim(-0.5, n - 1.5); ax.set_ylim(n - 0.5, 0.5)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("Quadratic weighted κ (color scale 0.50–1.00)"); cb.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); cb.outline.set_visible(False)
    save(fig, "Figure5")


# ---------------- Figure 4 (alternative with bootstrap CIs) ----------------
def fig4_ci(vc_csv):
    v = pd.read_csv(vc_csv)
    cond = {"C2_temperature": "C2, temperature", "C3_prompt": "C3, prompt"}
    order = [("Gemini 2.5 Pro", "C2_temperature"), ("Qwen 3.5", "C2_temperature")] + [(j, "C3_prompt") for j in JUDGES]
    comps = [("report", "Report", S1, (0, 100)), ("report_x_condition", "Report × condition", S2, (0, 25)),
             ("residual_run", "Run-to-run (residual)", S3, (0, 25))]
    fig, axs = plt.subplots(1, 3, figsize=(174 * MM, 66 * MM), sharey=True,
                            gridspec_kw={"width_ratios": [1.6, 1, 1], "wspace": 0.15})
    y = np.arange(len(order))[::-1]
    for ax, (c, title, col, xl), lab in zip(axs, comps, "abc"):
        q = v[v.component == c].set_index(["judge", "condition"]).loc[order]
        est, lo, hi = q.pct.values, q.lo.values, q.hi.values
        ax.errorbar(est, y, xerr=[est - lo, hi - est], fmt="o", ms=4.5, color=col, mec="white", mew=0.6,
                    ecolor=col, elinewidth=1, capsize=2.5, zorder=3)
        ax.set_xlim(*xl); ax.set_title(r"$\bf{" + lab + "}$   " + title, fontsize=8.5, loc="left"); grid(ax, "x")
    axs[0].set_yticks(y); axs[0].set_yticklabels([f"{j} · {cond[c]}" for j, c in order])
    fig.supxlabel("Share of modeled variance (%); panels a and b–c use different scales", fontsize=8, y=-0.06)
    axs[2].annotate("0 (identical runs)", (0, y[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6.5, color=INK2)
    for ax in axs:
        ax.tick_params(axis="y", length=0)
    save(fig, "Figure4")


for f in (fig1, fig2, fig3, fig4, fig5):
    f()
if len(sys.argv) > 3:
    fig4_ci(sys.argv[3])
print("done", sorted(p.name for p in OUT.iterdir()))
