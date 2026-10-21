"""reanalysis_phase1.py — DHJ-26-2797 revision, phase-1 reanalysis per SAP_revision_v1.md (locked 2026-10-06).
Inputs: input/scores_rebuilt_32000.parquet, input/report_status.csv. Outputs: out/*.csv. Deterministic seeds."""
import os, sys, json, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score as _sk_kappa

def cohen_kappa_score(a, b, weights='quadratic'):
    """Fast quadratic-weighted kappa identical to sklearn's (labels = union of observed values, positional weights)."""
    a = np.asarray(a); b = np.asarray(b)
    labs = np.unique(np.concatenate([a, b])); L = len(labs)
    if L < 2: return np.nan
    ia = np.searchsorted(labs, a); ib = np.searchsorted(labs, b)
    O = np.zeros((L, L)); np.add.at(O, (ia, ib), 1)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = (np.arange(L)[:, None] - np.arange(L)[None, :]) ** 2
    den = (W * E).sum()
    return 1 - (W * O).sum() / den if den > 0 else np.nan

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
B = int(os.environ.get('SAP_B', 2000)); SEED = 20261006
ITEMS = ['finding_completeness','diagnostic_accuracy','clarity_structure','clinical_appropriateness','overall_global_quality']
JUDGES = ['claude-opus-4-8','gpt-5.5','gemini-2.5-pro','qwen-3.5']
LAB = {'claude-opus-4-8':'Claude Opus 4.8','gpt-5.5':'GPT-5.5','gemini-2.5-pro':'Gemini 2.5 Pro','qwen-3.5':'Qwen 3.5'}
df = pd.read_parquet(os.path.join(HERE,'input','scores_rebuilt_32000.parquet'))
st = pd.read_csv(os.path.join(HERE,'input','report_status.csv'))
SETS = {'A_all100': st.report_id.tolist(),
        'B_88': st[st.group3!='failed_perturbation'].report_id.tolist()}
rng0 = np.random.default_rng(SEED)

def icc_a(X):
    """X: n targets x k raters (no NaN). Returns ICC(A,1), ICC(A,k) (McGraw&Wong; = pingouin ICC2/ICC2k)."""
    n,k = X.shape
    gm = X.mean(); rm = X.mean(1); cm = X.mean(0)
    ssr = k*((rm-gm)**2).sum(); ssc = n*((cm-gm)**2).sum(); sst = ((X-gm)**2).sum(); sse = sst-ssr-ssc
    msr = ssr/(n-1); msc = ssc/(k-1); mse = sse/((n-1)*(k-1))
    d1 = msr+(k-1)*mse+k*(msc-mse)/n; dk = msr+(msc-mse)/n
    i1 = (msr-mse)/d1 if d1>0 else np.nan; ik = (msr-mse)/dk if dk>0 else np.nan
    return i1, ik

def pairdiff(M):
    a,b = np.triu_indices(M.shape[1],1); return np.abs(M[:,a]-M[:,b])

def boot(stat_fn, ids, B=B, seed=0):
    rng = np.random.default_rng([SEED, seed]); ids = np.asarray(ids); vals=[]
    for _ in range(B):
        v = stat_fn(rng.choice(ids, size=len(ids), replace=True))
        vals.append(v)
    vals = np.array(vals, dtype=float)
    return np.nanpercentile(vals, 2.5, axis=0), np.nanpercentile(vals, 97.5, axis=0)

def c1_matrix(judge, item):
    d = df[(df.judge_id==judge)&(df.condition=='C1')]
    return d.pivot(index='report_id', columns='run', values=item)

C1 = {(j,i): c1_matrix(j,i) for j in JUDGES for i in ITEMS}
seedc = itertools.count(1)
SECTIONS = set(os.environ.get('SECTIONS','1,2,3,4,5,6,7,8').split(','))
def run(sec): return str(sec) in SECTIONS

# ---- shared C3 helpers (module level so sections run independently) ----
def c3_means(j):
    d = df[(df.judge_id==j)&(df.condition=='C3')]
    return d.groupby(['report_id','variant']).overall_global_quality.mean().unstack()   # report x v1..v5
def kendall_w_uncorrected(R):   # as in the submitted analysis (src/analysis/effects.py): average ranks, no tie correction
    import pandas as _pd
    n,m = R.shape; ranks = np.column_stack([_pd.Series(R[:,c]).rank().values for c in range(m)])
    Ri = ranks.sum(1); return 12*((Ri-Ri.mean())**2).sum()/(m**2*(n**3-n))
def kendall_w(R):   # R: n objects x m raters (scores) -> W with tie correction
    from scipy.stats import rankdata
    n,m = R.shape; ranks = np.column_stack([rankdata(R[:,c]) for c in range(m)])
    S = ((ranks.sum(1)-m*(n+1)/2)**2).sum()
    T = sum(((np.unique(R[:,c],return_counts=True)[1]**3-np.unique(R[:,c],return_counts=True)[1]).sum()) for c in range(m))
    return 12*S/(m**2*(n**3-n)-m*T)

# ---- section 1 ----
seedc = itertools.count(1000)
if run(1):
    # ---------- 1. C1 reliability by subset (R1-2, R1-14, R2-4) ----------
    subsets = dict(SETS)
    subsets['original30'] = st[st.group3=='original'].report_id.tolist()
    subsets['unchanged42'] = st[~st.actual_text_changed].report_id.tolist()
    subsets['changed58'] = st[st.actual_text_changed].report_id.tolist()
    for fam in ['finding_omission','hallucinated_finding','laterality_measure','readability']:
        subsets[f'fam_nominal:{fam}'] = st[st.assigned_family==fam].report_id.tolist()
        subsets[f'fam_changed:{fam}'] = st[(st.assigned_family==fam)&st.actual_text_changed].report_id.tolist()
    for it in [1,2,3]:
        subsets[f'intensity_nominal:{it}'] = st[st.nominal_intensity==it].report_id.tolist()
    rows=[]
    for sname, ids in subsets.items():
        for j in JUDGES:
            for item in ITEMS:
                M = C1[(j,item)].loc[ids].to_numpy(float)
                i1, ik = icc_a(M)
                pair = pairdiff(M)
                if np.isnan(i1) or M.var(axis=0).sum()==0 and M.var(axis=1).sum()==0:
                    lo=hi=(np.nan,np.nan)
                else:
                    pos = {r:i for i,r in enumerate(ids)}
                    lo, hi = boot(lambda s: icc_a(M[[pos[r] for r in s]]), ids, seed=next(seedc))
                within_var = M.var(axis=1, ddof=1).mean()
                rows.append(dict(subset=sname, n=len(ids), judge=LAB[j], item=item, ICC_A1=i1, ICC_A1_lo=lo[0], ICC_A1_hi=hi[0],
                                 ICC_A10=ik, ICC_A10_lo=lo[1], ICC_A10_hi=hi[1], mean_score=M.mean(), between_report_SD=M.mean(1).std(ddof=1),
                                 mean_within_report_var=within_var, mean_abs_pair_diff=pair.mean(), p_diff_ge1=(pair>=1).mean(), p_diff_ge2=(pair>=2).mean()))
    pd.DataFrame(rows).to_csv(os.path.join(OUT,'r1_c1_reliability_by_subset.csv'), index=False)
    print('[1] subsets done', flush=True)

# ---- section 2 ----
seedc = itertools.count(2000)
if run(2):
    # ---------- 2. absolute disagreement CIs + transition matrices (R1-11) ----------
    rows=[]; trans=[]
    for setname, ids in SETS.items():
        for j in JUDGES:
            M = C1[(j,'overall_global_quality')].loc[ids].to_numpy(float); pos={r:i for i,r in enumerate(ids)}
            def stats(s):
                X = M[[pos[r] for r in s]]; P = pairdiff(X)
                kap = np.mean([cohen_kappa_score(X[:,a],X[:,b],weights='quadratic') if len(set(X[:,a]))>1 or len(set(X[:,b]))>1 else np.nan for a,b in itertools.combinations(range(10),2)])
                return [P.mean(), (P>=1).mean(), (P>=2).mean(), kap]
            est = stats(ids); lo, hi = boot(stats, ids, seed=next(seedc))
            for nm,e,l,h in zip(['mean_abs_pair_diff','p_diff_ge1','p_diff_ge2','weighted_kappa_runpairs'], est, lo, hi):
                rows.append(dict(set=setname, judge=LAB[j], metric=nm, estimate=e, lo=l, hi=h))
            if setname=='A_all100':
                a,b = np.triu_indices(10,1); T = pd.crosstab(M[:,a].ravel(), M[:,b].ravel())
                for (x,y),c in T.stack().items(): trans.append(dict(judge=LAB[j], score_run_i=int(x), score_run_j=int(y), count=int(c)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT,'r1_11_absolute_disagreement.csv'), index=False)
    pd.DataFrame(trans).to_csv(os.path.join(OUT,'r1_11_transition_counts.csv'), index=False)
    print('[2] disagreement done', flush=True)

# ---- section 3 ----
seedc = itertools.count(3000)
if run(3):
    # ---------- 3. hypothetical case-mix (R1-2) ----------
    orig = st[st.group3=='original'].report_id.tolist()
    i1c = st[(st.actual_text_changed)&(st.nominal_intensity==1)].report_id.tolist()
    i23c = st[(st.actual_text_changed)&(st.nominal_intensity>=2)].report_id.tolist()
    scen = {'S1_original100': [(orig,1.0)], 'S2_orig80_i1_20': [(orig,.8),(i1c,.2)], 'S3_orig60_i1_25_i23_15': [(orig,.6),(i1c,.25),(i23c,.15)]}
    rows=[]
    for sname, parts in scen.items():
        for j in JUDGES:
            M = C1[(j,'overall_global_quality')]; rng = np.random.default_rng([SEED, next(seedc)]); vals=[]
            for _ in range(B):
                s = np.concatenate([rng.choice(p, size=int(round(100*w)), replace=True) for p,w in parts])
                vals.append(icc_a(M.loc[s].to_numpy(float))[0])
            rows.append(dict(scenario=sname, judge=LAB[j], ICC_A1_median=np.nanmedian(vals), p2_5=np.nanpercentile(vals,2.5), p97_5=np.nanpercentile(vals,97.5), n_undefined=int(np.isnan(vals).sum())))
    pd.DataFrame(rows).to_csv(os.path.join(OUT,'r1_2_hypothetical_casemix.csv'), index=False)
    print('[3] case-mix done', flush=True)

# ---- section 4 ----
seedc = itertools.count(4000)
if run(4):
    # ---------- 4. temperature (C2) with CIs ----------
    rows=[]
    for setname, ids in SETS.items():
        for j in ['gemini-2.5-pro','qwen-3.5']:
            d = df[(df.judge_id==j)&(df.condition=='C2')]
            for T in sorted(d.temperature.unique()):
                M = d[d.temperature==T].pivot(index='report_id', columns='run', values='overall_global_quality').loc[ids]
                A = M.to_numpy(float); pos={r:i for i,r in enumerate(ids)}
                f = lambda s: [np.nanmean(A[[pos[r] for r in s]]), np.nanmean(np.nanstd(A[[pos[r] for r in s]],axis=1,ddof=1))]
                est=f(ids); lo,hi=boot(f, ids, seed=next(seedc))
                rows.append(dict(set=setname, judge=LAB[j], temperature=T, mean=est[0], mean_lo=lo[0], mean_hi=hi[0], within_SD=est[1], within_SD_lo=lo[1], within_SD_hi=hi[1], n_missing=int(np.isnan(A).sum())))
    pd.DataFrame(rows).to_csv(os.path.join(OUT,'r1_10_temperature.csv'), index=False)
    print('[4] temperature done', flush=True)

# ---- section 5 ----
seedc = itertools.count(5000)
if run(5):
    # ---------- 5. prompt (C3) per-variant, within-report, W (R1-9, R2-3, R1-10) ----------
    rows=[]; ba=[]
    for setname, ids in SETS.items():
        for j in JUDGES:
            V = c3_means(j).loc[ids]; A = V.to_numpy(float); pos={r:i for i,r in enumerate(ids)}
            def f(s):
                X = A[[pos[r] for r in s]]; g = X.mean(0)
                rng_rep = X.max(1)-X.min(1); R = np.round(X)   # numpy round = half-to-even
                ge1 = ((R.max(1)-R.min(1))>=1).mean(); ge1_unr = (rng_rep>=1).mean()
                sp = np.mean([spearmanr(X[:,a],X[:,b]).correlation for a,b in itertools.combinations(range(5),2)])
                return [g.max()-g.min(), rng_rep.mean(), ge1, ge1_unr, kendall_w(X), kendall_w_uncorrected(X), sp] + list(X[:,1:].mean(0)-X[:,0].mean())
            est=f(ids); lo,hi=boot(f, ids, seed=next(seedc))
            names=['range_of_prompt_grand_means','mean_within_report_range','prop_reports_rounded_change_ge1','prop_reports_unrounded_range_ge1','kendall_W_tie_corrected','kendall_W_uncorrected_as_submitted','mean_pairwise_spearman','shift_v2_vs_v1','shift_v3_vs_v1','shift_v4_vs_v1','shift_v5_vs_v1']
            for nm,e,l,h in zip(names,est,lo,hi): rows.append(dict(set=setname, judge=LAB[j], metric=nm, estimate=e, lo=l, hi=h))
            if setname=='A_all100':
                for v in range(1,5):
                    dd = A[:,v]-A[:,0]; ba.append(dict(judge=LAB[j], variant=f'v{v+1}', mean_diff=dd.mean(), loa_lo=dd.mean()-1.96*dd.std(ddof=1), loa_hi=dd.mean()+1.96*dd.std(ddof=1)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT,'r1_9_r2_3_prompt.csv'), index=False)
    pd.DataFrame(ba).to_csv(os.path.join(OUT,'r1_9_bland_altman.csv'), index=False)
    print('[5] prompt done', flush=True)

# ---- section 6 ----
seedc = itertools.count(6000)
if run(6):
    # ---------- 6. few-shot overlap subgroup + leave-template-out (R1-5) ----------
    ov = st[st.fewshot_filler_overlap].report_id.tolist(); nonov = st[~st.fewshot_filler_overlap].report_id.tolist()
    rows=[]
    for j in JUDGES:
        V = c3_means(j); d5 = (V['v5']-V['v1'])
        rows.append(dict(judge=LAB[j], analysis='overlap_subgroup', n_overlap=len(ov), shift_v5_v1_overlap=d5.loc[ov].mean(), shift_v5_v1_other=d5.loc[nonov].mean(), difference=d5.loc[ov].mean()-d5.loc[nonov].mean()))
        for nm, ids in [('all100', st.report_id.tolist()), ('leave_template_out', nonov)]:
            X = V.loc[ids].to_numpy(float); g=X.mean(0)
            rows.append(dict(judge=LAB[j], analysis=nm, n=len(ids), range_of_prompt_grand_means=g.max()-g.min(), kendall_W_tie_corrected=kendall_w(X), kendall_W_uncorrected=kendall_w_uncorrected(X), shift_v5_vs_v1=g[4]-g[0]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT,'r1_5_fewshot_overlap.csv'), index=False)
    print('[6] few-shot done', flush=True)

# ---- section 7 ----
seedc = itertools.count(7000)
if run(7):
    # ---------- 7. inter-model agreement (R1-13) ----------
    def lin_ccc(x,y):
        mx,my=x.mean(),y.mean(); return 2*np.cov(x,y,ddof=0)[0,1]/(x.var()+y.var()+(mx-my)**2)
    rows=[]
    for setname, ids in SETS.items():
        mats = {j: C1[(j,'overall_global_quality')].loc[ids].to_numpy(float) for j in JUDGES}
        pos={r:i for i,r in enumerate(ids)}
        for ja, jb in itertools.combinations(JUDGES,2):
            A,Bm = mats[ja], mats[jb]
            def f(s):
                idx=[pos[r] for r in s]; a,b = A[idx],Bm[idx]; ma,mb=a.mean(1),b.mean(1)
                ra,rb = np.round(ma),np.round(mb)
                k_round = cohen_kappa_score(ra,rb,weights='quadratic'); exact=(ra==rb).mean()
                icc_un = icc_a(np.column_stack([ma,mb]))[0]; ccc = lin_ccc(ma,mb)
                k_single = np.mean([cohen_kappa_score(a[:,i],b[:,i],weights='quadratic') for i in range(10)])
                return [k_round, exact, icc_un, ccc, k_single]
            est=f(ids); lo,hi=boot(f, ids, seed=next(seedc))
            for nm,e,l,h in zip(['kappa_rounded_means','exact_agreement_rounded','ICC_A1_unrounded_means','lin_CCC_unrounded_means','kappa_single_run_matched'],est,lo,hi):
                rows.append(dict(set=setname, judge_a=LAB[ja], judge_b=LAB[jb], metric=nm, estimate=e, lo=l, hi=h))
    pd.DataFrame(rows).to_csv(os.path.join(OUT,'r1_13_intermodel.csv'), index=False)
    print('[7] inter-model done', flush=True)

# ---- section 8 ----
seedc = itertools.count(8000)
if run(8):
    # ---------- 8. k-curve (R1-15) ----------
    rows=[]
    KSETS = {'A_all100': SETS['A_all100'], 'B_88': SETS['B_88'], 'original30': st[st.group3=='original'].report_id.tolist()}
    for kset, kids in KSETS.items():
     for j in JUDGES:
        M = C1[(j,'overall_global_quality')].loc[kids].to_numpy(float); n=M.shape[0]
        i1,_ = icc_a(M)
        for k in [1,2,3,5,10]:
            model_k = (k*i1)/(1+(k-1)*i1) if not np.isnan(i1) else np.nan
            emp=np.nan; emp_lo=emp_hi=np.nan
            if k<=5:
                rng=np.random.default_rng([SEED,next(seedc)]); vals=[]
                for _ in range(500):
                    perm = np.array([rng.permutation(10) for _ in range(n)])
                    a = np.take_along_axis(M, perm[:,:k], 1).mean(1); b = np.take_along_axis(M, perm[:,k:2*k], 1).mean(1)
                    vals.append(icc_a(np.column_stack([a,b]))[0])
                emp=np.nanmean(vals); emp_lo,emp_hi=np.nanpercentile(vals,[2.5,97.5])
            rows.append(dict(set=kset, judge=LAB[j], k=k, ICC_Ak_model=model_k, ICC_Ak_empirical_split=emp, run_split_variation_p2_5=emp_lo, run_split_variation_p97_5=emp_hi))  # NOT a report-bootstrap CI
    pd.DataFrame(rows).to_csv(os.path.join(OUT,'r1_15_k_curve.csv'), index=False)
    print('[8] k-curve done', flush=True)
json.dump(dict(B=B, seed=SEED, n_rows=len(df), n_valid=int(df.parse_ok.sum())), open(os.path.join(OUT,'run_info.json'),'w'))
