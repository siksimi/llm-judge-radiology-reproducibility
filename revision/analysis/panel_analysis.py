"""Panel analysis per SAP_appendix_panel_v1.md (Task 1 external readers A/B; Task 2 co-author raters R1/R2).
Repository version: reads the de-identified rating tables in ../panel/ (free-text comments removed).
Usage: python panel_analysis.py   (run from analysis/)"""
import hashlib, json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
P = Path('..') / 'panel'; OUT = Path('out'); OUT.mkdir(exist_ok=True)
B, SEED = 2000, 20261012
ITEMS = ['finding_completeness','diagnostic_accuracy','clarity_structure','clinical_appropriateness','overall_global_quality']
Q1CAT = ['worse','better','no difference','cannot assess','no text change']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
INPUTS = {k: sha(v) for k, v in dict(t1a=P/'task1_readerA.csv', t1b=P/'task1_readerB.csv',
          t2=P/'task2_responses.csv', manifest=P/'panel_sample_manifest.csv', key=P/'forms_order_key.csv',
          judges='input/scores_rebuilt_32000.parquet').items()}

man = pd.read_csv(P/'panel_sample_manifest.csv'); key = pd.read_csv(P/'forms_order_key.csv', encoding='utf-8-sig')
man['w'] = 1/man.inclusion_prob
fam = lambda s: 'original' if s=='original' else 'noop' if s=='noop' else 'laterality' if s=='laterality_changed' else s.rsplit('_i',1)[0]
man['family'] = man.stratum.map(fam)

# ---------- stats helpers ----------
def qwk(a, b, cats=(1,2,3,4,5)):
    a, b = np.asarray(a), np.asarray(b); k = len(cats); idx = {c:i for i,c in enumerate(cats)}
    O = np.zeros((k,k));
    for x,y in zip(a,b): O[idx[x], idx[y]] += 1
    W = np.array([[(i-j)**2 for j in range(k)] for i in range(k)])/(k-1)**2
    E = np.outer(O.sum(1), O.sum(0))/O.sum()
    d = (W*E).sum(); return np.nan if d==0 else 1-(W*O).sum()/d
def kappa(a, b, cats):
    a, b = list(a), list(b); n = len(a)
    if n == 0: return np.nan
    po = np.mean([x==y for x,y in zip(a,b)]); pe = sum((a.count(c)/n)*(b.count(c)/n) for c in cats)
    return np.nan if pe == 1 else (po-pe)/(1-pe)
def icc_a1(X, w=None):
    X = np.asarray(X, float); n, k = X.shape
    w = np.ones(n) if w is None else np.asarray(w, float)*n/np.sum(w)
    g = np.sum(w[:,None]*X)/(n*k); r = X.mean(1); c = (w[:,None]*X).sum(0)/n
    SSR = k*np.sum(w*(r-g)**2); SSC = n*np.sum((c-g)**2); SSE = np.sum(w[:,None]*(X-r[:,None]-c[None,:]+g)**2)
    MSR, MSC, MSE = SSR/(n-1), SSC/(k-1), SSE/((n-1)*(k-1))
    den = MSR+(k-1)*MSE+k*(MSC-MSE)/n; return np.nan if den == 0 else (MSR-MSE)/den
def wilson(x, n, z=1.959964):
    if n == 0: return (np.nan, np.nan)
    p = x/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return (c-h, c+h)
def boot(df, fn, strata=None, b=B, seed=SEED):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(b):
        if strata is None: s = df.iloc[rng.integers(0, len(df), len(df))]
        else: s = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for _, g in df.groupby(strata)])
        try: vals.append(fn(s))
        except Exception: vals.append(np.nan)
    v = np.array(vals, float); v = v[~np.isnan(v)]
    return (np.nan, np.nan) if len(v) < 0.9*b else tuple(np.percentile(v, [2.5, 97.5]))
fmt = lambda x: 'NA' if x is None or (isinstance(x,float) and np.isnan(x)) else f'{x:.3f}'

# ---------- Task 1 ----------
def load_t1(path, file):
    d = pd.read_csv(path, encoding='utf-8-sig', dtype=str).merge(key[key.file==file][['code','report_id']], left_on='Code', right_on='code', how='left')
    assert d.report_id.notna().all() and len(d)==48
    for it in ITEMS: d[it] = pd.to_numeric(d[it].replace({'NA':np.nan,'N/A':np.nan}), errors='coerce')
    return d.set_index('report_id')[ITEMS]
A = load_t1(P/'task1_readerA.csv', 'Task1_ReaderA.xlsx'); Bd = load_t1(P/'task1_readerB.csv', 'Task1_ReaderB.xlsx')
t1 = man.set_index('report_id')[['stratum','family','inclusion_prob','w','actual_text_changed']].join(A[ITEMS].add_prefix('A_')).join(Bd[ITEMS].add_prefix('B_')).reset_index()

rows = []
for it in ITEMS:
    d = t1[['A_'+it,'B_'+it]].dropna(); a, b = d.iloc[:,0].astype(int), d.iloc[:,1].astype(int)
    f_k = lambda s: qwk(s.iloc[:,0].astype(int), s.iloc[:,1].astype(int)); f_i = lambda s: icc_a1(s.values)
    rows.append(dict(item=it, n_both=len(d), NA_A=int(t1['A_'+it].isna().sum()), NA_B=int(t1['B_'+it].isna().sum()),
        qwk=qwk(a,b), qwk_ci=boot(d, f_k), icc_a1=icc_a1(d.values), icc_ci=boot(d, f_i),
        exact=float(np.mean(a==b)), within1=float(np.mean(abs(a-b)<=1)), diff_ge2=int(np.sum(abs(a-b)>=2)), meanA=a.mean(), meanB=b.mean()))
T11 = pd.DataFrame(rows); T11.to_csv(OUT/'panel_T1_1_interreader.csv', index=False, encoding='utf-8-sig')

t1['R_overall'] = t1[['A_overall_global_quality','B_overall_global_quality']].mean(1, skipna=True)
t1['diff_ov'] = (t1.A_overall_global_quality - t1.B_overall_global_quality).abs()
j = pd.read_parquet('input/scores_rebuilt_32000.parquet')
j = j[(j.condition=='C1') & j.parse_ok.astype(bool)].groupby(['judge_id','report_id']).overall_global_quality.agg(['mean','count']).reset_index()
rows = []
for jid, g in j.groupby('judge_id'):
    d = t1.merge(g.rename(columns={'mean':'J'}), on='report_id').dropna(subset=['R_overall','J'])
    for label, dd in (('all', d), ('excl_reader_diff_ge2', d[~(d.diff_ov>=2)])):
        wmd = lambda s: np.average(s.J - s.R_overall, weights=s.w); umd = lambda s: np.mean(s.J - s.R_overall)
        wic = lambda s: icc_a1(s[['J','R_overall']].values, s.w); uic = lambda s: icc_a1(s[['J','R_overall']].values)
        rows.append(dict(judge=jid, set=label, n=len(dd), min_runs=int(dd['count'].min()),
            wmeandiff=wmd(dd), wmeandiff_ci=boot(dd, wmd, 'stratum'), wicc=wic(dd), wicc_ci=boot(dd, wic, 'stratum'),
            meandiff_unw=umd(dd), meandiff_unw_ci=boot(dd, umd), icc_unw=uic(dd), icc_unw_ci=boot(dd, uic)))
T13 = pd.DataFrame(rows); T13.to_csv(OUT/'panel_T1_3_judge_calibration.csv', index=False, encoding='utf-8-sig')

rows = []
for r in 'AB':
    for grp, dd in (('original', t1[t1.stratum=='original']), ('perturbed', t1[t1.stratum!='original'])):
        for it in ITEMS:
            v = dd[f'{r}_{it}'].dropna(); rows.append(dict(reader=r, group=grp, item=it, n=len(v), prop5=(v==5).mean() if len(v) else np.nan,
                **{f'n{s}': int((v==s).sum()) for s in range(1,6)}, NA=int(dd[f'{r}_{it}'].isna().sum())))
T14 = pd.DataFrame(rows); T14.to_csv(OUT/'panel_T1_4_ceiling.csv', index=False, encoding='utf-8-sig')
T15 = t1.groupby('stratum').apply(lambda g: pd.Series(dict(n=len(g), mean=g.R_overall.mean(), ci=boot(g[['R_overall']].dropna(), lambda s: s.R_overall.mean()),
        minv=g.R_overall.min(), maxv=g.R_overall.max())), include_groups=False).reset_index()
T15.to_csv(OUT/'panel_T1_5_by_stratum.csv', index=False, encoding='utf-8-sig')

# ---------- Task 2 ----------
t2 = pd.read_csv(P/'task2_responses.csv', encoding='utf-8-sig', dtype=str)
k2 = key[key.file=='Task2_Sim.xlsx'][['code','report_id']]
t2 = t2.merge(k2, on='code', how='left'); assert t2.report_id.notna().all()
t2 = t2.merge(man[['report_id','stratum','family','nominal_intensity','actual_text_changed']], on='report_id', how='left')
W = t2.pivot(index='report_id', columns='reader', values=['Q1','Q2','Q3'])
W.columns = [f'{q}_{r}' for q, r in W.columns]; W = W.join(man.set_index('report_id')[['stratum','family','nominal_intensity','actual_text_changed']]).reset_index()
assert len(W) == 36

rows = []
for lvl, col in (('stratum','stratum'), ('family','family')):
    for grp, g in W.groupby(col):
        for r in ('R1','R2'):
            vc = g[f'Q1_{r}'].value_counts(); rows.append(dict(level=lvl, group=grp, reader=r, n=len(g), **{c:int(vc.get(c,0)) for c in Q1CAT}))
T21 = pd.DataFrame(rows); T21.to_csv(OUT/'panel_T2_1_Q1_distribution.csv', index=False, encoding='utf-8-sig')
rows = []
for grp, g in list(W.groupby('family')) + [('ALL_changed_families', W[W.family.isin(['finding_omission','hallucinated_finding','readability','laterality'])])]:
    n = len(g); both = int(((g.Q1_R1=='worse')&(g.Q1_R2=='worse')).sum()); any1 = int(((g.Q1_R1=='worse')|(g.Q1_R2=='worse')).sum())
    rows.append(dict(family=grp, n=n, both_worse=both, both_worse_ci=wilson(both,n), any_worse=any1, any_worse_ci=wilson(any1,n),
        both_no_text_change=int(((g.Q1_R1=='no text change')&(g.Q1_R2=='no text change')).sum())))
T21b = pd.DataFrame(rows); T21b.to_csv(OUT/'panel_T2_1_worse_summary.csv', index=False, encoding='utf-8-sig')

def kq1(s): return kappa(s.Q1_R1, s.Q1_R2, Q1CAT)
sets = {'primary_all': W, 'excl_any_cannot_assess': W[(W.Q1_R1!='cannot assess')&(W.Q1_R2!='cannot assess')],
        'excl_text_unchanged_pairs': W[~((W.Q1_R1=='no text change')&(W.Q1_R2=='no text change'))]}
T22 = pd.DataFrame([dict(set=k, n=len(s), kappa=kq1(s), kappa_ci=boot(s, kq1), raw_agreement=float((s.Q1_R1==s.Q1_R2).mean())) for k, s in sets.items()])
T22.to_csv(OUT/'panel_T2_2_Q1_agreement.csv', index=False, encoding='utf-8-sig')
xt = pd.crosstab(W.Q1_R1, W.Q1_R2).reindex(index=Q1CAT, columns=Q1CAT, fill_value=0); xt.to_csv(OUT/'panel_T2_2_Q1_crosstab.csv', encoding='utf-8-sig')

D = ['worse','better']; num = lambda x: x in ('1','2','3')
q2 = W[W.Q1_R1.isin(D)&W.Q1_R2.isin(D)&W.Q2_R1.map(num)&W.Q2_R2.map(num)]
fq2 = lambda s: qwk(s.Q2_R1.astype(int), s.Q2_R2.astype(int), cats=(1,2,3))
T23 = pd.DataFrame([dict(n_pairs=len(q2), same_direction=int((q2.Q1_R1==q2.Q1_R2).sum()), qwk=fq2(q2) if len(q2)>1 else np.nan, qwk_ci=boot(q2, fq2) if len(q2)>2 else (np.nan,np.nan))])
T23.to_csv(OUT/'panel_T2_3_Q2_agreement.csv', index=False, encoding='utf-8-sig')
rows = []
for grp, g in W.groupby('family'):
    for r in ('R1','R2'):
        vc = g[f'Q2_{r}'].value_counts(); rows.append(dict(family=grp, reader=r, n=len(g), **{c:int(vc.get(c,0)) for c in ['1','2','3','cannot assess','not applicable']}))
pd.DataFrame(rows).to_csv(OUT/'panel_T2_3_Q2_distribution.csv', index=False, encoding='utf-8-sig')

rows = []
for r in ('R1','R2'):
    d = W[W[f'Q1_{r}'].isin(D) & W[f'Q2_{r}'].map(num) & ~W.family.isin(['laterality','noop'])]
    rho, p = spearmanr(d.nominal_intensity.astype(int), d[f'Q2_{r}'].astype(int)) if len(d)>2 else (np.nan, np.nan)
    rows.append(dict(reader=r, n=len(d), spearman_rho=rho, p_descriptive=p))
pd.DataFrame(rows).to_csv(OUT/'panel_T2_4_intensity_vs_Q2.csv', index=False, encoding='utf-8-sig')

rows = []
for grp, g in W.groupby('family'):
    for r in ('R1','R2'):
        vc = g[f'Q3_{r}'].value_counts(); rows.append(dict(family=grp, reader=r, n=len(g), **{c:int(vc.get(c,0)) for c in ['yes','no','cannot assess','not applicable']}))
pd.DataFrame(rows).to_csv(OUT/'panel_T2_5_Q3_distribution.csv', index=False, encoding='utf-8-sig')
q3 = W[(W.Q3_R1!='not applicable')&(W.Q3_R2!='not applicable')]; fq3 = lambda s: kappa(s.Q3_R1, s.Q3_R2, ['yes','no','cannot assess'])
T25 = pd.DataFrame([dict(n=len(q3), kappa=fq3(q3), kappa_ci=boot(q3, fq3), raw_agreement=float((q3.Q3_R1==q3.Q3_R2).mean()))])
T25.to_csv(OUT/'panel_T2_5_Q3_agreement.csv', index=False, encoding='utf-8-sig')

lat = t2[t2.family=='laterality'][['code','reader','Q1','Q2','Q3']].copy()
lat['text_detectable_posthoc'] = lat.code.map({'P997':'yes (Findings–Impression laterality conflict)','P838':'yes (left middle lobe: nonexistent)'}).fillna('no (single mention)')
lat.sort_values(['code','reader']).to_csv(OUT/'panel_T2_6_laterality_posthoc.csv', index=False, encoding='utf-8-sig')
json.dump(INPUTS, open(OUT/'panel_inputs_sha256.json','w'), indent=1)
print('DONE'); print(json.dumps(INPUTS, indent=1))
