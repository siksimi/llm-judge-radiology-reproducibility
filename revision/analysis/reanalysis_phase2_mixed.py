"""reanalysis_phase2_mixed.py — R1-12 per SAP v1 §4 (statsmodels MixedLM, REML).
C2 (Gemini, Qwen): y ~ C(temperature) + (1|report) + (1|report:temperature)
C3 (all judges):   y ~ C(variant)     + (1|report) + (1|report:variant)
Residual = within report×condition (run-to-run) variance. Then exploratory model of cell run-SD."""
import os, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,'out')
df=pd.read_parquet(os.path.join(HERE,'input','scores_rebuilt_32000.parquet')); st=pd.read_csv(os.path.join(HERE,'input','report_status.csv'))
LAB={'claude-opus-4-8':'Claude Opus 4.8','gpt-5.5':'GPT-5.5','gemini-2.5-pro':'Gemini 2.5 Pro','qwen-3.5':'Qwen 3.5'}
y='overall_global_quality'; rows=[]; fx=[]
def fit(d, cond_col, label, judge):
    d=d.dropna(subset=[y]).copy(); d['cond']=d[cond_col].astype(str)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        m=smf.mixedlm(f'{y} ~ C(cond)', d, groups='report_id', re_formula='1', vc_formula={'report_x_cond':'0 + C(cond)'})
        r=m.fit(reml=True, method=['lbfgs','powell'])
        msgs='; '.join(sorted({str(x.message)[:80] for x in w}))
    v_rep=float(r.cov_re.iloc[0,0]); v_rc=float(r.vcomp[0]); v_res=float(r.scale); tot=v_rep+v_rc+v_res
    rows.append(dict(judge=LAB[judge], condition=label, n_obs=len(d), var_report=v_rep, var_report_x_condition=v_rc, var_residual_run=v_res,
                     pct_report=100*v_rep/tot, pct_report_x_condition=100*v_rc/tot, pct_residual_run=100*v_res/tot,
                     converged=bool(r.converged), boundary_flag=bool(min(v_rep,v_rc)<1e-6), warnings=msgs))
    ci=r.conf_int()
    for nm in r.fe_params.index:
        fx.append(dict(judge=LAB[judge], condition=label, term=nm, estimate=r.fe_params[nm], lo=ci.loc[nm,0], hi=ci.loc[nm,1], p=r.pvalues[nm]))
for j in ['gemini-2.5-pro','qwen-3.5']:
    fit(df[(df.judge_id==j)&(df.condition=='C2')], 'temperature', 'C2_temperature', j)
for j in LAB:
    d=df[(df.judge_id==j)&(df.condition=='C3')]
    if d.groupby(['report_id','variant'])[y].std().max()==0:
        # deterministic judge: run-to-run variance is exactly 0, so the 3-component model is not identified.
        # Fit on cell values: y ~ C(variant) + (1|report); residual here = report x variant interaction.
        c=d.groupby(['report_id','variant'])[y].first().reset_index(); c['cond']=c.variant
        r=smf.mixedlm(f'{y} ~ C(cond)', c, groups='report_id').fit(reml=True)
        v_rep=float(r.cov_re.iloc[0,0]); v_rc=float(r.scale); tot=v_rep+v_rc
        rows.append(dict(judge=LAB[j], condition='C3_prompt', n_obs=len(d), var_report=v_rep, var_report_x_condition=v_rc, var_residual_run=0.0,
                         pct_report=100*v_rep/tot, pct_report_x_condition=100*v_rc/tot, pct_residual_run=0.0, converged=bool(r.converged),
                         boundary_flag=False, warnings='deterministic: run variance 0; fitted on report x variant cell values (2-component model)'))
        ci=r.conf_int()
        for nm in r.fe_params.index:
            fx.append(dict(judge=LAB[j], condition='C3_prompt', term=nm, estimate=r.fe_params[nm], lo=ci.loc[nm,0], hi=ci.loc[nm,1], p=r.pvalues[nm]))
        continue
    fit(d, 'variant', 'C3_prompt', j)
pd.DataFrame(rows).to_csv(os.path.join(OUT,'r1_12_mixed_variance_components.csv'), index=False)
pd.DataFrame(fx).to_csv(os.path.join(OUT,'r1_12_mixed_fixed_effects.csv'), index=False)
# ---- exploratory: what predicts run-to-run SD within a report x condition cell (C2/C3)? baseline from C1 (separate data) ----
base=df[df.condition=='C1'].groupby(['judge_id','report_id'])[y].mean().rename('baseline').reset_index()
cells=df[df.condition.isin(['C2','C3'])].groupby(['judge_id','condition','variant','temperature','report_id'])[y].std(ddof=1).rename('cell_sd').reset_index()
cells=cells.merge(base,on=['judge_id','report_id']).merge(st[['report_id','n_words','group3','assigned_family','nominal_intensity']],on='report_id')
cells['boundary_dist']=(cells.baseline-(np.floor(cells.baseline)+0.5)).abs()
er=[]
for j in LAB:
    d=cells[cells.judge_id==j].dropna(subset=['cell_sd'])
    # drop condition cells with no run variance by design (e.g., deterministic T=0 / Qwen C3)
    keep=d.groupby(['condition','temperature']).cell_sd.transform('max')>0; d=d[keep]
    if d.cell_sd.var()==0: er.append(dict(judge=LAB[j], note='no run-to-run variance in C2/C3 cells')); continue
    preds=[x for x in ['boundary_dist','baseline'] if d[x].var()>1e-12]   # Qwen C1 baseline is integer -> boundary_dist constant
    cond = ' + C(condition)' if d.condition.nunique()>1 else ''
    for name, form in [('M1','cell_sd ~ '+' + '.join(preds+['np.log(n_words)'])+cond),
                       ('M2','cell_sd ~ '+' + '.join(preds+['np.log(n_words)'])+cond+' + C(group3)')]:
        r=smf.ols(form,d).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d.report_id)[0]})
        for t in r.params.index:
            er.append(dict(judge=LAB[j], model=name, term=t, estimate=r.params[t], lo=r.conf_int().loc[t,0], hi=r.conf_int().loc[t,1], p=r.pvalues[t], n_cells=int(r.nobs), cond_number=float(np.linalg.cond(r.model.exog))))
pd.DataFrame(er).to_csv(os.path.join(OUT,'r1_12_residual_exploration.csv'), index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
