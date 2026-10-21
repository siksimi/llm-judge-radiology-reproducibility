"""campaign_analysis.py — C-F and C-Q per SAP_appendix_CF_CQ_v1.md (frozen 2026-10-06, before calls).
Inputs: campaigns/runs_CF2026-10/scores_7e8dcfc9f60d6dcb_<judge>.parquet, campaigns/runs_CQ2026-10/scores_9c318fd2c50d28a0.parquet,
        input/scores_rebuilt_32000.parquet (June C1/C3), input/report_status.csv. Outputs: out/cf_*.csv, out/cq_*.csv."""
import os, glob, hashlib, json, numpy as np, pandas as pd
from scipy.stats import rankdata
H=os.path.dirname(os.path.abspath(__file__)); C=os.path.join(H,'..','campaigns'); OUT=os.path.join(H,'out')
B=2000; SEED=20261020
LAB={'claude-opus-4-8':'Claude Opus 4.8','gpt-5.5':'GPT-5.5','gemini-2.5-pro':'Gemini 2.5 Pro','qwen-3.5':'Qwen 3.5'}
st=pd.read_csv(os.path.join(H,'input','report_status.csv'))
SETS={'A_all100':st.report_id.tolist(),'B_88':st[st.group3!='failed_perturbation'].report_id.tolist()}
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
def boot(fn, ids, seed):
    rng=np.random.default_rng([SEED,seed]); ids=np.asarray(ids); v=[]
    for _ in range(B): v.append(fn(rng.choice(ids,len(ids),replace=True)))
    v=np.array(v,float); return np.nanpercentile(v,2.5,axis=0), np.nanpercentile(v,97.5,axis=0)
def kendall_w(R):
    n,m=R.shape; ranks=np.column_stack([rankdata(R[:,c]) for c in range(m)])
    S=((ranks.sum(1)-m*(n+1)/2)**2).sum()
    T=sum(((np.unique(R[:,c],return_counts=True)[1]**3-np.unique(R[:,c],return_counts=True)[1]).sum()) for c in range(m))
    return 12*S/(m**2*(n**3-n)-m*T)
inputs={}
# ---------------- C-F ----------------
files=sorted(glob.glob(os.path.join(C,'runs_CF2026-10','scores_7e8dcfc9f60d6dcb_*.parquet')))
files=[f for f in files if 'checkpoint' not in f]
cf=pd.concat([pd.read_parquet(f) for f in files]); inputs.update({os.path.basename(f):sha(f) for f in files})
cf['valid']=cf.parse_ok.astype(bool)&cf.api_error.isna()
cell=cf[cf.valid].groupby(['judge_id','report_id','prompt']).overall_global_quality.agg(['mean','count']).reset_index()
cell.loc[cell['count']<3,'mean']=np.nan   # SAP: cell valid only if >=3/5 valid runs
M={j:g.pivot(index='report_id',columns='prompt',values='mean') for j,g in cell.groupby('judge_id')}
miss=cf.groupby(['judge_id','prompt']).valid.agg(lambda s:int((~s).sum())).reset_index(name='invalid_runs')
miss.to_csv(os.path.join(OUT,'cf_missing.csv'),index=False)
CON=[('notes: v5_as_sent - v5_clean','v5_as_sent','v5_clean'),('example set: v5_clean - v5prime','v5_clean','v5prime'),
     ('v5_as_sent - v1','v5_as_sent','v1'),('v5_clean - v1','v5_clean','v1'),('v5prime - v1','v5prime','v1')]
rows=[]; sc=iter(range(1,10**6))
for j in LAB:
    for sname,ids in SETS.items():
        X=M[j].reindex(ids)
        for lab,a,b in CON:
            d=(X[a]-X[b]).dropna(); idx=d.index.tolist(); xa=X.loc[idx,a]; xb=X.loc[idx,b]
            ra=np.round(xa); rb=np.round(xb)
            def f(s, X=X, a=a, b=b):
                dd=(X.loc[s,a]-X.loc[s,b]).to_numpy(float); ra=np.round(X.loc[s,a].to_numpy(float)); rb=np.round(X.loc[s,b].to_numpy(float))
                return [np.nanmean(dd), np.nanmean(np.abs(ra-rb)>=1), np.nanmean(np.abs(dd)>=1)]
            est=f(idx); lo,hi=boot(f,idx,next(sc))
            rows.append(dict(judge=LAB[j],set=sname,contrast=lab,n_reports=len(idx),mean_shift=est[0],shift_lo=lo[0],shift_hi=hi[0],
                prop_rounded_change_ge1=est[1],rounded_lo=lo[1],rounded_hi=hi[1],prop_unrounded_absdiff_ge1=est[2],unrounded_lo=lo[2],unrounded_hi=hi[2]))
        Y=X[['v1','v5_as_sent','v5_clean','v5prime']].dropna().to_numpy(float)
        w=kendall_w(Y); wlo,whi=boot(lambda s,X=X: kendall_w(X.loc[s,['v1','v5_as_sent','v5_clean','v5prime']].dropna().to_numpy(float)), X.index.tolist(), next(sc))
        rows.append(dict(judge=LAB[j],set=sname,contrast='Kendall W (4 prompts, tie-corrected)',n_reports=len(Y),mean_shift=w,shift_lo=wlo,shift_hi=whi))
pd.DataFrame(rows).to_csv(os.path.join(OUT,'cf_contrasts.csv'),index=False)
# historical June C3 v5 - v1 (descriptive only)
jn=pd.read_parquet(os.path.join(H,'input','scores_rebuilt_32000.parquet')); inputs['scores_rebuilt_32000.parquet']=sha(os.path.join(H,'input','scores_rebuilt_32000.parquet'))
c3=jn[(jn.condition=='C3')&jn.parse_ok.astype(bool)].groupby(['judge_id','report_id','variant']).overall_global_quality.mean().unstack()
hist=[]
for j in LAB:
    g=c3.loc[j]; d=(g['v5']-g['v1']); hist.append(dict(judge=LAB[j],june_v5_minus_v1=d.mean(),oct_v5_as_sent_minus_v1=(M[j]['v5_as_sent']-M[j]['v1']).mean(),
        june_v1_mean=g['v1'].mean(),oct_v1_mean=M[j]['v1'].mean()))
pd.DataFrame(hist).to_csv(os.path.join(OUT,'cf_historical_june_vs_oct.csv'),index=False)
# ---------------- C-Q ----------------
qf=os.path.join(C,'runs_CQ2026-10','scores_9c318fd2c50d28a0.parquet'); inputs[os.path.basename(qf)]=sha(qf)
q=pd.read_parquet(qf); q['valid']=q.parse_ok.astype(bool)&q.api_error.isna()
june=jn[(jn.judge_id=='qwen-3.5')&(jn.condition=='C1')&jn.parse_ok.astype(bool)]
jv=june.groupby('report_id').overall_global_quality.agg(['min','max','first'])
assert (jv['min']==jv['max']).all(), 'June Qwen C1 not deterministic'
J=jv['first']
qq=q[q.valid].groupby('report_id').overall_global_quality.agg(list)
cnt=q[q.valid].groupby('report_id').size()
inc=cnt[cnt>=8].index.tolist()
res=[]
for sname,ids in SETS.items():
    ids=[r for r in ids if r in inc]
    def g(s):
        s=list(s); allsame=np.array([len(set(qq[r]))==1 for r in s]); eq=np.array([allsame[i] and qq[r][0]==J[r] for i,r in enumerate(s)])
        mad=np.array([abs(np.mean(qq[r])-J[r]) for r in s]); return [eq.mean(), mad.mean(), allsame.mean()]
    est=g(ids); lo,hi=boot(g,ids,900+len(ids))
    allsame=[len(set(qq[r]))==1 for r in ids]
    res.append(dict(set=sname,n_reports_included=len(ids),n_excluded=len([r for r in SETS[sname] if r not in inc]),
        exact_agreement=est[0],ea_lo=lo[0],ea_hi=hi[0],mean_abs_diff_from_june=est[1],mad_lo=lo[1],mad_hi=hi[1],
        prop_oct_runs_identical=est[2],pi_lo=lo[2],pi_hi=hi[2],n_oct_internally_variable=int(len(ids)-sum(allsame)),
        n_identical_but_differs_from_june=int(sum(1 for r in ids if len(set(qq[r]))==1 and qq[r][0]!=J[r]))))
pd.DataFrame(res).to_csv(os.path.join(OUT,'cq_june_vs_oct.csv'),index=False)
json.dump(inputs,open(os.path.join(OUT,'campaign_inputs_sha256.json'),'w'),indent=1)
print('done')
