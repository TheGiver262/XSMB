#!/usr/bin/env python3
"""Full-history predictive research and feature screening for XSMB."""
from __future__ import annotations
import csv, math, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from xsmb_probability import load_draws as load2, matrices as matrices2
from xsmb_3digit import SPECS, matrices as matrices3
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data'/'upstream'/'xsmb.csv'; OUT=ROOT/'research'
@dataclass(frozen=True)
class RSpec:
    name:str; universe:int; baseline:float; short:int; long:int; prior:float; weekday_prior:float; gap_cap:int
S2=RSpec('two_digit',100,1-(99/100)**27,30,90,100.,30.,30)
S3A=RSpec('suffix3_any',1000,SPECS['suffix3_any'].baseline,60,180,500.,140.,180)
SG6=RSpec('g6_exact',1000,SPECS['g6_exact'].baseline,120,365,900.,210.,365)
RECIPES={'long_only':{'long':1.0},'recency':{'long':.50,'short':.30,'recent_long':.20},'weekday':{'long':.45,'short':.25,'recent_long':.15,'weekday':.15},'gap':{'long':.40,'short':.30,'recent_long':.15,'weekday':.10,'gap':.05}}
def norm_p(z):return math.erfc(abs(z)/math.sqrt(2))
def bh(p):
    m=len(p); order=sorted(range(m),key=lambda i:p[i]); q=[1.]*m; run=1.
    for r0 in range(m-1,-1,-1):
        i=order[r0]; run=min(run,p[i]*m/(r0+1)); q[i]=min(1.,run)
    return q
def recipe(f,name,u):
    p=np.zeros(u)
    for k,w in RECIPES[name].items():p+=w*f[k]
    return np.clip(p,1e-12,1-1e-12)
def brier(y,p):return float(np.mean((y-p)**2))
def stats(name,spec,dates,presence,counts):
    n=len(dates); observed=presence.sum(0); rate=observed/n; den=math.sqrt(spec.baseline*(1-spec.baseline)/n); z=(rate-spec.baseline)/den; pv=[norm_p(float(v)) for v in z]; q=bh(pv); rows=[]
    for num in range(spec.universe):
        hits=np.flatnonzero(presence[:,num]); gap=(n-1-int(hits[-1])) if len(hits) else n
        rr=lambda w:float(presence[max(0,n-w):,num].mean())
        rows.append({'number':f'{num:0{2 if spec.universe==100 else 3}d}','draws':n,'occurrences':int(counts[:,num].sum()),'draws_hit':int(observed[num]),'full_rate':float(rate[num]),'baseline':spec.baseline,'z_score':float(z[num]),'p_value_normal':pv[num],'bh_q_value':q[num],'rate_30':rr(30),'rate_90':rr(90),'rate_365':rr(365),'rate_1095':rr(1095),'current_gap_draws':gap})
    OUT.mkdir(exist_ok=True); p=OUT/f'{name}_number_stats.csv'
    with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def walk(name,spec,dates,presence):
    years=sorted(set(d.year for d in dates))[-6:]; first=years[0]; start=next((i for i,d in enumerate(dates) if d.year>=first),max(365,len(dates)-1825)); start=max(start,365); n=len(dates); u=spec.universe
    prefix=np.zeros((n+1,u),dtype=np.int32); np.cumsum(presence,axis=0,dtype=np.int32,out=prefix[1:])
    last=np.full(u,-1,dtype=np.int32); weekday_hits=np.zeros((7,u),dtype=np.int32); weekday_counts=np.zeros(7,dtype=np.int32)
    for k in range(start):
        hit=np.flatnonzero(presence[k]); last[hit]=k; wd=dates[k].weekday(); weekday_hits[wd]+=presence[k]; weekday_counts[wd]+=1
    by={}; scores={r:[] for r in RECIPES}; base=[]; p0=spec.baseline
    for i in range(start,n):
        wd=dates[i].weekday(); long=(prefix[i]+spec.prior*p0)/(i+spec.prior)
        ss=max(0,i-spec.short); ls=max(0,i-spec.long); short_hits=prefix[i]-prefix[ss]; long_hits=prefix[i]-prefix[ls]
        short=(short_hits+spec.short*p0)/((i-ss)+spec.short); recent=(long_hits+spec.long*p0)/((i-ls)+spec.long)
        wr=(weekday_hits[wd]+spec.weekday_prior*p0)/(weekday_counts[wd]+spec.weekday_prior)
        gaps=np.where(last>=0,i-1-last,i); centered=np.clip(gaps/spec.gap_cap,0,1)-.5; gap=np.clip(p0*(1+.12*centered),p0*.5,min(.999,p0*1.5)); f={'long':long,'short':short,'recent_long':recent,'weekday':wr,'gap':gap}
        y=presence[i].astype(float); bb=brier(y,np.full(u,p0)); rec={'baseline':bb}; base.append(bb)
        for r in RECIPES:rec[r]=brier(y,recipe(f,r,u)); scores[r].append(rec[r])
        by.setdefault(dates[i].year,[]).append(rec); hit=np.flatnonzero(presence[i]); last[hit]=i; weekday_hits[wd]+=presence[i]; weekday_counts[wd]+=1
    yr=[]
    for year,recs in sorted(by.items()):
        bb=sum(x['baseline'] for x in recs)/len(recs)
        for r in RECIPES:
            mb=sum(x[r] for x in recs)/len(recs); yr.append({'target':name,'year':year,'recipe':r,'draws':len(recs),'model_brier':mb,'baseline_brier':bb,'brier_improvement':bb-mb})
    overall=[]; bb=sum(base)/len(base)
    for r in RECIPES:
        mb=sum(scores[r])/len(scores[r]); ys=[x for x in yr if x['recipe']==r]; wins=sum(float(x['brier_improvement'])>0 for x in ys); need=max(1,math.ceil(.60*len(ys))); overall.append({'target':name,'year':'ALL','recipe':r,'draws':len(base),'model_brier':mb,'baseline_brier':bb,'brier_improvement':bb-mb,'positive_years':wins,'required_positive_years':need,'approved':bool(bb-mb>0 and wins>=need)})
    return yr,overall
def main():
    if not RAW.exists():raise SystemExit('Run src/sync_upstream.py first')
    draws=load2(RAW); dates=[d['date'] for d in draws]; p2,c2=matrices2(draws); p3,c3=matrices3(draws,SPECS['suffix3_any']); pg,cg=matrices3(draws,SPECS['g6_exact']); targets=[('two_digit',S2,p2,c2),('suffix3_any',S3A,p3,c3),('g6_exact',SG6,pg,cg)]; ally=[]; allo=[]
    for name,spec,p,c in targets:print('research',name); stats(name,spec,dates,p,c); y,o=walk(name,spec,dates,p); ally+=y; allo+=o
    fields=sorted(set().union(*(r.keys() for r in ally+allo))); OUT.mkdir(exist_ok=True)
    with (OUT/'walk_forward_summary.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(ally+allo)
    approved={t:[r['recipe'] for r in allo if r['target']==t and r['approved']] for t in ['two_digit','suffix3_any','g6_exact']}
    approved_2d=approved['two_digit']
    sets=[]
    if 'long_only' in approved_2d:sets.append('long_only')
    if 'recency' in approved_2d:sets.append('recency')
    if 'gap' in approved_2d:sets.append('recency_gap')
    if 'weekday' in approved_2d and 'gap' in approved_2d:sets.append('full')
    # Every live model needs a numerical candidate so validation can estimate a
    # blend. long_only is a technical fallback, NOT an approved predictive edge.
    if not sets:sets=['long_only']
    allowed_3d={t:(approved[t] if approved[t] else ['long_only']) for t in ['suffix3_any','g6_exact']}
    gate={
        'generated_from':'full-history walk-forward recipe screening',
        'criteria':'overall Brier improvement > 0 and positive in >=60% of year folds',
        'fallback_policy':'If nothing is approved, long_only remains as a technical candidate and live calibration may set blend=0 to return exactly to baseline.',
        'two_digit':{
            'allowed_feature_sets':sets,
            'approved_signal_recipes':approved_2d,
            'fallback_feature_set':'long_only' if not approved_2d else None,
        },
        'suffix3_any':{
            'allowed_recipes':allowed_3d['suffix3_any'],
            'approved_recipes':approved['suffix3_any'],
            'fallback_recipe':'long_only' if not approved['suffix3_any'] else None,
        },
        'g6_exact':{
            'allowed_recipes':allowed_3d['g6_exact'],
            'approved_recipes':approved['g6_exact'],
            'fallback_recipe':'long_only' if not approved['g6_exact'] else None,
        },
    }
    (OUT/'feature_gate.json').write_text(json.dumps(gate,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); print(json.dumps(gate,indent=2))
if __name__=='__main__':main()
