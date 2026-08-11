#!/usr/bin/env python3
from __future__ import annotations
import argparse, datetime as dt, json
from pathlib import Path
import numpy as np
from xsmb_probability import FEATURE_NAMES, P0, brier, build_dataset, features_for_day, fit_logistic_ridge, load_draws, log_loss, matrices, predict, write_prediction
ROOT=Path(__file__).resolve().parents[1]
FEATURE_SETS={'long_only':[4],'recency':[0,1,2,3,4],'recency_gap':[0,1,2,3,4,6,7],'full':list(range(len(FEATURE_NAMES)))}

def allowed_feature_sets():
    gate=ROOT/'research'/'feature_gate.json'
    if gate.exists():
        try:
            data=json.loads(gate.read_text(encoding='utf-8')); names=data.get('two_digit',{}).get('allowed_feature_sets',[])
            names=[n for n in names if n in FEATURE_SETS]
            if names:return names
        except Exception:pass
    return list(FEATURE_SETS)

def select_hyperparams(draws,presence,counts,warmup=90):
    n=len(draws)
    if n<240: raise ValueError('Need at least 240 observed draws for validation.')
    x_all,y_all=build_dataset(warmup,n,draws,presence,counts)
    sl=lambda a,b:slice((a-warmup)*100,(b-warmup)*100)
    c1=max(warmup+60,n-181); c2=max(c1+40,n-121); c3=max(c2+40,n-61); folds=[(c1,c2),(c2,c3),(c3,n)]
    best=None; actual_best=None
    for set_name,idx in FEATURE_SETS.items():
        if set_name not in allowed_feature_sets(): continue
        for l2 in [10.,100.,300.,1000.,3000.]:
            pp=[]; yy=[]
            for vs,ve in folds:
                tx=x_all[sl(warmup,vs)][:,idx]; ty=y_all[sl(warmup,vs)]; vx=x_all[sl(vs,ve)][:,idx]; vy=y_all[sl(vs,ve)]
                m=fit_logistic_ridge(tx,ty,l2=l2); pp.append(predict(m,vx)); yy.append(vy)
            pred=np.concatenate(pp); actual=np.concatenate(yy)
            for blend in np.linspace(0,1,101):
                cal=P0+blend*(pred-P0); cand=(brier(actual,cal),log_loss(actual,cal),set_name,l2,float(blend))
                if best is None or cand[:2]<best[:2]: best=cand; actual_best=actual
    base=np.full_like(actual_best,P0,dtype=float)
    metrics={'cv_brier':best[0],'cv_log_loss':best[1],'baseline_brier':brier(actual_best,base),'baseline_log_loss':log_loss(actual_best,base),'brier_improvement':brier(actual_best,base)-best[0]}
    return best[2],best[3],best[4],metrics

def next_probabilities(draws,target_date):
    presence,counts=matrices(draws); n=len(draws); warmup=90
    fs,l2,blend,metrics=select_hyperparams(draws,presence,counts,warmup); idx=FEATURE_SETS[fs]
    x_all,y_all=build_dataset(warmup,n,draws,presence,counts); model=fit_logistic_ridge(x_all[:,idx],y_all,l2=l2)
    x_next=features_for_day(n,target_date.weekday(),draws,presence,counts); raw=predict(model,x_next[:,idx]); final=P0+blend*(raw-P0)
    rows=[]
    for number in range(100):
        f=x_next[number]; rows.append({'number':f'{number:02d}','probability':float(final[number]),'raw_model_probability':float(raw[number]),**{name:float(f[i]) for i,name in enumerate(FEATURE_NAMES)}})
    rows.sort(key=lambda r:(-r['probability'],r['number']))
    metrics.update({'theoretical_baseline':P0,'selected_feature_set':fs,'selected_l2':l2,'selected_blend':blend,'observed_draws':n,'target_date':target_date.isoformat()})
    return rows,metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--target-date',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    rows,m=next_probabilities(load_draws(a.data),dt.date.fromisoformat(a.target_date)); write_prediction(rows,m,a.out)
    print(f"feature_set={m['selected_feature_set']} blend={m['selected_blend']:.2f}")
if __name__=='__main__':main()
