#!/usr/bin/env python3
"""Live pre-draw forecasts and post-draw evaluation for XSMB 3-digit targets."""
from __future__ import annotations
import argparse, csv, datetime as dt, json, math
from daily_pipeline import ROOT, append_unique_csv, fetch_upstream, local_today, read_csv, rows_between, sha256_file, utc_now_iso, write_raw_csv
from xsmb_3digit import SPECS, load_draws, next_probabilities, write_prediction
WINDOW_DAYS=1095

def generate(history,target_date,target,out_path,metadata):
    runtime=ROOT/'.runtime';runtime.mkdir(exist_ok=True);temp=runtime/f'3d_{target}_{target_date}.csv';write_raw_csv(temp,history)
    try:
        draws=load_draws(temp);rows,metrics=next_probabilities(draws,target_date,target);metrics.update(metadata);write_prediction(rows,metrics,out_path);return rows,metrics
    finally:temp.unlink(missing_ok=True)

def forecast(target_date):
    upstream=fetch_upstream()
    if any(r['date']==target_date.isoformat() for r in upstream):raise RuntimeError(f'Draw {target_date} already exists upstream; refusing retrospective 3-digit forecast')
    cutoff=target_date-dt.timedelta(days=1);start=target_date-dt.timedelta(days=WINDOW_DAYS);hist=rows_between(upstream,start,cutoff)
    if len(hist)<540:raise RuntimeError(f'Only {len(hist)} historical draws')
    for target in SPECS:
        directory=ROOT/'forecasts_3d'/target;out=directory/f'{target_date}.csv'
        if out.exists():print(f'Immutable forecast already exists: {out}');continue
        rows,m=generate(hist,target_date,target,out,{'forecast_kind':'live_pre_draw','generated_at_utc':utc_now_iso(),'data_cutoff_date':cutoff.isoformat(),'history_calendar_start':start.isoformat(),'history_calendar_end':cutoff.isoformat()})
        mp=out.with_name(out.stem+'_metrics.csv');manifest=directory/'manifest.csv';rec={'target_date':target_date.isoformat(),'generated_at_utc':m['generated_at_utc'],'data_cutoff_date':cutoff.isoformat(),'observed_draws':m['observed_draws'],'selected_recipe':m['selected_recipe'],'selected_blend':m['selected_blend'],'forecast_sha256':sha256_file(out),'metrics_sha256':sha256_file(mp)};append_unique_csv(manifest,rec,'target_date',list(rec));print(f'{target}: recipe={m["selected_recipe"]} blend={m["selected_blend"]:.2f} top={",".join(r["number"] for r in rows[:10])}')
    return 0

def actual(draw,target):
    return {str(draw[c]).zfill(3)[-3:] for c in SPECS[target].cols}
def ll(y,p):p=min(max(p,1e-12),1-1e-12);return -(y*math.log(p)+(1-y)*math.log(1-p))
def evaluate(rows,draw,target):
    spec=SPECS[target];truth=actual(draw,target)
    if len(rows)!=1000:raise ValueError(f'Expected 1000 rows for {target}')
    ordered=sorted(rows,key=lambda r:(-float(r['probability']),int(r['number'])));se=[];los=[];bse=[];bll=[]
    for r in ordered:
        num=f"{int(r['number']):03d}";y=int(num in truth);p=float(r['probability']);se.append((y-p)**2);los.append(ll(y,p));bse.append((y-spec.baseline)**2);bll.append(ll(y,spec.baseline))
    mb=sum(se)/1000;bb=sum(bse)/1000;ml=sum(los)/1000;bl=sum(bll)/1000
    return {'brier':mb,'baseline_brier':bb,'brier_improvement':bb-mb,'log_loss':ml,'baseline_log_loss':bl,'log_loss_improvement':bl-ml,'actual_unique_numbers':len(truth),'top20_hits':sum(f"{int(r['number']):03d}" in truth for r in ordered[:20]),'top50_hits':sum(f"{int(r['number']):03d}" in truth for r in ordered[:50]),'top100_hits':sum(f"{int(r['number']):03d}" in truth for r in ordered[:100])}
def append_composite(path,row,fields,key_fields):
    existing=read_csv(path)
    if any(all(r.get(k)==str(row.get(k,'')) for k in key_fields) for r in existing):return False
    path.parent.mkdir(parents=True,exist_ok=True);wh=not path.exists() or path.stat().st_size==0
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');
        if wh:w.writeheader()
        w.writerow({k:row.get(k,'') for k in fields})
    return True
def rolling_summary():
    src=read_csv(ROOT/'evaluation'/'3d_daily_metrics.csv');fields=['target','window','evaluated_draws','model_brier','baseline_brier','brier_improvement','model_log_loss','baseline_log_loss','log_loss_improvement','mean_top20_hits','mean_top50_hits','mean_top100_hits'];out=[]
    for target in SPECS:
        vals=[r for r in src if r.get('target')==target and r.get('status')=='evaluated']
        for window in (30,60,90):
            s=vals[-window:]
            if not s:out.append({'target':target,'window':window,'evaluated_draws':0});continue
            avg=lambda k:sum(float(r[k]) for r in s)/len(s);mb=avg('brier');bb=avg('baseline_brier');ml=avg('log_loss');bl=avg('baseline_log_loss');out.append({'target':target,'window':window,'evaluated_draws':len(s),'model_brier':mb,'baseline_brier':bb,'brier_improvement':bb-mb,'model_log_loss':ml,'baseline_log_loss':bl,'log_loss_improvement':bl-ml,'mean_top20_hits':avg('top20_hits'),'mean_top50_hits':avg('top50_hits'),'mean_top100_hits':avg('top100_hits')})
    p=ROOT/'evaluation'/'3d_rolling_summary.csv';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(out)
def settle(day):
    statep=ROOT/'output'/'pipeline_3d_state.json'
    if statep.exists():
        try:state=json.loads(statep.read_text())
        except Exception:state={}
        if state.get('last_settled_draw')==day.isoformat():print(f'3-digit {day} already settled');return 0
    upstream=fetch_upstream();draw={r['date']:r for r in upstream}.get(day.isoformat())
    if draw is None:print(f'No draw for {day}');return 0
    ep=ROOT/'evaluation'/'3d_daily_metrics.csv';ef=['date','target','status','brier','baseline_brier','brier_improvement','log_loss','baseline_log_loss','log_loss_improvement','actual_unique_numbers','top20_hits','top50_hits','top100_hits','forecast_sha256']
    for target in SPECS:
        fp=ROOT/'forecasts_3d'/target/f'{day}.csv';row={'date':day.isoformat(),'target':target,'status':'missing_forecast','forecast_sha256':''}
        if fp.exists():row={'date':day.isoformat(),'target':target,'status':'evaluated',**evaluate(read_csv(fp),draw,target),'forecast_sha256':sha256_file(fp)}
        append_composite(ep,row,ef,['date','target'])
    start=day-dt.timedelta(days=WINDOW_DAYS-1);hist=rows_between(upstream,start,day);next_day=day+dt.timedelta(days=1);generated=utc_now_iso();rf=['as_of_date','target_date','target','observed_draws','selected_recipe','selected_blend','cv_brier','baseline_brier','brier_improvement','generated_at_utc']
    for target in SPECS:
        out=ROOT/'output'/f'next_preview_{target}.csv';rows,m=generate(hist,next_day,target,out,{'forecast_kind':'post_draw_preview','generated_at_utc':generated,'data_cutoff_date':day.isoformat(),'history_calendar_start':start.isoformat(),'history_calendar_end':day.isoformat()});rec={'as_of_date':day.isoformat(),'target_date':next_day.isoformat(),'target':target,'observed_draws':m['observed_draws'],'selected_recipe':m['selected_recipe'],'selected_blend':m['selected_blend'],'cv_brier':m['cv_brier'],'baseline_brier':m['baseline_brier'],'brier_improvement':m['brier_improvement'],'generated_at_utc':generated};append_composite(ROOT/'evaluation'/'3d_model_runs.csv',rec,rf,['as_of_date','target']);print(f'preview {target}: {m["selected_recipe"]} blend={m["selected_blend"]:.2f} top={",".join(r["number"] for r in rows[:10])}')
    rolling_summary();statep.write_text(json.dumps({'last_settled_draw':day.isoformat(),'next_preview_target':next_day.isoformat(),'observed_draws_in_window':len(hist),'updated_at_utc':generated},indent=2)+'\n');return 0
def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True);a=sub.add_parser('forecast');a.add_argument('--target-date');b=sub.add_parser('settle');b.add_argument('--date');args=p.parse_args();return forecast(dt.date.fromisoformat(args.target_date) if args.target_date else local_today()) if args.cmd=='forecast' else settle(dt.date.fromisoformat(args.date) if args.date else local_today())
if __name__=='__main__':raise SystemExit(main())
