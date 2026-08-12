#!/usr/bin/env python3
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from xsmb_probability import P0, PRIZE_COLS, load_draws, matrices

def gap_streak(s):
    mg=cg=ms=cs=0
    for h in s.astype(int):
        if h:mg=max(mg,cg);cg=0;cs+=1;ms=max(ms,cs)
        else:cg+=1;cs=0
    return max(mg,cg),ms
def current_gap(s):
    h=np.flatnonzero(s);return int(len(s)-1-h[-1]) if len(h) else len(s)
def main():
    root=Path(__file__).resolve().parents[1];draws=load_draws(root/'data'/'parts');presence,counts=matrices(draws);dates=[d['date'] for d in draws];n=len(draws)
    special=np.zeros((n,100),dtype=np.int8)
    for i,d in enumerate(draws):special[i,int(d['special'][-2:])]=1
    rows=[]
    for num in range(100):
        mg,ms=gap_streak(presence[:,num]);rr=lambda w:float(presence[max(0,n-w):,num].mean());rows.append({'number':f'{num:02d}','draw_presence_3y':int(presence[:,num].sum()),'presence_rate_3y':float(presence[:,num].mean()),'occurrences_3y':int(counts[:,num].sum()),'occurrence_rate_per_prize':float(counts[:,num].sum()/(n*27)),'presence_rate_last_30':rr(30),'presence_rate_last_90':rr(90),'presence_rate_last_180':rr(180),'presence_rate_last_365':rr(365),'presence_rate_last_730':rr(730),'current_gap_draws':current_gap(presence[:,num]),'max_gap_draws_3y':mg,'longest_presence_streak_3y':ms,'special_hits_3y':int(special[:,num].sum())})
    out=root/'output'/'statistics_3y_00_99.csv';out.parent.mkdir(exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    bp=sorted(rows,key=lambda r:r['presence_rate_3y'],reverse=True);bo=sorted(rows,key=lambda r:r['occurrences_3y'],reverse=True);bg=sorted(rows,key=lambda r:r['current_gap_draws'],reverse=True)
    summary=[['window_start',dates[0].isoformat()],['window_end',dates[-1].isoformat()],['calendar_window_days',1095],['observed_draws',n],['total_prize_observations',n*len(PRIZE_COLS)],['theoretical_presence_probability',P0],['mean_empirical_presence_probability',float(presence.mean())],['highest_presence_number',bp[0]['number']],['highest_presence_rate',bp[0]['presence_rate_3y']],['lowest_presence_number',bp[-1]['number']],['lowest_presence_rate',bp[-1]['presence_rate_3y']],['highest_occurrence_number',bo[0]['number']],['highest_occurrences',bo[0]['occurrences_3y']],['longest_current_gap_number',bg[0]['number']],['longest_current_gap_draws',bg[0]['current_gap_draws']]]
    with (root/'output'/'statistics_3y_summary.csv').open('w',newline='',encoding='utf-8') as f:w=csv.writer(f);w.writerow(['metric','value']);w.writerows(summary)
    print(f'Wrote {out} with {n} observed draws')
if __name__=='__main__':main()
