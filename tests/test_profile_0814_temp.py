import csv, json, math, statistics, subprocess, sys, warnings
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

P0 = 1 - 0.99**27
TARGET = date(2026, 8, 14)


def _stats(num, rr):
    n=len(rr); hits=sum(c[num]>0 for _,c in rr); occ=sum(c[num] for _,c in rr)
    rate=hits/n if n else 0.0
    return {'n':n,'hits':hits,'occ':occ,'rate':rate,'lift_pp':(rate-P0)*100}


def test_profile_0814():
    out=Path('/tmp/rank_profile_0814')
    subprocess.run([sys.executable,'src/xsmb_rank_challenger.py','--target-date','2026-08-14','--out-dir',str(out)],check=True,capture_output=True,text=True)
    with Path('data/upstream/xsmb.csv').open(encoding='utf-8') as f:
        r=csv.DictReader(f); cols=[c for c in r.fieldnames if c!='date']; rows=[]
        for row in r:
            d=date.fromisoformat(row['date']); cnt=Counter(str(row[c]).zfill(2)[-2:] for c in cols); rows.append((d,cnt))
    rows.sort(key=lambda x:x[0]); assert rows[-1][0]==date(2026,8,13)
    with (out/'current_ranking.csv').open(encoding='utf-8') as f:
        ranks={x['number']:int(x['rank']) for x in csv.DictReader(f)}

    horizons={'30':30,'60':60,'90':90,'180':180,'1y':365,'2y':730,'3y':1095,'5y':1825,'10y':3650,'full':len(rows)}
    profiles={}
    for i in range(100):
        num=f'{i:02d}'; hs={k:_stats(num,rows[-min(n,len(rows)):]) for k,n in horizons.items()}
        idx=[j for j,(_,c) in enumerate(rows) if c[num]>0]
        gaps=[idx[j]-idx[j-1]-1 for j in range(1,len(idx))]
        gap=len(rows)-1-idx[-1]
        support=sum(hs[k]['rate']>=P0 for k in ['1y','3y','5y','10y','full'])
        profiles[num]={'num':num,'rank':ranks[num],'support':support,'gap':gap,
            'gap_mean':statistics.mean(gaps),'gap_med':statistics.median(gaps),
            'h':hs}
    summary=sorted(profiles.values(),key=lambda p:(-p['support'],p['rank']))
    compact=[]
    for p in summary[:30]:
        compact.append({'n':p['num'],'r':p['rank'],'s':p['support'],'g':p['gap'],
            '30':round(p['h']['30']['rate']*100,3),'90':round(p['h']['90']['rate']*100,3),
            '1y':round(p['h']['1y']['rate']*100,3),'3y':round(p['h']['3y']['rate']*100,3),
            '5y':round(p['h']['5y']['rate']*100,3),'10y':round(p['h']['10y']['rate']*100,3),
            'full':round(p['h']['full']['rate']*100,3),
            '1yh':p['h']['1y']['hits'],'1yo':p['h']['1y']['occ'],'fh':p['h']['full']['hits'],'fo':p['h']['full']['occ']})
    warnings.warn('PROFILE_TOP30='+json.dumps(compact,separators=(',',':')))

    focus={p['num'] for p in summary[:15]}
    focus.update(['83','91','92','14','60','95','27','69','09','13','62','26'])
    for num in sorted(focus,key=lambda n:ranks[n]):
        p=profiles[num]
        monthly=defaultdict(list)
        for d,c in rows[-365:]: monthly[d.strftime('%Y-%m')].append((d,c))
        months=[{'m':m,'n':len(rr),'h':sum(c[num]>0 for _,c in rr),'o':sum(c[num] for _,c in rr)} for m,rr in sorted(monthly.items())]
        weekly=defaultdict(list)
        for d,c in rows[-120:]:
            y,w,_=d.isocalendar(); weekly[f'{y}-W{w:02d}'].append((d,c))
        weeks=[{'w':w,'h':sum(c[num]>0 for _,c in rr),'o':sum(c[num] for _,c in rr)} for w,rr in sorted(weekly.items())[-12:]]
        dates=[]
        for d,c in reversed(rows):
            if c[num]>0:
                dates.append([d.isoformat(),c[num]])
                if len(dates)>=15: break
        same=[x for x in rows if x[0].weekday()==TARGET.weekday()]
        same1=[x for x in rows[-365:] if x[0].weekday()==TARGET.weekday()]
        detail={'n':num,'rank':p['rank'],'support':p['support'],'gap':p['gap'],'gap_mean':round(p['gap_mean'],3),'gap_med':p['gap_med'],
            'h':{k:{'hits':v['hits'],'occ':v['occ'],'rate':round(v['rate']*100,3),'lift':round(v['lift_pp'],3)} for k,v in p['h'].items()},
            'months':months,'weeks':weeks,'recent':dates,
            'weekday_full':_stats(num,same),'weekday_1y':_stats(num,same1)}
        warnings.warn('DETAIL_'+num+'='+json.dumps(detail,separators=(',',':')))
    assert True
