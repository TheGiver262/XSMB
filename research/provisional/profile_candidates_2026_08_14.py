#!/usr/bin/env python3
import csv, json, math, statistics
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

DATA = Path('data/upstream/xsmb.csv')
RANKING = Path('/tmp/rank/current_ranking.csv')
P0 = 1 - 0.99**27
TARGET = date(2026,8,14)

rows=[]
with DATA.open(encoding='utf-8') as f:
    r=csv.DictReader(f)
    prize_cols=[c for c in r.fieldnames if c!='date']
    for row in r:
        d=date.fromisoformat(row['date'])
        vals=[str(row[c]).zfill(2)[-2:] for c in prize_cols]
        cnt=Counter(vals)
        rows.append((d,cnt))
rows.sort(key=lambda x:x[0])
assert rows[-1][0].isoformat()=='2026-08-13', rows[-1][0]

with RANKING.open(encoding='utf-8') as f:
    rank_rows=list(csv.DictReader(f))
rank_by_num={r['number']: int(r['rank']) for r in rank_rows}
score_by_num={r['number']: float(r['score_percentile']) for r in rank_rows}


def subset_last(n):
    return rows[-min(n,len(rows)):]

def stats_for(num, rr):
    n=len(rr)
    hits=sum(c[num]>0 for _,c in rr)
    occ=sum(c[num] for _,c in rr)
    rate=hits/n if n else None
    lift=(rate-P0)*100 if n else None
    z=(rate-P0)/math.sqrt(P0*(1-P0)/n) if n else None
    return {'draws':n,'hit_draws':hits,'occurrences':occ,'presence_rate':rate,'lift_pp':lift,'z_vs_fair':z}

def gap_info(num):
    idx=[i for i,(_,c) in enumerate(rows) if c[num]>0]
    current=(len(rows)-1-idx[-1]) if idx else len(rows)
    gaps=[idx[i]-idx[i-1]-1 for i in range(1,len(idx))]
    gaps_sorted=sorted(gaps)
    def q(p):
        if not gaps_sorted: return None
        j=(len(gaps_sorted)-1)*p
        lo=int(math.floor(j)); hi=int(math.ceil(j))
        if lo==hi:return gaps_sorted[lo]
        return gaps_sorted[lo]*(hi-j)+gaps_sorted[hi]*(j-lo)
    return {'current_gap_draws':current,'historical_mean_gap':statistics.mean(gaps) if gaps else None,'historical_median_gap':statistics.median(gaps) if gaps else None,'gap_p75':q(.75),'gap_p90':q(.90),'max_gap':max(gaps) if gaps else None}

def monthly_1y(num):
    rr=rows[-365:]
    g=defaultdict(list)
    for d,c in rr:g[d.strftime('%Y-%m')].append((d,c))
    return [{'month':m,**stats_for(num,v)} for m,v in sorted(g.items())]

def weekly_recent(num, weeks=16):
    g=defaultdict(list)
    for d,c in rows[-140:]:
        y,w,_=d.isocalendar(); g[f'{y}-W{w:02d}'].append((d,c))
    items=sorted(g.items())[-weeks:]
    return [{'week':k,**stats_for(num,v)} for k,v in items]

def recent_dates(num, limit=20):
    out=[]
    for d,c in reversed(rows):
        if c[num]>0:
            out.append({'date':d.isoformat(),'count':c[num]})
            if len(out)>=limit:break
    return out

def friday_stats(num):
    rr=[x for x in rows if x[0].weekday()==TARGET.weekday()]
    rr1=[x for x in rows[-365:] if x[0].weekday()==TARGET.weekday()]
    return {'full_same_weekday':stats_for(num,rr),'last365_same_weekday':stats_for(num,rr1)}

horizons=[('30d',30),('60d',60),('90d',90),('180d',180),('1y',365),('2y',730),('3y',1095),('5y',1825),('10y',3650),('full',len(rows))]
profiles={}
for num in [f'{i:02d}' for i in range(100)]:
    hs={k:stats_for(num,subset_last(n)) for k,n in horizons}
    long_keys=['1y','3y','5y','10y','full']
    above=sum(hs[k]['presence_rate']>=P0 for k in long_keys)
    positive_z=sum(hs[k]['z_vs_fair']>0 for k in long_keys)
    profiles[num]={
        'number':num,'v3_rank':rank_by_num[num],'v3_score_percentile':score_by_num[num],
        'horizons':hs,'long_horizons_at_or_above_fair':above,'long_positive_z_count':positive_z,
        'gap':gap_info(num),'monthly_last365':monthly_1y(num),'weekly_recent':weekly_recent(num),
        'recent_hit_dates':recent_dates(num),'weekday':friday_stats(num)
    }

# Evidence shortlist: descriptive only, not a calibrated predictive score.
# Keep numbers with broad long-horizon support and rank them by support, then V3 rank.
short=sorted(profiles.values(), key=lambda p:(-p['long_horizons_at_or_above_fair'], p['v3_rank']))

print('DATA', rows[0][0], rows[-1][0], len(rows), 'FAIR', P0)
print('=== V3 TOP 20 WITH MULTI-HORIZON EVIDENCE ===')
for p in sorted(profiles.values(), key=lambda p:p['v3_rank'])[:20]:
    h=p['horizons']
    print(json.dumps({
        'number':p['number'],'v3_rank':p['v3_rank'],'support':p['long_horizons_at_or_above_fair'],
        'gap':p['gap']['current_gap_draws'],
        '30d':h['30d']['presence_rate'],'90d':h['90d']['presence_rate'],'1y':h['1y']['presence_rate'],
        '3y':h['3y']['presence_rate'],'5y':h['5y']['presence_rate'],'10y':h['10y']['presence_rate'],'full':h['full']['presence_rate'],
        '1y_hits':h['1y']['hit_draws'],'1y_occ':h['1y']['occurrences'],'full_hits':h['full']['hit_draws'],'full_occ':h['full']['occurrences']
    }, ensure_ascii=False))

print('=== TOP 20 BROAD LONG-HORIZON SUPPORT (DESCRIPTIVE) ===')
for p in short[:20]:
    print(p['number'], 'support',p['long_horizons_at_or_above_fair'],'v3',p['v3_rank'],
          '1y',round(p['horizons']['1y']['presence_rate']*100,3),
          '3y',round(p['horizons']['3y']['presence_rate']*100,3),
          '5y',round(p['horizons']['5y']['presence_rate']*100,3),
          '10y',round(p['horizons']['10y']['presence_rate']*100,3),
          'full',round(p['horizons']['full']['presence_rate']*100,3),
          'gap',p['gap']['current_gap_draws'])

# Full detailed profile for top V3 + broad support candidates, and selected diagnostic numbers.
focus=set([p['number'] for p in sorted(profiles.values(), key=lambda p:p['v3_rank'])[:15]])
focus.update([p['number'] for p in short[:15]])
focus.update(['83','91','92','14','60','62','26'])
print('=== DETAILED_PROFILES_JSON ===')
print(json.dumps({n:profiles[n] for n in sorted(focus)}, ensure_ascii=False))
