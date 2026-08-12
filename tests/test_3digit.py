import datetime as dt
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from xsmb_3digit import SPECS, matrices, recipe_probability

def make_draw(day,values):
    row={'date':day};cols={'special':5,'prize1':5,'prize2_1':5,'prize2_2':5,**{f'prize3_{i}':5 for i in range(1,7)},**{f'prize4_{i}':4 for i in range(1,5)},**{f'prize5_{i}':4 for i in range(1,7)},**{f'prize6_{i}':3 for i in range(1,4)},**{f'prize7_{i}':2 for i in range(1,5)}}
    for n,(c,w) in enumerate(cols.items()):row[c]=str(values[n%len(values)]).zfill(w)[-w:]
    return row

def test_baselines():
    assert SPECS['suffix3_any'].positions==23
    assert SPECS['g6_exact'].positions==3
    assert abs(SPECS['suffix3_any'].baseline-(1-(999/1000)**23))<1e-15

def test_matrices_distinguish_targets():
    row=make_draw(dt.date(2026,1,1),[123,456,789]);a,_=matrices([row],SPECS['suffix3_any']);g,_=matrices([row],SPECS['g6_exact'])
    assert a.shape==(1,1000) and g.shape==(1,1000)
    assert a.sum()>=g.sum()

def test_recipe_probability_vector():
    f={k:np.full(1000,.02) for k in ['long','short','recent_long','weekday','gap']};p=recipe_probability(f,'gap')
    assert p.shape==(1000,) and np.all((p>0)&(p<1))
