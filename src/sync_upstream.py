#!/usr/bin/env python3
"""Mirror canonical XSMB CSV datasets from khiemdoan and validate consistency."""
from __future__ import annotations
import csv, hashlib, io, json, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE='https://raw.githubusercontent.com/khiemdoan/vietnam-lottery-xsmb-analysis/refs/heads/main/data'
FILES=['xsmb.csv','xsmb-2-digits.csv','xsmb-sparse.csv']
DEST=ROOT/'data'/'upstream'
def download(name):
    req=urllib.request.Request(f'{BASE}/{name}',headers={'User-Agent':'xsmb-research-mirror/1.0'})
    with urllib.request.urlopen(req,timeout=60) as r:return r.read()
def sha(data):return hashlib.sha256(data).hexdigest()
def parse_dates(data):
    rows=list(csv.DictReader(io.StringIO(data.decode('utf-8-sig')))); dates=[r['date'][:10] for r in rows]
    if dates!=sorted(dates):raise ValueError('Upstream dates are not sorted')
    if len(dates)!=len(set(dates)):raise ValueError('Duplicate dates in upstream dataset')
    return dates
def main():
    DEST.mkdir(parents=True,exist_ok=True); payloads={n:download(n) for n in FILES}; date_sets={n:parse_dates(payloads[n]) for n in FILES}; raw=date_sets['xsmb.csv']
    for n,d in date_sets.items():
        if d!=raw:raise ValueError(f'Date mismatch between xsmb.csv and {n}')
    for n,b in payloads.items():(DEST/n).write_bytes(b)
    meta={'source_repository':'khiemdoan/vietnam-lottery-xsmb-analysis','license':'MIT','first_date':raw[0],'last_date':raw[-1],'observed_draws':len(raw),'files':{n:{'sha256':sha(b),'bytes':len(b)} for n,b in payloads.items()}}
    (DEST/'metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
