#!/usr/bin/env python3
"""Three-year rolling wrapper for the existing 2-digit daily pipeline."""
from __future__ import annotations
import argparse, datetime as dt, subprocess, sys
import daily_pipeline as base
base.WINDOW_DAYS=1095

def lazy_model_v2():
    from xsmb_probability_v2 import load_draws, matrices, next_probabilities, write_prediction
    return load_draws, matrices, next_probabilities, write_prediction
base.lazy_model=lazy_model_v2

def run_statistics_3y():
    subprocess.run([sys.executable,str(base.ROOT/'src'/'build_statistics_3y.py')],cwd=base.ROOT,check=True)
base.run_statistics=run_statistics_3y

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True);a=sub.add_parser('forecast');a.add_argument('--target-date');b=sub.add_parser('settle');b.add_argument('--date');args=p.parse_args()
    if args.cmd=='forecast':return base.forecast(dt.date.fromisoformat(args.target_date) if args.target_date else base.local_today())
    return base.settle(dt.date.fromisoformat(args.date) if args.date else base.local_today())
if __name__=='__main__':raise SystemExit(main())
