#!/usr/bin/env python3
"""Fig. 3 uses exactly the ten seed-resolved frozen-GAT reports from Fig. 2.

No additional model is trained: this is deliberately an audit of the same
training trajectories.  ``--source`` makes the dependency explicit instead of
silently regenerating data with a different seed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default='../fig02_training_curves/data/raw');p.add_argument('--nodes',default='20,30,40,50');p.add_argument('--seeds',default='0,1,2,3,4,5,6,7,8,9');a=p.parse_args();root=(Path(__file__).resolve().parent/a.source).resolve()
 missing=[str(root/f'N{n}'/f'seed{s}'/'training.json') for n in a.nodes.split(',') for s in a.seeds.split(',') if not (root/f'N{n}'/f'seed{s}'/'training.json').is_file()]
 if missing: raise FileNotFoundError('Run Fig. 2 first; missing:\n'+'\n'.join(missing[:8]))
 print(f'Fig. 3 audit source verified: {root}')
if __name__=='__main__':main()
