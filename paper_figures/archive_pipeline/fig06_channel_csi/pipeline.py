#!/usr/bin/env python3
import argparse, hashlib, json, subprocess, sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
def call(c): subprocess.run(c,cwd=ROOT,check=True)
def manifest(command):
    artifacts={}
    for path in HERE.rglob('*'):
        if path.is_file() and path.name != 'manifest.json':
            artifacts[str(path.relative_to(HERE))]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (HERE/'manifest.json').write_text(json.dumps({'schema':'tnse_figure_manifest_v1','figure':'fig06_channel_csi','created_at':datetime.now().astimezone().isoformat(),'frozen_git_revision':'ad2ee67d76e05effa4953db18ab677502ddd3ba1','command':command,'artifacts':artifacts},indent=2),encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('evaluate','plot','all'));a=p.parse_args()
    if a.stage in ('evaluate','all'):
        for n in (20,30,40,50): call([sys.executable,'-u',str(HERE/'run_csi.py'),'--num-uav',str(n),'--output',str(HERE/f'data/raw/csi_N{n}.json')])
    if a.stage in ('plot','all'): call([sys.executable,'-u',str(HERE/'plot.py')])
    manifest([sys.executable,*sys.argv])
if __name__=='__main__': main()
