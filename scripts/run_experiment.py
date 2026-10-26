import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import argparse,json,os,runpy
p=argparse.ArgumentParser(description='Launch one research runner; never starts an operation/shutdown queue.')
p.add_argument('--family',choices=['replication','development'],required=True)
p.add_argument('--paths',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
p.add_argument('--phase',required=True,choices=['validation','final','fold1']);p.add_argument('--calibrate',action='store_true')
a=p.parse_args()
if (a.family=='development') != (a.phase=='fold1'):p.error('development requires fold1; replication requires validation/final')
os.environ['SIVR_PATHS']=str(Path(a.paths).resolve())
from sivr.paths import load_paths
paths=load_paths();c=paths[a.family+'_cache']/a.phase
missing=[str(c/f) for f in ['data.npz','brain_pca.npz','fit_unique_z.npy'] if not (c/f).exists()]
if not (paths['backbone_root']/'run_epochs_trimodal.py').exists():missing.append('backbone/run_epochs_trimodal.py')
if missing:p.error('Missing inputs: '+', '.join(missing))
import torch
if not torch.cuda.is_available():p.error('Archived training runners require a CUDA-enabled PyTorch runtime.')
if a.family=='replication':
    root=paths['replication_root'];root.mkdir(parents=True,exist_ok=True)
    f=root/'frozen_configuration.json'
    source=ROOT/'configs/replication/frozen_configuration.json'
    if f.exists() and f.read_bytes()!=source.read_bytes():p.error('Existing frozen configuration differs; use a separate output root.')
    if not f.exists():f.write_bytes(source.read_bytes())
sys.argv=['run','--config',str(Path(a.config).resolve()),'--out',str(Path(a.out).resolve()),'--phase',a.phase]+(['--calibrate'] if a.calibrate else [])
runpy.run_module('sivr.'+a.family+'.run',run_name='__main__')
