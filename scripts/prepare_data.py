import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import argparse,os,runpy
p=argparse.ArgumentParser();p.add_argument('--paths',required=True);a=p.parse_args();os.environ['SIVR_PATHS']=str(Path(a.paths).resolve())
from sivr.paths import load_paths
v=load_paths()
for k in ['replication_root','replication_cache']:v[k].mkdir(parents=True,exist_ok=True)
import torch
if not torch.cuda.is_available():p.error('The archived PCA/feature preparation requires CUDA.')
runpy.run_module('sivr.replication.prepare',run_name='__main__')
