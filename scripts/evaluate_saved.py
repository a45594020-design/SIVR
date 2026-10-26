import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import argparse,os,runpy
p=argparse.ArgumentParser(description='Development saved-checkpoint evaluation (requires original inputs and CUDA).')
p.add_argument('--paths',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--config',required=True);a=p.parse_args()
os.environ['SIVR_PATHS']=str(Path(a.paths).resolve())
sys.argv=['evaluate_saved',str(Path(a.source).resolve()),str(Path(a.out).resolve()),str(Path(a.config).resolve())]
runpy.run_module('sivr.development.evaluate_saved',run_name='__main__')
