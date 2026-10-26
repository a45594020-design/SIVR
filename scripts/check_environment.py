import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import argparse,importlib.util,json
p=argparse.ArgumentParser();p.add_argument('--paths');a=p.parse_args()
from sivr.paths import load_paths
v=load_paths(a.paths)
mods=['numpy','pandas','scipy','sklearn','torch','networkx','h5py','tensorboardX','torchvision','matplotlib','seaborn','timm','safetensors']
r={'packages':{m:bool(importlib.util.find_spec(m)) for m in mods},'paths':{k:{'value':str(x),'exists':x.exists()} for k,x in v.items()}}
try:
 import torch
 r['torch_version']=torch.__version__;r['cuda_available']=torch.cuda.is_available()
except ImportError:pass
print(json.dumps(r,indent=2))
