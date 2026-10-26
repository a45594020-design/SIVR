"""Shared immutable paths and atomic provenance helpers; no credentials."""
import hashlib,json,os
from pathlib import Path
from sivr.paths import load_paths
_paths=load_paths()
ROOT=_paths['development_root']
BASE=_paths['development_base']
CACHE=_paths['development_cache']
DATA=_paths['data_root']
CODE=_paths['backbone_root']

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2,ensure_ascii=False));os.replace(q,p)
