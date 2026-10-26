import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import argparse,json
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
for src in sorted((ROOT/'configs/replication').glob('*_seed0.json')):
 c=json.loads(src.read_text())
 for seed in range(5):
  cfg=dict(c,seed=seed,graph_seed=10000+seed)
  (out/f"{cfg['group']}_seed{seed}.json").write_text(json.dumps(cfg,indent=2))
print('Wrote 45 configurations; no training started.')
