import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pathlib import Path
import pandas as pd
(ROOT/"results").mkdir(exist_ok=True)
p=ROOT/"data/aggregates"
d=pd.read_csv(p/"all_85_runs_percent.csv")
assert len(d)==255
assert not d.duplicated(["subject","group","seed","space"]).any()
assert d[["subject","group","seed"]].drop_duplicates().shape[0]==85
assert set(d.space)=={"V","T","VT"}
assert d[["top1","top5"]].ge(0).all().all() and d[["top1","top5"]].le(100).all().all()
g=d.groupby(["subject","group","space"])[["top1","top5"]].agg(["mean","std"])
assert abs(g.loc[("sub-04","B0","VT"),("top1","mean")]-7.0285714285714)<1e-8
assert abs(g.loc[("sub-04","G","VT"),("top1","mean")]-6.9714285714286)<1e-8
g.to_csv(ROOT/"results/summary_verified.csv")
print("PASS: 85 units, 255 reference-space rows; aggregate mean/sample-SD verified")
