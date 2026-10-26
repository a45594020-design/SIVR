import json
from pathlib import Path
import numpy as np
import pytest
from sivr.paths import load_paths
ROOT=Path(__file__).resolve().parents[1]

def test_paths_relative_to_config_file(tmp_path):
    p=tmp_path/'paths.json'
    p.write_text(json.dumps({'replication_root':'runs','replication_cache':'cache'}))
    v=load_paths(p)
    assert v['replication_root']==tmp_path/'runs'
    assert v['replication_cache']==tmp_path/'cache'

def test_aggregate_units_and_metrics():
    import pandas as pd
    d=pd.read_csv(ROOT/'data/aggregates/all_85_runs_percent.csv')
    assert len(d)==255
    assert not d.duplicated(['subject','group','seed','space']).any()
    assert len(d[['subject','group','seed']].drop_duplicates())==85
    a=d[(d.subject=='sub-04') & (d.group=='B0') & (d.space=='VT')]
    assert len(a)==5
    assert a.top1.mean()==pytest.approx(7.0285714285714)

def test_duplicate_pooling_and_loss_gradient(tmp_path,monkeypatch):
    import torch
    from sivr.replication import priors
    q=tmp_path/'final';q.mkdir()
    inv=np.array([0,1,0,2]);comp=np.eye(4)[:2];mean=np.array([1.,2.,3.,4.])
    np.savez(q/'brain_pca.npz',components=comp,mean=mean,legacy_to_unique=inv,coordinates=np.array([[0,0,0],[1,0,0],[0,1,0]]))
    monkeypatch.setattr(priors,'CACHE',tmp_path)
    h=torch.tensor([[.4,.2],[-.3,.8]],dtype=torch.double,requires_grad=True)
    pr=priors.Prior('final',{'group':'R'},device='cpu')
    legacy=h@torch.tensor(comp)+torch.tensor(mean)
    expected=torch.stack([legacy[:,inv==i].mean(1) for i in range(3)],1)
    assert torch.allclose(pr.physical(h),expected)
    loss=pr.loss(h);g=torch.autograd.grad(loss,h)[0]
    delta=1e-6;hp=h.detach().clone();hm=h.detach().clone();hp[0,0]+=delta;hm[0,0]-=delta
    assert float(g[0,0])==pytest.approx(float((pr.loss(hp)-pr.loss(hm))/(2*delta)),rel=1e-5)
    assert torch.allclose(loss,pr.loss(h.repeat(2,1)))

def test_source_controls_preserve_structure(tmp_path,monkeypatch):
    from sivr.replication import priors
    from scipy import sparse
    monkeypatch.setattr(priors,'CACHE',tmp_path)
    n=18;q=tmp_path/'final';q.mkdir();rng=np.random.default_rng(19)
    np.savez(q/'brain_pca.npz',coordinates=rng.normal(size=(n,3)))
    np.save(q/'fit_unique_z.npy',rng.normal(size=(100,n)))
    cfg={'group':'S','k':3,'sign':'absolute','graph_seed':77}
    a=priors.adjacency('final',cfg)
    b=priors.adjacency('final',dict(cfg,group='S-perm'))
    c=priors.adjacency('final',dict(cfg,group='S-rewire'))
    assert np.allclose(np.sort(a.data),np.sort(b.data))
    assert np.allclose(np.sort(a.data),np.sort(c.data))
    assert np.array_equal(np.diff(a.indptr),np.diff(c.indptr))
    assert np.allclose(np.linalg.eigvalsh(a.toarray()),np.linalg.eigvalsh(b.toarray()))
