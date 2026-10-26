"""Duplicate-safe inverse PCA with trace-normalized geometry/signed graphs."""
import numpy as np
import torch
from scipy import sparse
from scipy.spatial import cKDTree
from .common import CACHE,digest

def adjacency(phase,cfg):
 m=np.load(CACHE/phase/'brain_pca.npz');coords=m['coordinates'];n=len(coords)
 mode=cfg['group'];rng=np.random.default_rng(cfg.get('graph_seed',10000))
 if mode in ['G','G-perm','G-rewire']:
  k=cfg.get('k',6);tree=cKDTree(coords);d,j=tree.query(coords,k=k+1);spacing=float(np.median(d[:,1]));bw=spacing*cfg.get('scale',1.)
  i=np.repeat(np.arange(n),k);d=d[:,1:].ravel();j=j[:,1:].ravel();ok=(d>1e-10)&(d<=3*spacing)
  a=sparse.csr_matrix((np.exp(-.5*(d[ok]/bw)**2),(i[ok],j[ok])),shape=(n,n));a=a.maximum(a.T)
  if mode=='G-perm':
   p=rng.permutation(n);a=a[p][:,p]
  if mode=='G-rewire':
   import networkx as nx
   e=sparse.triu(a,k=1).tocoo();g=nx.Graph();g.add_nodes_from(range(n));g.add_edges_from(zip(e.row.tolist(),e.col.tolist()))
   before=np.array([g.degree(i) for i in range(n)]);nx.double_edge_swap(g,nswap=10*g.number_of_edges(),max_tries=200*g.number_of_edges(),seed=int(cfg.get('graph_seed',10000)))
   assert np.array_equal(before,[g.degree(i) for i in range(n)])
   edges=np.array(list(g.edges()));w=rng.permutation(e.data)
   a=sparse.coo_matrix((np.r_[w,w],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
  return a
 if mode=='S':
  x=np.load(CACHE/phase/'fit_unique_z.npy');corr=np.corrcoef(x,rowvar=False);corr=np.nan_to_num(corr);np.fill_diagonal(corr,0)
  sign=cfg.get('sign','signed');k=cfg.get('k',20);score=np.maximum(corr,0) if sign=='positive' else np.abs(corr)
  j=np.argpartition(score,-k,axis=1)[:,-k:];i=np.repeat(np.arange(n),k);j=j.ravel();w=score[i,j]
  a=sparse.csr_matrix((w,(i,j)),shape=(n,n));a=a.maximum(a.T).tocoo()
  if sign=='signed':a.data*=np.sign(corr[a.row,a.col])
  return a.tocsr()
 raise ValueError(mode)

class Prior:
 def __init__(self,phase,cfg,device='cuda',artifact_dir=None):
  self.cfg=cfg;self.phase=phase;self.device=device
  m=np.load(CACHE/phase/'brain_pca.npz');inv=m['legacy_to_unique'];n=len(m['coordinates']);self.n=n
  b=sparse.csr_matrix((1/np.bincount(inv)[inv],(np.arange(len(inv)),inv)),shape=(len(inv),n))
  self.components=torch.tensor(m['components']@b,dtype=torch.double,device=device)
  self.mean=torch.tensor(m['mean']@b,dtype=torch.double,device=device)
  self.graphs=[];self.graph_stats=[]
  configs=[] if cfg['group'] in ['B0','R'] else ([dict(cfg,group='S',**cfg['stat']),dict(cfg,group='G',**cfg['geo'])] if cfg['group']=='S+G' else [cfg])
  for index,c in enumerate(configs):
   a=adjacency(phase,c);deg=np.asarray(abs(a).sum(1)).ravel();md=deg.mean();assert md>0
   if artifact_dir is not None:sparse.save_npz(artifact_dir/f'graph_{index}_adjacency.npz',a)
   l=(sparse.diags(deg)-a)/md;l=l.tocoo();indices=torch.tensor(np.vstack([l.row,l.col]),device=device)
   t=torch.sparse_coo_tensor(indices,torch.tensor(l.data,dtype=torch.double,device=device),l.shape).coalesce();self.graphs.append(t)
   e=sparse.triu(a,k=1).tocoo()
   self.graph_stats.append({'group':c['group'],'edges':int(len(e.data)),'negative_edge_fraction':float(np.mean(e.data<0)),'isolates':int(np.sum(deg==0)),'mean_weighted_degree':float(md),'trace_normalized':float(l.diagonal().sum()/n),'binary_degrees_sha256':digest(np.diff(a.indptr).tolist()),'weighted_degree_quantiles':np.quantile(deg,[0,.25,.5,.75,1]).tolist(),'edge_hash':digest([e.row.tolist(),e.col.tolist(),e.data.tolist()])})
 def physical(self,pc):return pc@self.components+self.mean
 def losses(self,pc):
  u=self.physical(pc)
  if self.cfg['group']=='B0':return [u.sum()*0]
  if self.cfg['group']=='R':return [(u-u.mean(1,keepdim=True)).square().mean()/(1-1/self.n)]
  return [(u*torch.sparse.mm(l,u.T).T).sum(1).mean()/self.n for l in self.graphs]
 def loss(self,pc):
  losses=self.losses(pc)
  if len(losses)==2:
   ratio=self.cfg.get('stat_fraction',.5);return ratio*losses[0]+(1-ratio)*losses[1]
  return losses[0]
