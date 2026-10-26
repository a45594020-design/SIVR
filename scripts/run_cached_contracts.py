import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
"""Mathematical counterexamples and graph-control invariants before training."""
import numpy as np,torch,json
from scipy import sparse
from sivr.replication.common import ROOT,CACHE,write
from sivr.replication.priors import Prior,adjacency

def main():
 # Correlation does not identify values; signed agreement is distinct from smoothing.
 x=np.arange(1,6,dtype=float);y=3*x+7
 assert np.corrcoef(x,y)[0,1]>.99999 and np.mean((x-y)**2)>1
 # Separate the reviewer's raw-value counterexample from the standardized field
 # actually constrained in this implementation.
 zx=(x-x.mean())/x.std(ddof=1);zy=(y-y.mean())/y.std(ddof=1)
 assert np.allclose(zx,zy)
 u=torch.tensor([[1.,-1.]],dtype=torch.double,requires_grad=True)
 signed=(u[:,0]+u[:,1]).square().mean();unsigned=(u[:,0]-u[:,1]).square().mean()
 assert signed.item()==0 and unsigned.item()==4
 results={'correlation_counterexample':True,'negative_sign_counterexample':True}
 g=adjacency('validation',{'group':'G','k':6,'scale':1})
 perm=adjacency('validation',{'group':'G-perm','k':6,'scale':1,'graph_seed':10000})
 rew=adjacency('validation',{'group':'G-rewire','k':6,'scale':1,'graph_seed':10000})
 assert np.allclose(np.sort(g.data),np.sort(perm.data)) and np.array_equal(np.sort(np.diff(g.indptr)),np.sort(np.diff(perm.indptr)))
 assert np.array_equal(np.diff(g.indptr),np.diff(rew.indptr)) and np.allclose(np.sort(g.data),np.sort(rew.data))
 results['permutation_weight_degree_invariants']=True;results['rewire_exact_binary_degree_and_weights']=True
 prior=Prior('validation',{'group':'G','k':6,'scale':1});m=np.load(CACHE/'validation/brain_pca.npz')
 pc=torch.randn(4,len(m['components']),dtype=torch.double,device='cuda',requires_grad=True)
 legacy=pc@torch.tensor(m['components'],device='cuda')+torch.tensor(m['mean'],device='cuda')
 inv=m['legacy_to_unique'];manual=torch.stack([legacy[:,inv==i].mean(1) for i in range(prior.n)],1)
 assert torch.allclose(prior.physical(pc),manual,atol=1e-10,rtol=1e-10)
 loss=prior.loss(pc);grad=torch.autograd.grad(loss,pc)[0];direction=grad/grad.norm();eps=1e-5
 fd=(prior.loss(pc.detach()+eps*direction)-prior.loss(pc.detach()-eps*direction))/(2*eps)
 assert abs(float(fd-grad.norm()))/float(grad.norm())<1e-5
 results['inverse_pca_mean_duplicate_pooling']=True;results['finite_difference_relative_error']=abs(float(fd-grad.norm()))/float(grad.norm())
 # Batch replication and constant geometry field must preserve normalization/nullspace.
 assert torch.allclose(prior.loss(pc),prior.loss(pc.repeat(2,1)),rtol=1e-10,atol=1e-10)
 assert torch.sparse.mm(prior.graphs[0],torch.ones(prior.n,1,device='cuda',dtype=torch.double)).abs().max()<1e-10
 results['batch_replication_invariance']=True;results['geometry_constant_nullspace']=True
 results['status']='passed';write(ROOT/'X2_contract_tests.json',results);print(json.dumps(results))
if __name__=='__main__':main()
