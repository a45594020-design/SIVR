"""Decoder-only shared-node residual; all geometric buffers are frozen."""
import numpy as np
import torch
from torch import nn
from scipy import sparse

def permute_graph(a,fraction,seed):
    n=a.shape[0];rng=np.random.default_rng(seed);p=np.arange(n)
    if fraction==1:p=rng.permutation(n)
    elif fraction>0:
        selected=rng.choice(n,int(round(n*fraction)),replace=False)
        p[selected]=rng.permutation(selected)
    return a[p][:,p].tocsr(),p

class ResidualGraph(nn.Module):
    def __init__(self,components,mean,inv,a,seed=30000,device='cuda',mlp=False):
        super().__init__();n=a.shape[0];inv=np.asarray(inv)
        q=sparse.csr_matrix((1/np.bincount(inv)[inv],(np.arange(len(inv)),inv)),shape=(len(inv),n))
        j=sparse.csr_matrix((np.ones(len(inv)),(inv,np.arange(len(inv)))),shape=(n,len(inv)))
        assert np.allclose((j@q).toarray(),np.eye(n))
        self.register_buffer('components',torch.as_tensor(components@q,dtype=torch.double,device=device))
        self.register_buffer('mean',torch.as_tensor(mean@q,dtype=torch.double,device=device))
        self.register_buffer('lift',torch.as_tensor(j@components.T,dtype=torch.double,device=device))
        a=a+sparse.eye(n);d=np.asarray(a.sum(1)).ravel()**-.5
        a=(sparse.diags(d)@a@sparse.diags(d)).tocoo()
        self.register_buffer('adj',torch.sparse_coo_tensor(np.stack([a.row,a.col]),torch.as_tensor(a.data,dtype=torch.double),a.shape,device=device).coalesce())
        self.mlp=mlp;self.last={}
        # Independent RNG: module initialization cannot change backbone/training RNG.
        gen=torch.Generator(device='cpu').manual_seed(seed)
        def parameter(shape):return nn.Parameter((torch.randn(shape,generator=gen,dtype=torch.double)*.1).to(device))
        self.self_weight=parameter((1,8));self.neigh_weight=parameter((1,8));self.out_weight=parameter((8,1))
        self.bias=nn.Parameter(torch.zeros(8,dtype=torch.double,device=device))
        self.gate=nn.Parameter(torch.tensor(.1,dtype=torch.double,device=device))

    def forward(self,h):
        from torch.utils.checkpoint import checkpoint
        shape=h.shape;flat=h.reshape(-1,shape[-1])
        parts=[]
        for lo in range(0,len(flat),128):
            chunk=flat[lo:lo+128]
            parts.append(checkpoint(self.compute,chunk,use_reentrant=False) if torch.is_grad_enabled() else self.compute(chunk))
        return torch.cat(parts).reshape(shape)

    def compute(self,flat):
        u=flat@self.components+self.mean
        neighbor=u if self.mlp else torch.sparse.mm(self.adj,u.T).T
        hidden=torch.relu(u[:,:,None]*self.self_weight+neighbor[:,:,None]*self.neigh_weight+self.bias)
        delta=(hidden@self.out_weight).squeeze(-1)*torch.tanh(self.gate)
        change=delta@self.lift;result=flat+change
        with torch.no_grad():
            projected=change@self.components
            self.last={'gate':float(torch.tanh(self.gate)),'delta_rms':float(delta.square().mean().sqrt()),'pca_change_rms':float(change.square().mean().sqrt()),'projection_retained_norm_ratio':float(projected.norm()/(delta.norm()+1e-20)),'projection_error_ratio':float((projected-delta).norm()/(delta.norm()+1e-20))}
        return result

class InjectedDecoder(nn.Module):
    def __init__(self,base,residual):super().__init__();self.base=base;self.residual=residual
    def forward(self,*args,**kwargs):
        values=self.base(*args,**kwargs)
        return self.residual(values[0]),values[1]
