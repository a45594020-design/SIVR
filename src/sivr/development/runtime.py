import numpy as np
import torch
from scipy import sparse
from .common import CACHE,write,digest
from .priors import Prior,adjacency
from .injection_models import ResidualGraph,InjectedDecoder,permute_graph

class ExperimentPrior(Prior):
    def __init__(self,phase,cfg,device='cuda',artifact_dir=None):
        group=cfg['group'];self.has_loss=group in ['R','Loss','Hybrid']
        super().__init__(phase,dict(cfg,group='R' if group=='R' else 'B0'),device)
        self.cfg=cfg;self.a=None;self.p=None
        if group not in ['B0','R']:
            original=adjacency(phase,dict(cfg,group='G'));self.a,self.p=permute_graph(original,cfg.get('mismatch',0),cfg.get('graph_seed',10000))
            deg=np.asarray(self.a.sum(1)).ravel();l=((sparse.diags(deg)-self.a)/deg.mean()).tocoo()
            self.graphs=[torch.sparse_coo_tensor(np.stack([l.row,l.col]),torch.tensor(l.data,dtype=torch.double,device=device),l.shape).coalesce()]
            aedges=set(zip(*sparse.triu(original,k=1).nonzero()));bedges=set(zip(*sparse.triu(self.a,k=1).nonzero()))
            self.graph_stats=[{'edges':len(bedges),'mismatch':cfg.get('mismatch',0),'nodes_actually_moved':int(np.sum(self.p!=np.arange(len(self.p)))),'original_edges_removed_fraction':len(aedges-bedges)/len(aedges),'edge_hash':digest([self.a.indptr.tolist(),self.a.indices.tolist(),self.a.data.tolist()])}]
            if artifact_dir is not None:
                sparse.save_npz(artifact_dir/'graph_0_adjacency.npz',self.a);np.save(artifact_dir/'node_permutation.npy',self.p)
    def loss(self,pc):
        u=self.physical(pc)
        if not self.has_loss:return u.sum()*0
        if self.cfg['group']=='R':return (u-u.mean(1,keepdim=True)).square().mean()/(1-1/self.n)
        return (u*torch.sparse.mm(self.graphs[0],u.T).T).sum(1).mean()/self.n

def setup_injection(exp,phase,cfg,out):
    if cfg['group'] not in ['Spatial','Hybrid','MLP-match']:return
    m=np.load(CACHE/phase/'brain_pca.npz')
    residual=ResidualGraph(m['components'],m['mean'],m['legacy_to_unique'],exp.prior.a,seed=30000+cfg['seed'],mlp=cfg['group']=='MLP-match')
    exp.mm_vae.decoders['brain']=InjectedDecoder(exp.mm_vae.decoders['brain'],residual)
    exp.injection=residual
    write(out/'injection_manifest.json',{'position':'brain decoder mean only','hidden_channels':8,'parameter_count':sum(p.numel() for p in residual.parameters()),'module_seed':30000+cfg['seed'],'variance_branch_unchanged':True,'classification_latent_graph_dependency':False,'current_legacy_full_forward_invokes_graph_during_brain_inference':True,'inference_optimization_not_applied':True})
