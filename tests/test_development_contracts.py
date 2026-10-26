import unittest
import numpy as np
import torch
from scipy import sparse
from sivr.development.injection_models import ResidualGraph, permute_graph
from sivr.development.modality_ablation import active_inputs, negative_pairs

class Contracts(unittest.TestCase):
    def test_duplicate_projection_identity_and_zero_gate(self):
        inv=np.array([0,1,0,2]);p=np.eye(4)[:3]
        a=sparse.csr_matrix([[0,1,0],[1,0,2],[0,2,0]])
        mod=ResidualGraph(p,np.zeros(4),inv,a,seed=55,device='cpu')
        h=torch.randn(5,3,dtype=torch.double,requires_grad=True)
        with torch.no_grad():mod.gate.zero_()
        self.assertTrue(torch.equal(h,mod(h)))
        self.assertTrue(torch.allclose(mod.components,torch.tensor(p@np.array([[.5,0,0],[0,1,0],[.5,0,0],[0,0,1]]),dtype=torch.double)))
        with torch.no_grad():mod.gate.fill_(.1)
        mod(h).square().sum().backward()
        self.assertGreater(h.grad.norm(),0)
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in mod.parameters()))

    def test_graph_permutation_invariants_and_pairing(self):
        rng=np.random.default_rng(4);x=rng.uniform(size=(9,9));x=(x+x.T)/2;np.fill_diagonal(x,0)
        a=sparse.csr_matrix(x)
        b,p=permute_graph(a,.5,77);c,q=permute_graph(a,.5,77)
        self.assertTrue(np.array_equal(p,q))
        self.assertTrue(np.allclose(b.toarray(),c.toarray()))
        self.assertTrue(np.allclose(np.linalg.eigvalsh(x),np.linalg.eigvalsh(b.toarray())))
        self.assertTrue(np.allclose(np.sort(a.data),np.sort(b.data)))
        d,p=permute_graph(a,0,77);self.assertTrue(np.array_equal(d.toarray(),x))

    def test_missing_modalities_cannot_enter_negative_samples(self):
        batch=[torch.randn(8,3) for _ in range(3)]+[torch.arange(8)]
        b=active_inputs(batch,['brain','text'])
        self.assertEqual(set(b),{'brain','text'})
        n=negative_pairs(b)
        self.assertEqual(len(n),2)
        self.assertTrue(all(set(v)=={'brain','text'} for v in n))
        self.assertEqual(negative_pairs(active_inputs(batch,['brain'])),[])

if __name__=='__main__':unittest.main()
