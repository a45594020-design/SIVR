"""Complete supervision removal; absent modalities never enter any objective."""
import itertools
import torch

ORDER=['brain','image','text']
def active_inputs(batch,modalities):return {k:batch[ORDER.index(k)] for k in modalities}

def negative_pairs(inputs):
    if len(inputs)<2:return []
    n=len(next(iter(inputs.values())));result=[]
    # Two directed mismatch cases for BV/BT. BVT retains legacy six triple cases.
    for left,right in itertools.permutations(inputs,2):
        ix=torch.randperm(n)
        different=(inputs[left][ix]!=inputs[left]).reshape(n,-1).any(1)
        if different.any():result.append({left:inputs[left][ix][different],right:inputs[right][different]})
    return result

def install_training(legacy,modalities):
    def mcubo(x,q,px,z):
        prior=torch.distributions.Normal(torch.zeros(1,z.size(2),device=z.device),torch.ones(1,z.size(2),device=z.device))
        value=prior.log_prob(z).sum(-1)-q.log_prob(z).sum(-1)
        value=value+sum(px[k].log_prob(x[k]).sum(-1) for k in x)
        return (.5*legacy.log_mean_exp(2*value)).mean()
    legacy.m_cubo=mcubo
    def train(epoch,exp,logger):
        exp.modalities={k:v for k,v in exp.modalities.items() if k in modalities}
        exp.mm_vae.modalities=exp.modalities;exp.mm_vae.num_modalities=len(modalities)
        exp.mm_vae.subsets={k:v for k,v in exp.mm_vae.subsets.items() if all(m.name in modalities for m in v)}
        exp.mm_vae.train()
        semantic=[m for m in modalities if m!='brain']
        stages=[]
        if semantic:stages.append((semantic,4000+epoch))
        stages.append((modalities,5000+epoch))
        for mods,offset in stages:
            for batch in legacy.make_loader(exp,exp.dataset_train,exp.flags.batch_size,True,False,offset):
                inputs=active_inputs(batch,mods)
                qloss=legacy.update_Qnet(exp,{0:dict(inputs)})
                for opt in exp.optimizer.values():opt.zero_grad(set_to_none=True)
                legacy.finite_backward_step(qloss,exp.optimizer['Qnet'],itertools.chain(exp.Q1.parameters(),exp.Q2.parameters(),exp.Q3.parameters()),'Qnet active')
                result=legacy.basic_routine_epoch(exp,{0:dict(inputs)},epoch)
                negatives=negative_pairs(inputs)
                inter=result['elbo_loss']+sum(legacy.elbo_contrast(exp,{0:neg},epoch)['elbo_nega_sample_loss'] for neg in negatives)/len(negatives) if negatives else result['elbo_loss']*0
                base=result['base_loss']+exp.flags.lambda2*inter
                loss=result['total_loss']+exp.flags.lambda2*inter
                for opt in exp.optimizer.values():opt.zero_grad(set_to_none=True)
                if 'brain' in inputs:legacy.topology_diagnostics(exp,base,result['topology_loss'],'active modalities')
                legacy.finite_backward_step(loss,exp.optimizer['mvae'],exp.mm_vae.parameters(),'active mvae')
                # Missing modules may be registered to preserve shared initialization,
                # but they must have no gradient from complete training objectives.
                for absent in set(ORDER)-set(modalities):
                    for module in [exp.mm_vae.encoders[absent],exp.mm_vae.decoders[absent],getattr(exp,{'brain':'Q1','image':'Q2','text':'Q3'}[absent])]:
                        assert all(p.grad is None or not torch.count_nonzero(p.grad) for p in module.parameters()),absent
    legacy.train=train
