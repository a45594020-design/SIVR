"""Frozen fit-only ridge/cosine protocol shared by all training modalities."""
import numpy as np
import torch
from sklearn.linear_model import Ridge
from scipy import sparse
from .priors import adjacency
from .common import write

def cosine_predictions(ztrain,zeval,target_train,target_eval,y,alpha=1.):
    mu=ztrain.mean(0);sd=ztrain.std(0);sd=np.maximum(sd,1e-8)
    reg=Ridge(alpha=alpha);reg.fit((ztrain-mu)/sd,target_train)
    pred=reg.predict((zeval-mu)/sd)
    labels,first=np.unique(y,return_index=True);candidates=target_eval[first]
    pred=pred/np.maximum(np.linalg.norm(pred,axis=1,keepdims=True),1e-12)
    candidates=candidates/np.maximum(np.linalg.norm(candidates,axis=1,keepdims=True),1e-12)
    score=pred@candidates.T;order=labels[np.argsort(score,axis=1)[:,::-1]]
    return {'top1':float(np.mean(order[:,0]==y)),'top5':float(np.mean(np.any(order[:,:5]==y[:,None],1))),'predicted_classes':int(len(np.unique(order[:,0]))),'largest_class_share':float(np.unique(order[:,0],return_counts=True)[1].max()/len(y))},order,score

def embeddings(exp,data,split,mods):
    values=[]
    with torch.no_grad():
        for lo in range(0,len(data[split+'_brain']),128):
            inputs={k:torch.tensor(data[split+'_'+k][lo:lo+128],dtype=torch.double,device='cuda') for k in mods}
            values.append(exp.mm_vae.inference(inputs)['joint'][0].cpu().numpy())
    return np.vstack(values)

def evaluate_auxiliary(exp,data,prior,out,cfg):
    ztrain=embeddings(exp,data,'train',['brain']);zeval=embeddings(exp,data,'eval',['brain'])
    result,order,scores=cosine_predictions(ztrain,zeval,data['train_text'],data['eval_text'],data['eval_label'].ravel())
    np.savez_compressed(out/'predictions_auxiliary.npz',y_true=data['eval_label'],top1=order[:,0],top5=order[:,:5],scores=scores,class_order=np.unique(data['eval_label']))
    np.savez_compressed(out/'auxiliary_latents.npz',train=ztrain,eval=zeval)
    rows=[];a=adjacency(prior.phase,dict(cfg,group='G',k=12,scale=2.));d=np.asarray(a.sum(1)).ravel();l=(sparse.diags(d)-a)/d.mean()
    for split,z in [('train',ztrain),('eval',zeval)]:
        rec=[]
        with torch.no_grad():
            for lo in range(0,len(z),128):rec.append(exp.mm_vae.decoders['brain'](None,torch.tensor(z[lo:lo+128],dtype=torch.double,device='cuda'))[0].cpu().numpy())
        rec=np.vstack(rec);truth=data[split+'_brain'];u=prior.physical(torch.tensor(rec,dtype=torch.double,device='cuda')).detach().cpu().numpy()
        if split=='eval':np.savez_compressed(out/'eval_brain_reconstruction.npz',pca=rec,unique=u)
        ac=rec-rec.mean(1,keepdims=True);bc=truth-truth.mean(1,keepdims=True)
        rows.append({'split':split,'brain_pca_mse':float(np.mean((rec-truth)**2)),'brain_pca_row_correlation':float(np.mean((ac*bc).sum(1)/np.maximum(np.linalg.norm(ac,axis=1)*np.linalg.norm(bc,axis=1),1e-12))),'unique_variance':float(np.var(u,axis=0).mean()),'true_graph_energy':float(np.mean(np.sum(u*(l@u.T).T,axis=1))/u.shape[1]),'latent_std':float(np.std(z,axis=0).mean()),'structure':exp.injection.last if hasattr(exp,'injection') else {}})
    write(out/'reconstruction_diagnostics.json',rows)
    if hasattr(exp,'injection'):
        from injection_models import permute_graph
        module=exp.injection;original=module.adj
        # Fixed posterior means and trained weights: change only adjacency.
        altered,_=permute_graph(a,1,cfg['graph_seed']+50000)
        altered=altered+sparse.eye(altered.shape[0]);degree=np.asarray(altered.sum(1)).ravel()**-.5
        altered=(sparse.diags(degree)@altered@sparse.diags(degree)).tocoo()
        module.adj=torch.sparse_coo_tensor(np.stack([altered.row,altered.col]),torch.tensor(altered.data,dtype=torch.double,device='cuda'),altered.shape).coalesce()
        with torch.no_grad():
            changed=[]
            for lo in range(0,len(zeval),128):changed.append(exp.mm_vae.decoders['brain'](None,torch.tensor(zeval[lo:lo+128],dtype=torch.double,device='cuda'))[0].cpu().numpy())
        module.adj=original;changed=np.vstack(changed)
        write(out/'graph_counterfactual.json',{'fixed_posterior_mean_and_weights':True,'alternative_graph_seed':cfg['graph_seed']+50000,'pca_output_change_rms':float(np.mean((changed-rec)**2)**.5),'classification_latent_changes':False,'meaning':'decoder graph sensitivity only, not retraining or task causality'})
    write(out/'auxiliary_metrics.json',result)
    return result
