import argparse,json,os,sys,contextlib,io
from pathlib import Path
import numpy as np,torch
from torch.utils.data import TensorDataset
from .common import ROOT,BASE,CACHE,CODE,write,sha,digest
from .runtime import ExperimentPrior,setup_injection
from .auxiliary_eval import evaluate_auxiliary

def evaluate(source,out,cfg):
    torch.set_num_threads(4);torch.set_default_dtype(torch.float64)
    os.chdir(CODE);sys.path.insert(0,str(CODE))
    from brain_image_text.experiment import BrainImageText
    import argparse
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    old=json.loads((source/'metrics.json').read_text());meta=json.loads((source/'run_manifest.json').read_text())
    assert old['status']=='complete' and old['config']['epochs']==100
    assert meta['data_sha256']==sha(CACHE/'fold1/data.npz')
    assert old['config']['seed']==cfg['seed']
    if cfg['group'] in ['Loss','R']:assert old['lambda_actual']==cfg['lambda_actual']
    assert json.loads((source/'progress.json').read_text())['completed_epochs']==100
    state=torch.load(source/'final_model.pt',map_location='cuda',weights_only=False)
    flags=argparse.Namespace(**state['manifest']['training_flags']);flags.device=torch.device('cuda')
    data=np.load(CACHE/'fold1/data.npz')
    train=TensorDataset(*[torch.tensor(data['train_'+k],dtype=torch.long if k=='label' else torch.double) for k in ['brain','image','text','label']])
    def dataset(self):
        self.dataset_train=train;self.dataset_test=train;self.m1_dim=flags.m1_dim;self.m2_dim=flags.m2_dim;self.m3_dim=flags.m3_dim
    BrainImageText.set_dataset=dataset
    with contextlib.redirect_stdout(io.StringIO()):exp=BrainImageText(flags,'')
    exp.mm_vae.load_state_dict(state['mm_vae'],strict=True)
    exp.prior=ExperimentPrior('fold1',cfg,artifact_dir=out);exp.mm_vae.eval()
    aux=evaluate_auxiliary(exp,data,exp.prior,out,cfg)
    write(out/'configuration.json',cfg)
    write(out/'reuse_manifest.json',{'source':str(source),'source_model_sha256':sha(source/'final_model.pt'),'source_manifest_sha256':sha(source/'run_manifest.json'),'data_sha256':meta['data_sha256'],'epochs':100,'original_metrics_unchanged':True,'training_group_alias':{'Loss':'G' if cfg.get('mismatch',0)==0 else 'G-perm'}.get(cfg['group'],cfg['group'])})
    for file in ['final_model.pt','predictions_V.npz','predictions_T.npz','predictions_VT.npz','gradient_trajectory.jsonl']:
        target=out/file
        if not target.exists() and (source/file).exists():target.symlink_to(source/file)
    write(out/'progress.json',{'completed_epochs':100,'epochs':100,'reused':True})
    write(out/'metrics.json',dict(old,config=cfg,config_hash=digest(cfg),auxiliary=aux,reused=True,graph_statistics=exp.prior.graph_stats))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('out');p.add_argument('config');a=p.parse_args()
    evaluate(a.source,a.out,json.loads(Path(a.config).read_text()))
