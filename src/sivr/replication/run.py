"""Isolated BraVL objective runner with train-only preprocessing and fail-fast gates."""
import argparse,json,os,sys,random,time,contextlib,io,hashlib,shutil
from pathlib import Path
import numpy as np,torch
from torch.utils.data import TensorDataset
from sklearn.svm import SVC
from .common import ROOT,CACHE,CODE,DATA,write,digest,sha
from .priors import Prior

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def flags_defaults():
 # Read argparse declarations without executing the legacy file's eager MAT fitting/load.
 scope={'os':os,'__name__':'x3_flags'}
 src=(CODE/'utils/BaseFlags.py').read_text().split('FLAGS = parser.parse_args()')[0]
 exec(compile(src,str(CODE/'utils/BaseFlags.py'),'exec'),scope)
 src=(CODE/'brain_image_text/flags.py').read_text().replace('from utils.BaseFlags import parser as parser','')
 exec(compile(src,str(CODE/'brain_image_text/flags.py'),'exec'),scope)
 return scope['parser'].parse_args([])

class NullLogger:
 def write_training_logs(self,*a):pass

def grad_summary(loss,params):
 return torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
def grad_norm(gs):
 values=[g.detach().square().sum() for g in gs if g is not None]
 return torch.stack(values).sum().sqrt() if values else torch.zeros((),device='cuda')


def wait_for_disk_space(out):
 while shutil.disk_usage(out).free < 2*2**30:
  write(Path(out)/'disk_wait.json',{'stage':'waiting_for_disk','minimum_free_GiB':2,'free_bytes':shutil.disk_usage(out).free,'unix':time.time()})
  print('Waiting for disk space; no training writes',flush=True);time.sleep(30)

def run(cfg,out,phase,calibrate=False):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if phase=='final' and not calibrate and not (ROOT/'frozen_configuration.json').exists():raise RuntimeError('Final evaluation forbidden before configuration freeze')
 if (out/'metrics.json').exists():
  m=json.loads((out/'metrics.json').read_text());assert m['config_hash']==digest(cfg)
  manifest=json.loads((out/'run_manifest.json').read_text());assert manifest['data_sha256']==sha(CACHE/phase/'data.npz')
  return m
 wait_for_disk_space(out)
 torch.set_num_threads(4);torch.set_default_dtype(torch.float64);seed_all(cfg['seed']);os.chdir(CODE);sys.path.insert(0,str(CODE))
 import run_epochs_trimodal as legacy
 from brain_image_text.experiment import BrainImageText
 seed_all(cfg['seed']) # after all legacy imports
 data=np.load(CACHE/phase/'data.npz');train=TensorDataset(*[torch.tensor(data['train_'+k],dtype=torch.long if k=='label' else torch.double) for k in ['brain','image','text','label']])
 flags=flags_defaults();flags.m1_dim=train.tensors[0].shape[1];flags.m2_dim=train.tensors[1].shape[1];flags.m3_dim=train.tensors[2].shape[1]
 flags.device=torch.device('cuda');flags.seed=cfg['seed'];flags.num_workers=0;flags.joint_elbo=True;flags.aug_type='no_aug';flags.dataname='GOD-Wiki';flags.sbj='sub-04';flags.roi='VC'
 flags.alpha_modalities=[.25]*4;flags.lambda_topo=0.;flags.data_dir_root=str(DATA);flags.results_dir=str(out);flags.dir_checkpoints=str(out/'recovery');flags.resume_checkpoint='';flags.auto_resume=True;flags.start_epoch=0
 # Constructor installs only current fitted rows. No heldout candidates in gradient updates.
 def set_dataset(self):
  self.dataset_train=train;self.dataset_test=train;self.m1_dim=flags.m1_dim;self.m2_dim=flags.m2_dim;self.m3_dim=flags.m3_dim
 BrainImageText.set_dataset=set_dataset
 exp=BrainImageText(flags,'')
 h=hashlib.sha256()
 for name,tensor in sorted(exp.mm_vae.state_dict().items()):
  h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
 write(out/'initialization.json',{'seed':cfg['seed'],'backbone_sha256':h.hexdigest(),'reseeded_after_imports':True})
 if cfg.get('init_only'):return {'initialization_only':True}
 if cfg.get('replay_checkpoint'):
  saved=torch.load(cfg['replay_checkpoint'],map_location='cuda');exp.mm_vae.load_state_dict(saved['mm_vae'])
 exp.set_optimizer();exp.prior=Prior(phase,cfg,artifact_dir=out)
 params=list(exp.mm_vae.parameters());actual=float(cfg.get('lambda_actual',0));calibrated=cfg['group']=='B0' or 'lambda_actual' in cfg
 records=[];pending=None;epoch=0
 def prior_hook(e,results):
  nonlocal pending
  zero=torch.zeros((),device='cuda')
  if 'brain' not in results['rec']:return zero,zero
  pc=results['rec']['brain'].loc;pending=pc
  raw=e.prior.loss(pc)
  return actual*raw,raw
 def diag_hook(e,base,raw,label):
  nonlocal actual,calibrated
  if pending is None:return
  if cfg['group']=='B0':
   if epoch==0 or (epoch+1)%10==0:
    with torch.no_grad():u=exp.prior.physical(pending)
    with (out/'gradient_trajectory.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch+1,'group':'B0','output_variance':float(u.var(0,unbiased=False).mean()),'output_mean_square':float(u.square().mean()),'weighted_prior_to_task_gradient':0.})+'\n')
   return
  if calibrated and not (epoch==0 or (epoch+1)%10==0):return
  gt=grad_summary(base,params);gp=grad_summary(raw,params);tn=float(grad_norm(gt));pn=float(grad_norm(gp))
  if not np.isfinite([tn,pn]).all() or pn<=0 or tn<=0:raise RuntimeError('Invalid task/prior gradient')
  if not calibrated:
   actual=float(cfg['target_grad_ratio'])*tn/pn;calibrated=True
   write(out/'calibration.json',{'lambda_actual':actual,'target_gradient_ratio':cfg['target_grad_ratio'],'task_gradient_norm':tn,'prior_gradient_norm':pn,'fitted_data_only':True,'phase':phase,'seed':cfg['seed'],'graph_statistics':exp.prior.graph_stats})
  dot=sum(float((a.detach()*b.detach()).sum()) for a,b in zip(gt,gp) if a is not None and b is not None)
  with torch.no_grad():u=exp.prior.physical(pending)
  modnorm={}
  param_index={id(p):i for i,p in enumerate(params)}
  for name,mod in {**{'enc_'+k:v for k,v in exp.mm_vae.encoders.items()},'dec_brain':exp.mm_vae.decoders['brain']}.items():
   gs=[gp[param_index[id(p)]] for p in mod.parameters()];modnorm[name]=float(grad_norm(gs))
  assert all(np.isfinite(v) and v>0 for v in modnorm.values()),modnorm
  record={'epoch':epoch+1,'lambda_actual':actual,'task_gradient_norm':tn,'prior_gradient_norm':pn,'weighted_prior_to_task_gradient':actual*pn/tn,'cosine':dot/(tn*pn),'output_variance':float(u.var(0,unbiased=False).mean()),'output_mean_square':float(u.square().mean()),'loss':float(raw.detach()),'module_prior_gradient_norms':modnorm}
  records.append(record)
  with (out/'gradient_trajectory.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 legacy.calc_spatial_regularization=prior_hook;legacy.topology_diagnostics=diag_hook
 # Original routine computes the total before diagnostic calibration. Recompute its
 # prior term after calibration on the first call, before any optimizer update.
 original_basic=legacy.basic_routine_epoch
 def basic(e,batch,ep):
  nonlocal pending
  result=original_basic(e,batch,ep)
  if cfg['group']!='B0' and 'brain' in batch[0] and not calibrated:
   diag_hook(e,result['base_loss'],result['topology_loss'],'initial calibration')
   result['total_loss']=result['base_loss']+actual*result['topology_loss']
  return result
 legacy.basic_routine_epoch=basic
 def strict_step(loss,optimizer,parameters,label):
  parameters=list(parameters)
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss '+label)
  loss.backward()
  if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):raise FloatingPointError('Nonfinite gradient '+label)
  torch.nn.utils.clip_grad_norm_(parameters,5.);optimizer.step();return True
 legacy.finite_backward_step=strict_step
 # Keep every fitted row, including the tail; norm_value remains the original
 # batch512 constant in the legacy objective and is shared across all groups.
 original_loader=legacy.make_loader
 def loader(e,dataset,batch_size,shuffle=True,drop_last=True,seed_offset=0):return original_loader(e,dataset,batch_size,shuffle,False,seed_offset)
 legacy.make_loader=loader
 meta={'config':cfg,'config_hash':digest(cfg),'phase':phase,'data_sha256':sha(CACHE/phase/'data.npz'),'preprocessing_sha256':sha(CACHE/phase/'brain_pca.npz'),'graph_statistics':exp.prior.graph_stats,'train_rows':len(train),'batch_size':512,'tail_rows_retained':True,'candidate_gradient_training':'fit rows only','no_external_augmentation':True,'training_flags':{k:str(v) if isinstance(v,(Path,torch.device)) else v for k,v in vars(flags).items()},'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},'model_parameters':sum(p.numel() for p in params),'script_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}}
 write(out/'run_manifest.json',meta)
 # Validation checkpoints are resumed independently; finished candidates discarded
 # only their transient recovery checkpoint, while final model state is retained.
 os.makedirs(flags.dir_checkpoints,exist_ok=True)
 start=legacy.load_training_state_if_requested(exp)
 if start>0:
  cal=json.loads((out/'calibration.json').read_text()) if (out/'calibration.json').exists() else {}
  actual=cal.get('lambda_actual',actual);calibrated=True
 tic=time.time()
 for epoch in range(start,1 if calibrate else cfg.get('epochs',100)):
  wait_for_disk_space(out)
  with contextlib.redirect_stdout(io.StringIO()):legacy.train(epoch,exp,NullLogger())
  if (epoch+1)%5==0 and not calibrate:legacy.atomic_save_training_state(exp,epoch)
  write(out/'progress.json',{'completed_epochs':epoch+1,'epochs':cfg.get('epochs',100),'elapsed_seconds':time.time()-tic,'lambda_actual':actual,'config_hash':digest(cfg)})
  print(out.name,'epoch',epoch+1,'lambda',actual,'seconds',round(time.time()-tic,1),flush=True)
 if calibrate:
  write(out/'calibration_complete.json',{'lambda_actual':actual,'config_hash':digest(cfg)});return
 exp.mm_vae.eval();metrics={}
 # Candidate semantics are known at evaluation, but no brain response is used in
 # the reference classifier and no candidate gradient enters model training.
 evals={k:torch.tensor(data['eval_'+k],dtype=torch.double,device='cuda') for k in ['brain','image','text']};y=data['eval_label'].ravel();stim=data['stimulus_id'].ravel()
 unique,first=np.unique(stim,return_index=True);refy=y[first]
 for i,(space,mods) in enumerate([('V',['image']),('T',['text']),('VT',['image','text'])]):
  seed_all(cfg['seed']+50000+i)
  with torch.no_grad():
   ref=[]
   # Preserve the legacy five draws over the full candidate semantic row list,
   # including repeated references, so SVC's effective C/sample weight is fixed.
   # These are not independent observations in the inferential statistics.
   for _ in range(5):
    part=[]
    for lo in range(0,len(y),200):part.append(exp.mm_vae({k:evals[k][lo:lo+200] for k in mods})['class_embeddings'].cpu().numpy())
    ref.append(np.vstack(part))
   # Brain inference in chunks prevents test-size memory-dependent failures.
   z=[]
   for lo in range(0,len(y),200):z.append(exp.mm_vae({'brain':evals['brain'][lo:lo+200]})['class_embeddings'].cpu().numpy())
  clf=SVC(gamma=1e-5,C=1.,probability=True,random_state=cfg['seed']);clf.fit(np.vstack(ref),np.tile(y,5));z=np.vstack(z)
  pred=clf.predict(z);prob=clf.predict_proba(z);top5=clf.classes_[np.argsort(prob,axis=1)[:,-5:]]
  metrics[space]={'top1':float(np.mean(pred==y)),'top5':float(np.mean(np.any(top5==y[:,None],1)))}
  np.savez_compressed(out/f'predictions_{space}.npz',y_true=y,top1=pred,top5=top5,probabilities=prob,class_order=clf.classes_,stimulus_id=stim,reference_stimulus_id=unique)
 # Store final models only for confirmatory runs; optimizer/RNG for interrupted
 # run recovery remains a single checkpoint at a time, not 35 full optimizers.
 if phase=='final' and not cfg.get('replay_checkpoint'):torch.save({'mm_vae':exp.mm_vae.state_dict(),'Q1':exp.Q1.state_dict(),'Q2':exp.Q2.state_dict(),'Q3':exp.Q3.state_dict(),'manifest':meta},out/'final_model.pt')
 payload={'status':'complete','config_hash':digest(cfg),'config':cfg,'lambda_actual':actual,'phase':phase,'metrics':metrics,'elapsed_seconds':time.time()-tic,'graph_statistics':exp.prior.graph_stats}
 write(out/'metrics.json',payload)
 recovery=Path(flags.dir_checkpoints)/'training_state_latest.pt'
 if recovery.exists():recovery.unlink() # Only this runner's redundant transient checkpoint.
 return payload

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--phase',choices=['validation','final'],required=True);p.add_argument('--calibrate',action='store_true');args=p.parse_args()
 run(json.loads(Path(args.config).read_text()),args.out,args.phase,args.calibrate)
