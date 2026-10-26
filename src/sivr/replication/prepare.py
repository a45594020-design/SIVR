"""Refit all preprocessing inside the class fold from raw responses/images."""
import csv,json,sys,os,re
from pathlib import Path
import numpy as np
import h5py,torch
from scipy.io import loadmat
from .common import ROOT,CACHE,DATA,sha,write
torch.set_num_threads(4)
ROIS=['ROI_V1','ROI_V2','ROI_V3','ROI_V4','ROI_LOC','ROI_FFA','ROI_PPA']

def pca(x,fit,extra=None):
 # Exact sample Gram eigendecomposition, not randomized PCA; all fitted rows explicit.
 x=np.asarray(x,dtype=np.float64);mu=x[fit].mean(0);sd=x[fit].std(0,ddof=1)+1e-8
 z=(x-mu)/sd;zm=z[fit].mean(0)
 a=torch.from_numpy(z[fit]-zm).cuda();gram=a@a.T
 vals,vecs=torch.linalg.eigh(gram);order=torch.arange(len(vals)-1,-1,-1,device='cuda')
 vals=vals[order].clamp_min(0);vecs=vecs[:,order]
 r=int(torch.searchsorted(vals.cumsum(0)/vals.sum(),torch.tensor(.99,device='cuda',dtype=vals.dtype)).item()+1)
 r=min(r,len(fit)-1);comp=(vecs[:,:r].T@a/vals[:r].sqrt()[:,None]).cpu().numpy()
 result=(z-zm)@comp.T
 ext=None if extra is None else ((np.asarray(extra,dtype=np.float64)-mu)/sd-zm)@comp.T
 return result,ext,{'components':comp,'mean':zm,'raw_mean':mu,'raw_std':sd,'fit_rows':fit,'explained_fraction':np.array(float(vals[:r].sum()/vals.sum()))}

def extract():
 out=CACHE/'raw_visual';out.mkdir(exist_ok=True)
 if (out/'complete.json').exists():return json.loads((out/'complete.json').read_text())
 import timm
 from PIL import Image
 from torchvision import transforms
 from torch.utils.data import DataLoader,Dataset
 # Both modules have independently loaded the same pretrained checkpoint.
 from safetensors.torch import load_file
 weight_path=CACHE/'repvgg_model.safetensors'
 state=load_file(str(weight_path))
 model=timm.create_model('repvgg_b3g4',pretrained=False);model.load_state_dict(state,strict=True);model=model.cuda().eval()
 mult=timm.create_model('repvgg_b3g4',pretrained=False,features_only=True)
 feature_state={re.sub(r'^stages\.(\d+)\.',r'stages_\1.',k):v for k,v in state.items() if not k.startswith('head.')}
 mult.load_state_dict(feature_state,strict=True)
 mult=mult.cuda().eval()
 tx=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])
 train=sorted((CACHE/'images/training').glob('*.JPEG'))
 test=sorted(Path(os.environ.get('SIVR_TEST_IMAGES', str(DATA/'images/test'))).glob('*.JPEG'))
 assert len(train)==1200 and len(test)==50
 paths=train+test
 class Images(Dataset):
  def __len__(self):return len(paths)
  def __getitem__(self,i):
   with Image.open(paths[i]) as im:return tx(im.convert('RGB'))
 arrays={};pos=0
 with torch.inference_mode():
  for batch in DataLoader(Images(),batch_size=8,num_workers=2):
   batch=batch.cuda();layers=mult(batch)
   pooled=model.forward_head(model.forward_features(batch),pre_logits=True)
   logits=model.get_classifier()(pooled)
   vals={'final':logits,'linear':pooled,**{'Conv_'+str(i):layers[i] for i in [2,3,4]}}
   for key,val in vals.items():
    val=val.flatten(1).cpu().numpy()
    if key not in arrays:arrays[key]=np.lib.format.open_memmap(out/(key+'.npy'),mode='w+',dtype='float32',shape=(1250,val.shape[1]))
    arrays[key][pos:pos+len(batch)]=val;arrays[key].flush()
   pos+=len(batch)
   if pos%80==0:print('visual extraction',pos,'/1250',flush=True)
 weights=Path(CACHE/'repvgg_pretrained.pt');torch.save(model.cpu().state_dict(),weights)
 meta={'timm_version':timm.__version__,'model':'repvgg_b3g4','pretrained_config':str(model.pretrained_cfg),'weights_sha256':sha(weights),'transform':'Resize256 CenterCrop224 ToTensor; matches repository transform, no added normalization','paths':[str(p) for p in paths],'names':[p.name for p in paths],'image_sha256':[sha(p) for p in paths],'layers':{k:list(v.shape) for k,v in arrays.items()}}
 write(out/'complete.json',meta);return meta

def main():
 ROOT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
 meta=extract();tr=loadmat(DATA/'brain_feature/VC/sub-04/fmri_train_data.mat')
 te=loadmat(DATA/'brain_feature/VC/sub-04/fmri_test_data.mat')
 labels=np.floor(tr['WordNetID'].ravel()).astype('int64');test_labels=np.floor(te['WordNetID'].ravel()).astype('int64')
 classes=np.unique(labels);assert len(classes)==150 and not set(classes)&set(test_labels)
 rng=np.random.default_rng(20261003);val_classes=np.sort(rng.permutation(classes)[:30]);fit=np.flatnonzero(~np.isin(labels,val_classes));val=np.flatnonzero(np.isin(labels,val_classes))
 write(ROOT/'class_split.json',{'seed':20261003,'fit_rows':fit.tolist(),'validation_rows':val.tolist(),'fit_wordnet_classes':np.unique(labels[fit]).tolist(),'validation_wordnet_classes':val_classes.tolist(),'final_classes_disjoint':True})
 names={n:i for i,n in enumerate(meta['names'])}
 def indices(ids):
  # Source IDs encode category.integer and image-index fractional part.
  result=[]
  for x in ids:
   wn=int(x);im=int(round((x-wn)*1e6));name=f'n{wn:08d}_{im}.JPEG';result.append(names[name])
  return np.array(result)
 train_idx=indices(tr['WordNetID'].ravel());test_idx=indices(te['WordNetID'].ravel())
 h5=Path(os.environ.get('SIVR_GOD_H5', str(DATA.parent/'GenericObjectDecoding-v2/Subject4.h5')))
 with h5py.File(h5) as f:
  keys=[x.decode().rstrip('\x00') for x in f['metadata/key'][()].ravel()];v=f['metadata/value'][()]
  cols=np.concatenate([np.flatnonzero(v[keys.index(k)]==1) for k in ROIS]);uniq,inv=np.unique(cols,return_inverse=True)
  # h5py fancy indexing needs sorted unique columns; re-expand legacy duplicates.
  raw=f['dataset'][:2950,uniq][:,inv]
  coords=np.stack([v[keys.index(k),uniq] for k in ['voxel_x','voxel_y','voxel_z']],1)
 rawtrain=raw[:1200];rawtest=raw[1200:2950];first=np.array([np.flatnonzero(inv==i)[0] for i in range(len(uniq))])
 texttrain=loadmat(DATA/'textual_feature/ImageNetTraining/text/GPTNeo/sub-04/text_feat_train.mat')['data']
 texttest=loadmat(DATA/'textual_feature/ImageNetTest/text/GPTNeo/sub-04/text_feat_test.mat')['data']
 for phase,rows in [('validation',fit),('final',np.arange(1200))]:
  out=CACHE/phase;out.mkdir(exist_ok=True)
  if (out/'complete.json').exists():continue
  brain,brain_test,model=pca(rawtrain,rows,rawtest if phase=='final' else None)
  np.savez_compressed(out/'brain_pca.npz',**model,legacy_to_unique=inv,first_unique=first,coordinates=coords,source_columns=cols,unique_source_columns=uniq)
  rawz=(rawtrain-model['raw_mean'])/model['raw_std'];np.save(out/'fit_unique_z.npy',rawz[rows][:,first]);images=[];extra=[]
  for layer in ['final','linear','Conv_4','Conv_3','Conv_2']:
   feature=np.load(CACHE/'raw_visual'/f'{layer}.npy',mmap_mode='r')
   y,yt,ymodel=pca(feature[train_idx],rows,feature[test_idx] if phase=='final' else None)
   np.savez_compressed(out/f'visual_pca_{layer}.npz',**ymodel);images.append(y)
   if yt is not None:extra.append(yt)
   print('PCA',phase,layer,y.shape,flush=True)
  image=np.concatenate(images,1)
  if phase=='validation':
   np.savez_compressed(out/'data.npz',train_brain=brain[fit],train_image=image[fit],train_text=texttrain[fit],train_label=labels[fit],eval_brain=brain[val],eval_image=image[val],eval_text=texttrain[val],eval_label=labels[val],stimulus_id=tr['WordNetID'].ravel()[val])
  else:
   np.savez_compressed(out/'data.npz',train_brain=brain,train_image=image,train_text=texttrain,train_label=labels,eval_brain=brain_test,eval_image=np.concatenate(extra,1),eval_text=texttest,eval_label=test_labels,stimulus_id=te['WordNetID'].ravel())
  write(out/'complete.json',{'phase':phase,'brain_dim':brain.shape[1],'image_dim':image.shape[1],'text_dim':texttrain.shape[1],'fitted_rows':rows.tolist(),'all_preprocessing_fit_only':True,'data_sha256':sha(out/'data.npz'),'brain_pca_sha256':sha(out/'brain_pca.npz')})
 with (ROOT/'node_coordinate_map.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['legacy_node','h5_column','unique_node','roi','x_native','y_native','z_native'])
  roi=np.concatenate([np.repeat(k,np.sum(v[keys.index(k)]==1)) for k in ROIS])
  for i,c in enumerate(cols):w.writerow([i,int(c),int(inv[i]),roi[i],*coords[inv[i]]])
 write(ROOT/'data_manifest.json',{'subject':'GOD sub04','source_h5':str(h5),'source_sha256':sha(h5),'legacy_dim':len(cols),'unique_nodes':len(uniq),'roi_duplicates':len(cols)-len(uniq),'coordinate_unit':'native numeric units; mm and world frame undocumented','visual_source':meta,'original_test_rows':1750,'independent_test_stimuli':50,'preprocessing':'refit every fold from original responses and frozen visual features','text_source':'unstandardized frozen GPTNeo per-class embeddings; no fold-dependent fitting','augmentation':'none, common to all seven groups','candidate_gradient_training':'fit-category image/text only','protocol_difference_from_old_paper':'no external augmentation, no heldout-candidate gradient training, newly extracted frozen visual weights version recorded'})
 write(ROOT/'prepare_complete.json',{'status':'complete','validation':json.loads((CACHE/'validation/complete.json').read_text()),'final':json.loads((CACHE/'final/complete.json').read_text())})
 print('PREPARATION COMPLETE',flush=True)
if __name__=='__main__':main()
