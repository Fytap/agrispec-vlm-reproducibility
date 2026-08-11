#!/usr/bin/env python3
"""Eight-GPU nonsealed RGB semantic specialist training on specialist_train only."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,random,subprocess,sys
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

SOIL={0,1,97}; CROP={10000,10001,10002}; WEED={2}|set(range(20000,20012))|set(range(20100,20106)); KNOWN=SOIL|CROP|WEED
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def rows(p:Path):
 with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p:Path,fields,data):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(data)
def commit(root:Path):
 try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
 except Exception:return None
def semantic(v:np.ndarray)->np.ndarray:
 unknown=set(np.unique(v).tolist())-KNOWN
 if unknown:raise ValueError(f'Unknown iMap ids: {sorted(unknown)}')
 out=np.zeros(v.shape,dtype=np.int64);out[np.isin(v,list(CROP))]=1;out[np.isin(v,list(WEED))]=2;return out
def pad(a,x,y,s):
 c=a[y:min(a.shape[0],y+s),x:min(a.shape[1],x+s)]; b=s-c.shape[0];r=s-c.shape[1]
 return c if b==0 and r==0 else np.pad(c,((0,b),(0,r),(0,0)) if c.ndim==3 else ((0,b),(0,r)),mode='edge')
def parse():
 p=argparse.ArgumentParser();
 for n,t in [('--project-root',Path),('--data-root',Path),('--supervision-manifest',Path),('--config',Path),('--output-dir',Path)]:p.add_argument(n,type=t,required=True)
 p.add_argument('--run-id',required=True);p.add_argument('--seed',type=int,default=20260808);p.add_argument('--epochs',type=int,default=8);p.add_argument('--train-patches',type=int,default=16384);p.add_argument('--patch-size',type=int,default=128);p.add_argument('--batch-size-per-gpu',type=int,default=64);p.add_argument('--learning-rate',type=float,default=8e-4);p.add_argument('--physical-gpu-ids',default='0,1,2,3,4,5,6,7');return p.parse_args()
def main():
 a=parse();import torch,torch.distributed as dist;import torch.nn.functional as F
 from torch import nn;from torch.nn.parallel import DistributedDataParallel as DDP;from torch.utils.data import DataLoader,Dataset;from torch.utils.data.distributed import DistributedSampler;from torchvision.models import ResNet18_Weights,resnet18
 rank,world,local=(int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'));phys=[int(x) for x in a.physical_gpu_ids.split(',')]
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=','.join(map(str,phys)) or world!=len(phys):raise RuntimeError('Frozen GPU assignment mismatch')
 dist.init_process_group('nccl');torch.cuda.set_device(local);dev=torch.device(f'cuda:{local}')
 root,data,manifest,cfg,out=a.project_root.resolve(),a.data_root.resolve(),a.supervision_manifest.resolve(),a.config.resolve(),a.output_dir.resolve()
 exists=torch.tensor(int(rank==0 and out.exists()),device=dev);dist.broadcast(exists,0)
 if int(exists):raise FileExistsError(f'Refusing overwrite: {out}')
 allrows=rows(manifest)
 if any(r['split']=='sealed_test' for r in allrows):raise ValueError('Manifest includes sealed row')
 source=[r for r in allrows if r['split']=='specialist_train']
 if len(source)!=7050:raise ValueError(f'Expected 7050 specialist_train rows, got {len(source)}')
 crops=[r for r in source if int(r['crop_pixels'])>0];weeds=[r for r in source if int(r['weed_pixels'])>0]
 if not crops or not weeds:raise ValueError('Role coverage missing')
 def specs():
  z=[]
  for i in range(a.train_patches):
   role,ids,pool=('crop',CROP,crops) if i%2==0 else ('weed',WEED,weeds);rng=np.random.default_rng(a.seed*1000003+i);r=pool[int(rng.integers(len(pool)))]
   with Image.open(data/r['imap_file']) as h:v=np.asarray(h)
   pos=np.argwhere(np.isin(v,list(ids)));y,x=pos[int(rng.integers(len(pos)))];z.append({'patch_index':i,'sampling_role':role,'session_id':r['session_id'],'file':r['file'],'imap_file':r['imap_file'],'x0':int(np.clip(x-a.patch_size//2,0,max(0,v.shape[1]-a.patch_size))),'y0':int(np.clip(y-a.patch_size//2,0,max(0,v.shape[0]-a.patch_size)))})
  return z
 if rank==0:
  out.mkdir(parents=True,exist_ok=False);write(out/'train_source_manifest.tsv',['session_id','split','file','bytes','imap_file','crop_pixels','weed_pixels'],source);write(out/'train_semantic_patch_specs.tsv',['patch_index','sampling_role','session_id','file','imap_file','x0','y0'],specs())
 dist.barrier();sp=rows(out/'train_semantic_patch_specs.tsv')
 class DS(Dataset):
  def __len__(self):return len(sp)
  def __getitem__(self,i):
   r=sp[i]
   with Image.open(data/r['file']) as h:im=np.asarray(h.convert('RGB'))
   with Image.open(data/r['imap_file']) as h:target=semantic(np.asarray(h))
   im=pad(im,int(r['x0']),int(r['y0']),a.patch_size);target=pad(target,int(r['x0']),int(r['y0']),a.patch_size)
   x=torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1))).float().div_(255);x=(x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
   return x,torch.from_numpy(np.ascontiguousarray(target))
 class Net(nn.Module):
  def __init__(self):
   super().__init__();e=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1);self.s=nn.Sequential(e.conv1,e.bn1,e.relu);self.p,self.l1,self.l2,self.l3,self.l4=e.maxpool,e.layer1,e.layer2,e.layer3,e.layer4
   def b(i,o):return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(),nn.Conv2d(o,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU())
   self.d4,self.d3,self.d2,self.d1=b(768,256),b(384,128),b(192,64),b(128,64);self.h=nn.Conv2d(64,3,1)
  def forward(self,x):
   sh=x.shape[-2:];s0=self.s(x);s1=self.l1(self.p(s0));s2=self.l2(s1);s3=self.l3(s2);s4=self.l4(s3);up=lambda z,ref:F.interpolate(z,size=ref.shape[-2:],mode='bilinear',align_corners=False)
   d4=self.d4(torch.cat([up(s4,s3),s3],1));d3=self.d3(torch.cat([up(d4,s2),s2],1));d2=self.d2(torch.cat([up(d3,s1),s1],1));d1=self.d1(torch.cat([up(d2,s0),s0],1));return self.h(F.interpolate(d1,size=sh,mode='bilinear',align_corners=False))
 random.seed(a.seed+rank);np.random.seed(a.seed+rank);torch.manual_seed(a.seed+rank);torch.cuda.manual_seed_all(a.seed+rank)
 ds=DS();sam=DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=True,seed=a.seed);loader=DataLoader(ds,batch_size=a.batch_size_per_gpu,sampler=sam,num_workers=2,pin_memory=True,persistent_workers=True)
 model=DDP(Net().to(dev),device_ids=[local],broadcast_buffers=False);opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4);hist=[];weights=torch.tensor([.2,1.,1.],device=dev)
 for ep in range(1,a.epochs+1):
  sam.set_epoch(ep);model.train();total=torch.zeros(2,device=dev)
  for x,y in loader:
   x,y=x.to(dev,non_blocking=True),y.to(dev,non_blocking=True);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    log=model(x);ce=F.cross_entropy(log,y,weight=weights);pr=log.softmax(1);oh=F.one_hot(y,3).permute(0,3,1,2).float();dice=1-(2*(pr[:,1:]*oh[:,1:]).sum()+1)/(pr[:,1:].sum()+oh[:,1:].sum()+1);loss=ce+dice
   if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss epoch {ep}')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=torch.tensor([float(loss.detach()),1],device=dev)
  dist.all_reduce(total)
  if rank==0:r={'epoch':ep,'mean_train_loss':float(total[0]/total[1])};hist.append(r);print(json.dumps(r),flush=True)
 dist.barrier()
 if rank==0:
  ck=out/'last.pt';torch.save({'model_state':model.module.state_dict(),'epochs':a.epochs,'history':hist},ck);cache=Path(torch.hub.get_dir())/'checkpoints'/'resnet18-f37072fd.pth'
  s={'run_id':a.run_id,'status':'completed_nonsealed_ddp_semantic_specialist_training_no_validation_read','scope':{'public_data_only':True,'sealed_test_read':False,'specialist_val_read':False,'controller_policy_risk_read':False,'train_split_only':'specialist_train','model_input':'RGB_only','iMap_use':'supervised_patch_selection_and_semantic_targets_only'},'inputs':{'supervision_manifest':str(manifest),'supervision_manifest_sha256':sha(manifest),'config':str(cfg),'config_sha256':sha(cfg),'train_source_manifest_sha256':sha(out/'train_source_manifest.tsv'),'train_patch_specs_sha256':sha(out/'train_semantic_patch_specs.tsv'),'encoder_weight_cache_sha256':sha(cache)},'model':{'architecture':'custom_resnet18_unet','classes':['soil','crop','weed']},'run_parameters':{'seed':a.seed,'epochs':a.epochs,'train_patches':a.train_patches,'patch_size':a.patch_size,'batch_size_per_gpu':a.batch_size_per_gpu,'world_size':world,'physical_gpu_ids':phys,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')},'history':hist,'checkpoint_sha256':sha(ck),'limitations':['This creates a semantic specialist, not controller claims or a risk result.','No validation, controller, policy, risk or sealed data was read.'],'script_sha256':sha(Path(__file__).resolve()),'git_commit':commit(root)};(out/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2),flush=True)
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
