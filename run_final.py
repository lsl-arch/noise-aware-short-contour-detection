"""Frozen Lesson 5 experiment. Use --output for a new run; never tunes parameters."""
from pathlib import Path
import sys,json,hashlib,argparse,time,platform,csv
from collections import defaultdict
import numpy as np
from PIL import Image
import method as m
from baselines import Baselines,canny
from data_v6 import Config,_geometry,_noise,FAMILIES,SHAPES,CONTRASTS,SIGMAS
ROOT=Path(__file__).resolve().parent

def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fresh_cases(null=False):
 cfg=Config()
 for si,shape in enumerate(['none'] if null else SHAPES):
  for ci,contrast in enumerate([0.] if null else CONTRASTS):
   for fi,family in enumerate(FAMILIES):
    for rep in range(64 if null else 4):
     key=f'lesson5-final-20261105:{shape}:{contrast}:{family}:{rep}'
     seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
     sigma=SIGMAS[(si*4+ci+fi+rep)%4]
     clean,gt,geometry=_geometry(shape,contrast,seed,cfg,null)
     image,_,_=_noise(clean,sigma,family,seed^0x9E3779B97F4A7C15,cfg)
     yield image,gt,dict(case_id=key,scene=shape,noise_family=family,contrast=contrast,sigma=sigma,seed=seed,geometry=geometry,input_sha256=hashlib.sha256(image.tobytes()).hexdigest())

def natural_input(path,factor):
 rgb=np.asarray(Image.open(path).convert('RGB'),np.float32)/255;gray=rgb@np.array([.299,.587,.114],np.float32)
 clean=(factor*gray**1.5).astype(np.float32)
 # Same declared noise construction as development; new images, fixed conditions.
 seed=int.from_bytes(hashlib.sha256(('lesson3-bsds-lowlight-v1:'+path.stem).encode()).digest()[:8],'little')
 im=(clean+np.random.default_rng(seed).standard_normal(clean.shape)*np.sqrt(.004**2+.0006*clean)).astype(np.float32)
 return im

def predictions(im,configs):
 t=time.perf_counter();f=m.features(im,coarse_high=2.5);common=time.perf_counter()-t;out={};times={}
 for k,c in configs.items():
  t=time.perf_counter()
  if k=='no_noise':g=m.features(im,alpha=0,coarse_high=2.5);out[k]=m.predict_features(g,c);times[k]=time.perf_counter()-t
  else:out[k]=m.predict_features(f,c);times[k]=common+time.perf_counter()-t
 return out,times

def main():
 p=argparse.ArgumentParser();p.add_argument('--bsds-root',type=Path,default=ROOT/'data/BSDS500');p.add_argument('--output',type=Path,default=ROOT/'results/final');args=p.parse_args();out=args.output
 if (out/'summary.json').exists():raise SystemExit('此目录已有完整结果；请使用--output指定新目录。')
 protocol=json.loads((ROOT/'frozen_protocol.json').read_text())
 for name,h in protocol['source_sha256'].items():
  if sha(ROOT/name)!=h:raise RuntimeError('冻结源码校验失败:'+name)
 configs=protocol['internal_methods'];thresholds=protocol['baseline_thresholds'];out.mkdir(parents=True,exist_ok=True);save(out/'protocol.json',protocol)
 start=time.perf_counter();rows=[];manifest=[];summary={}
 for null in [False,True]:
  stage='fresh_null384' if null else 'fresh_synthetic384';bucket=defaultdict(list)
  for j,(im,gt,meta) in enumerate(fresh_cases(null)):
   preds,times=predictions(im,configs);manifest.append(dict(stage=stage,**meta))
   for name,e in preds.items():
    vals={'predicted':int(e[13:-13,13:-13].sum())} if null else m.synthetic_metrics(e,gt)
    row=dict(stage=stage,method=name,**meta,**vals,seconds=times[name]);rows.append(row);bucket[name].append(row)
   # Full prediction masks and exact input for all samples, compact NPZ.
   safe=hashlib.sha256(meta['case_id'].encode()).hexdigest()[:16];d=out/'predictions'/stage;d.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(d/(safe+'.npz'),input=im,truth=gt,**preds)
   if (j+1)%48==0:print(stage,j+1,flush=True)
  summary[stage]={k:(m.null_aggregate(v) if null else m.synth_aggregate(v)) for k,v in bucket.items()};save(out/'summary_partial.json',summary)
 base=Baselines();print('Deep device',base.device,flush=True)
 test=sorted((args.bsds_root/'images/test').glob('*.jpg'))
 if len(test)!=200:raise RuntimeError(f'需要BSDS test200图，找到{len(test)}')
 for factor,stage in [(.25,'bsds_test200_low'),(.10,'bsds_test200_darker')]:
  bucket=defaultdict(list)
  for j,path in enumerate(test):
   im=natural_input(path,factor);gts=m.truths(args.bsds_root/'groundTruth/test'/(path.stem+'.mat'));preds,times=predictions(im,configs)
   maps,tm=base.responses(im)
   for name in ['Canny','CLAHE_Canny']:
    t=time.perf_counter();preds[name]=canny(im,thresholds[name],name=='CLAHE_Canny');times[name]=time.perf_counter()-t
   for name,mp in maps.items():preds[name]=mp>=thresholds[name];times[name]=tm[name]
   for name,e in preds.items():
    ar=[m.evaluate(e,g,border=13) for g in gts]
    row=dict(stage=stage,method=name,image_id=path.stem,seconds=times[name],annotators=len(gts),**{k:float(np.mean([v[k] for v in ar])) for k in ['precision','recall','f1']})
    rows.append(row);bucket[name].append(row)
   manifest.append(dict(stage=stage,image_id=path.stem,image_sha256=sha(path),annotation_sha256=sha(args.bsds_root/'groundTruth/test'/(path.stem+'.mat')),input_sha256=hashlib.sha256(im.tobytes()).hexdigest(),factor=factor))
   d=out/'predictions'/stage;d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/(path.stem+'.npz'),input=im,truth_union=np.logical_or.reduce(gts),**preds)
   if (j+1)%10==0:
    print(stage,j+1,'elapsed',round(time.perf_counter()-start,1),flush=True);save(out/'per_image_partial.json',rows)
  summary[stage]={k:dict(images=len(v),**{x:float(np.mean([r[x] for r in v])) for x in ['precision','recall','f1','seconds']}) for k,v in bucket.items()};save(out/'summary_partial.json',summary)
 save(out/'summary.json',summary);save(out/'per_image.json',rows);save(out/'data_manifest.json',manifest)
 save(out/'run_metadata.json',dict(elapsed_seconds=time.perf_counter()-start,platform=platform.platform(),python=sys.version,deep_device=base.device,protocol_sha256=sha(ROOT/'frozen_protocol.json'),record_count=len(rows)))
 fields=['stage','method','images','precision','recall','f1','short_recall','any_output_images','any_output_rate','seconds']
 with (out/'summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
  for stage,groups in summary.items():
   for name,v in groups.items():w.writerow(dict(stage=stage,method=name,**v))
 print('FINAL COMPLETE',out,flush=True)
if __name__=='__main__':main()
