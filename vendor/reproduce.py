"""Reproduce v7 locked synthetic data and seen BSDS development/regression."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, os, platform, sys, tarfile, time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from data_v7 import generate_cases
from bsds_support import read_input, truths, evaluate, fast_match
from 运行 import detect, METHOD
EXPECTED=json.loads((ROOT/'expected_metrics.json').read_text(encoding='utf-8'))
FREEZE=EXPECTED['frozen_protocol_sha256']

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def valid(shape):
    a=np.zeros(shape,bool);a[13:-13,13:-13]=True;return a

def synthetic_metrics(pred,gt):
    va=valid(pred.shape);pred=pred&va
    labels,_=ndi.label(gt,structure=np.ones((3,3),bool));sizes=np.bincount(labels.ravel())
    short=(labels>0)&(sizes[labels]<=32)&va
    ds,npred,ngt=fast_match(pred,gt&va,2.)
    st=len(fast_match(pred,short,2.)[0]) if short.any() else 0
    tp=len(ds);p=tp/npred if npred else 0.;r=tp/ngt if ngt else 0.
    return dict(tp=tp,predicted=npred,truth=ngt,short_tp=st,short_truth=int(short.sum()),precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0.)

def synth_aggregate(rows):
    tp=sum(x['tp'] for x in rows);p=sum(x['predicted'] for x in rows);g=sum(x['truth'] for x in rows)
    st=sum(x['short_tp'] for x in rows);sg=sum(x['short_truth'] for x in rows)
    return dict(images=len(rows),precision=tp/p if p else 0.,recall=tp/g if g else 0.,f1=2*tp/(p+g) if p+g else 0.,
                short_recall=st/sg if sg else None,tp=tp,predicted=p,truth=g,short_tp=st,short_truth=sg,
                macro_f1=float(np.mean([x['f1'] for x in rows])))

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den;d=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,c-d),min(1.,c+d)]

def null_aggregate(rows):
    n=len(rows);k=sum(x['predicted']>0 for x in rows)
    return dict(images=n,any_output_images=k,any_output_rate=k/n,wilson95=wilson(k,n),
                mean_predicted=float(np.mean([x['predicted'] for x in rows])),pixel_response_rate=sum(x['predicted'] for x in rows)/(n*102*102))

def synth_worker(item):
    cv2.setNumThreads(1)
    split,case=item;pred=detect(case['input'])
    row=dict(case['metadata'])
    row.update(synthetic_metrics(pred,case['truth']) if split=='locked_test' else dict(predicted=int((pred&valid(pred.shape)).sum())))
    return row

def bsds_worker(item):
    cv2.setNumThreads(1)
    path,truthpath=item;_,_,image,seed=read_input(Path(path));pred=detect(image)
    gt=truths(Path(truthpath));ar=[dict(annotator=i,**evaluate(pred,t,border=13)) for i,t in enumerate(gt)]
    return dict(image_id=Path(path).stem,input_seed=seed,annotators=len(gt),
                **{k:float(np.mean([r[k] for r in ar])) for k in ('precision','recall','f1')},annotation_rows=ar)

def map_rows(worker,items,workers,label):
    if workers==1:
        rows=[]
        for i,item in enumerate(items):
            rows.append(worker(item))
            if (i+1)%25==0 or i+1==len(items):print(f'{label}: {i+1}/{len(items)}',flush=True)
        return rows
    rows=[]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i,row in enumerate(executor.map(worker,items,chunksize=4)):
            rows.append(row)
            if (i+1)%25==0 or i+1==len(items):print(f'{label}: {i+1}/{len(items)}',flush=True)
    return rows

def locate_data(path):
    path=Path(path)
    for candidate in (path,path/'data',path/'BSDS500/data',path/'BSR/BSDS500/data'):
        if (candidate/'images/train').is_dir() and (candidate/'groundTruth/train').is_dir():return candidate
    raise ValueError('找不到BSDS500 data目录；应含images/train与groundTruth/train。')

def extract_bsds(archive,destination):
    """Extract ordinary files/directories only, rejecting traversal and links."""
    archive=Path(archive);destination=Path(destination).resolve()
    if not archive.is_file():raise FileNotFoundError(archive)
    destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tf:
        members=tf.getmembers()
        for member in members:
            target=(destination/member.name).resolve()
            if os.path.commonpath([str(destination),str(target)])!=str(destination):
                raise ValueError('归档包含超出目标文件夹的路径，已拒绝解压。')
            if not (member.isfile() or member.isdir()):
                raise ValueError('归档包含链接或特殊文件，已拒绝解压。')
        tf.extractall(destination,members=members,filter='data')
    return locate_data(destination)

def compare(actual,expected,prefix=''):
    errors=[]
    if isinstance(expected,dict):
        for key,value in expected.items():
            if key not in actual:errors.append(prefix+key+': missing')
            else:errors+=compare(actual[key],value,prefix+key+'.')
    elif isinstance(expected,list):
        if len(actual)!=len(expected):errors.append(prefix+': length mismatch')
        else:
            for i,value in enumerate(expected):errors+=compare(actual[i],value,prefix+str(i)+'.')
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        if actual is None or not np.isclose(actual,expected,rtol=1e-10,atol=1e-12):errors.append(f'{prefix}: {actual} != {expected}')
    elif actual!=expected:errors.append(f'{prefix}: {actual} != {expected}')
    return errors

def main():
    p=argparse.ArgumentParser(description='固定方法复现。BSDS train/val均为已见开发回归，非独立测试。')
    p.add_argument('--synthetic',action='store_true',help='生成并评估192张锁定合成图+192张空场图')
    p.add_argument('--bsds-root',type=Path,help='BSDS500 data目录或其上级')
    p.add_argument('--split',choices=['train200','val100','both'],default='both')
    p.add_argument('--extract-bsds',action='store_true',help='校验并解压随包BSDS归档，随后评估')
    p.add_argument('--bsds-archive',type=Path,default=ROOT.parent/'数据/BSR_bsds500.tgz')
    p.add_argument('--extract-to',type=Path,default=ROOT/'data/BSR_bsds500')
    p.add_argument('--check',action='store_true',help='与随包期望指标核对；不一致时退出码为1')
    p.add_argument('--workers',type=int,default=1,help='并行线程数；内存有限时用1，通常4较快')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/reproduction')
    args=p.parse_args()
    if args.workers<1:p.error('--workers必须至少为1')
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    cv2.setNumThreads(1)
    if args.extract_bsds:args.bsds_root=extract_bsds(args.bsds_archive,args.extract_to)
    if not args.synthetic and not args.bsds_root:
        hint='；随包数据可用 --extract-bsds 解压并评估' if args.bsds_archive.exists() else ''
        p.error('请选择 --synthetic 或 --bsds-root'+hint)
    start=time.perf_counter();summaries={}
    if args.synthetic:
        for split in ('locked_test','locked_null'):
            cases=generate_cases(split,frozen_protocol_sha256=FREEZE)
            rows=map_rows(synth_worker,[(split,c) for c in cases],args.workers,split)
            rows.sort(key=lambda r:r['case_id'])
            save(args.output/(split+'_per_image.json'),rows)
            summaries['synthetic' if split=='locked_test' else 'null']=synth_aggregate(rows) if split=='locked_test' else null_aggregate(rows)
    if args.bsds_root:
        data=locate_data(args.bsds_root)
        stages=('train200','val100') if args.split=='both' else (args.split,)
        for stage in stages:
            split='train' if stage=='train200' else 'val';paths=sorted((data/'images'/split).glob('*.jpg'))
            required=200 if stage=='train200' else 100
            if len(paths)!=required:raise ValueError(f'{stage}要求{required}张JPG，实际{len(paths)}张')
            rows=map_rows(bsds_worker,[(str(q),str(data/'groundTruth'/split/(q.stem+'.mat'))) for q in paths],args.workers,stage)
            rows.sort(key=lambda r:r['image_id']);save(args.output/(stage+'_per_image.json'),rows)
            summaries['bsds_seen_'+stage]=dict(images=len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in ('precision','recall','f1')})
    errors=[]
    if args.check:
        for key,value in summaries.items():errors+=compare(value,EXPECTED[key],key+'.')
    report=dict(method=METHOD,frozen_protocol_sha256=FREEZE,metrics=summaries,
                runtime=dict(python=platform.python_version(),platform=platform.platform(),packages={k:importlib.metadata.version(k) for k in ('numpy','scipy','Pillow','opencv-python-headless')}),
                evaluation=dict(border=13,tolerance_pixels=2,matching='one-to-one maximum-cardinality then minimum-distance',
                    bsds='mean annotations within each image, then unweighted image mean; custom fixed-threshold metric, not official ODS/OIS/AP',
                    synthetic='pixel-count micro aggregation; short GT=8-connected component <=32 pixels',
                    bsds_data_role='previously seen development/regression; not independent test',
                    input='BSDS: .25*gray**1.5 with deterministic signal-dependent additive noise; synthetic: original v7 floating point arrays'),
                elapsed_seconds=time.perf_counter()-start,checked=args.check,check_passed=not errors if args.check else None,check_errors=errors)
    save(args.output/'summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
