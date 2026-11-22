"""Image-only local and cross-scale component support for Lesson 5."""
from pathlib import Path
import sys
import numpy as np
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'));sys.path.insert(0,str(ROOT/'vendor/scripts'))
from refined_bank import RefinedBank
from reproduce import synthetic_metrics,synth_aggregate,null_aggregate
from bsds_support import read_input,truths,evaluate
from data_v7 import generate_cases

def features(image,alpha=.5,scale=2.,coarse_high=3.5):
    obj=RefinedBank(image);b=obj.get_bank(scale,strategy='mad32',alpha=alpha)
    fine=obj.get_bank(2.8,strategy='mad32',alpha=alpha)
    mask=b['keep']&(b['C']>=2.5)
    labels,n=ndi.label(mask,np.ones((3,3),bool));sizes=np.bincount(labels.ravel(),minlength=n+1)
    # Local axial coherence: aggregate only neighbouring candidate orientations.
    z=np.exp(2j*b['theta'])*mask
    den=ndi.convolve(mask.astype(float),np.ones((3,3)),mode='constant')
    loc=np.abs(ndi.convolve(z.real,np.ones((3,3)),mode='constant')+1j*ndi.convolve(z.imag,np.ones((3,3)),mode='constant'))/np.maximum(den,1)
    # A nearby fine-scale edge has to agree in axial orientation.
    support=np.zeros(mask.shape,bool)
    for dy in [-1,0,1]:
      for dx in [-1,0,1]:
        fk=ndi.shift((fine['keep']&(fine['C']>=coarse_high)).astype('uint8'),(dy,dx),order=0,mode='constant')>0
        ft=ndi.shift(fine['theta'],(dy,dx),order=0,mode='nearest')
        support |= fk&(np.cos(2*(b['theta']-ft))>=.5)
    score=np.zeros(n+1);local=np.zeros(n+1);global_dir=np.zeros(n+1);cross=np.zeros(n+1)
    for i,sl in enumerate(ndi.find_objects(labels),1):
      if sl is None:continue
      q=labels[sl]==i
      score[i]=np.median(b['C'][sl][q]);local[i]=np.mean(loc[sl][q]);global_dir[i]=abs(np.mean(np.exp(2j*b['theta'][sl][q])));cross[i]=np.mean(support[sl][q])
    return dict(labels=labels,sizes=sizes,score=score,local=local,global_dir=global_dir,cross=cross,mask=mask)

def predict_features(f,config):
    a=f['sizes']>=32
    if config.get('gate_long',False):a &= f['cross']>=config.get('cross',0)
    if config.get('mode')=='adaptive':
        a=(f['sizes']>=config.get('min_size',4)) & ((f['score']-2.5)*np.sqrt(f['sizes'])>=config['k']) & (f['local']>=config.get('local',0))
        a[0]=False
        return a[f['labels']]
    q=(f['sizes']>=4)&(f['sizes']<32)&(f['score']>=config['score'])
    if config.get('local',0):q &= f['local']>=config['local']
    if config.get('cross',0):q &= f['cross']>=config['cross']
    if config.get('global',0):q &= f['global_dir']>=config['global']
    if config.get('short',True):a |= q
    a[0]=False
    return a[f['labels']]

def detect(image,config):return predict_features(features(image,alpha=config.get('alpha',.5),coarse_high=config.get('coarse_high',2.5)),config)
