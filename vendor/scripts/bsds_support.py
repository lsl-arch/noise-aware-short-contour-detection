"""Exact cached classroom detector and faster sparse graph construction.

The detector keeps the original kernels, convolutions, NMS, chain rule and
threshold semantics. Only repeated work and Python graph-construction loops
are removed; `verify` compares predictions and matching costs to the package.
"""
from pathlib import Path
import sys, math, time, hashlib
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.io import loadmat
from PIL import Image
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'原版复现'/'lab'))
from edge_lab import core
from edge_lab.experiments import match_boundaries

SCALES=(.7,1.4,2.8)
HIGHS=(3.,5.,7.,10.,14.,20.)
CANNY_HIGHS=(10,20,30,50,80,120)
FAMILIES=('A0_detect','A1_detect','A1_conf','Fixed_s2.8','Canny')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_input(path):
    rgb=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)/255
    gray=rgb@np.array([.299,.587,.114],np.float32)
    clean=(.25*np.power(gray,1.5)).astype(np.float32)
    seed=int.from_bytes(hashlib.sha256(('lesson3-bsds-lowlight-v1:'+Path(path).stem).encode()).digest()[:8],'little')
    sigma=np.sqrt(.004**2+.0006*clean).astype(np.float32)
    image=(clean+np.random.default_rng(seed).standard_normal(clean.shape)*sigma).astype(np.float32)
    return gray,clean,image,seed

def truths(path):
    a=loadmat(path)['groundTruth']
    return [np.asarray(x['Boundaries'][0,0],bool) for x in a.ravel()]

def fast_mutual(a,b,radius,angle_deg):
    if not len(a['yx']) or not len(b['yx']): return []
    pairs=cKDTree(a['yx']).sparse_distance_matrix(cKDTree(b['yx']),radius,output_type='coo_matrix')
    ia,ib=pairs.row,pairs.col
    ok=core.angular_difference(a['theta'][ia],b['theta'][ib])<=np.radians(angle_deg)
    ia,ib=ia[ok],ib[ok]
    if not len(ia): return []
    dist=((a['yx'][ia]-b['yx'][ib])**2).sum(axis=1)
    order=np.lexsort((ib,dist,ia)); ai,bi=ia[order],ib[order]
    first=np.r_[True,ai[1:]!=ai[:-1]]
    best_a=np.full(len(a['yx']),-1,int);best_a[ai[first]]=bi[first]
    order=np.lexsort((ia,dist,ib));ai,bi=ia[order],ib[order]
    first=np.r_[True,bi[1:]!=bi[:-1]]
    best_b=np.full(len(b['yx']),-1,int);best_b[bi[first]]=ai[first]
    good=np.flatnonzero(best_a>=0)
    good=good[best_b[best_a[good]]==good]
    return [(int(i),int(best_a[i])) for i in good]

def fast_chains(candidates):
    successor,predecessor={},set()
    for s in range(len(SCALES)-1):
        for i,j in fast_mutual(candidates[s],candidates[s+1],math.ceil(SCALES[s+1]),20):
            successor[(s,i)]=(s+1,j);predecessor.add((s+1,j))
    chains=[]
    for s,c in enumerate(candidates):
        for j in range(len(c['yx'])):
            node=(s,j)
            if node in predecessor: continue
            chain=[node]
            while chain[-1] in successor: chain.append(successor[chain[-1]])
            chains.append(chain)
    return chains

def global_match(pred,truth,tolerance=2.):
    a,b=np.column_stack(np.nonzero(pred)),np.column_stack(np.nonzero(truth))
    if not len(a) or not len(b): return np.empty(0),len(a),len(b)
    pairs=cKDTree(a).sparse_distance_matrix(cKDTree(b),tolerance,output_type='coo_matrix')
    penalty=(min(len(a),len(b))+1)*(tolerance+1)
    rows=np.r_[pairs.row,np.arange(len(a))]
    cols=np.r_[pairs.col,len(b)+np.arange(len(a))]
    costs=np.r_[pairs.data+1e-8,np.full(len(a),penalty)]
    graph=coo_matrix((costs,(rows,cols)),shape=(len(a),len(b)+len(a))).tocsr()
    ia,ib=min_weight_full_bipartite_matching(graph)
    real=ib<len(b)
    return np.linalg.norm(a[ia[real]]-b[ib[real]],axis=1),len(a),len(b)

def fast_match(pred,truth,tolerance=2.):
    # The real radius graph is a disjoint union of components; cardinality
    # and total distance are additive. Retain the original global dummy cost.
    from component_match_probe import component_match
    return component_match(pred,truth,tolerance)

def evaluate(pred,gt,border=9):
    valid=np.zeros_like(pred);valid[border:-border,border:-border]=True
    ds,npred,ngt=fast_match(pred&valid,gt&valid)
    tp=len(ds);p=tp/npred if npred else 0.;r=tp/ngt if ngt else 0.
    return dict(precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0.,
                matched_mean_px=float(ds.mean()) if tp else None,
                matched_p95_px=float(np.percentile(ds,95)) if tp else None,
                matched=tp,predicted=npred,truth=ngt)

class Detector:
    def __init__(self,image):
        self.image=image;self.shape=image.shape
        self.V,self.D,_=core.estimate_noise(image,32)
        self.magnitude=[];theta=[];self.kernels=[]
        for scale in SCALES:
            hx,hy=core.derivative_kernels(scale)
            gx=ndi.convolve(image.astype(np.float64),hx,mode='reflect')
            gy=ndi.convolve(image.astype(np.float64),hy,mode='reflect')
            self.magnitude.append(np.hypot(gx,gy));theta.append(np.arctan2(gy,gx))
            self.kernels.append(hx*hx+hy*hy)
        self.M=np.asarray(self.magnitude,np.float32);self.theta=np.asarray(theta,np.float32)
        self.keep=np.asarray([core.nms(m,t) for m,t in zip(self.M,self.theta)])
        self.banks={};self.chains={}

    def bank(self,tag):
        if tag not in self.banks:
            v={'A0':self.V,'A1':self.V+self.D,'A2':1.5*self.V}[tag]
            v=np.maximum(v,(1/255)**2)
            n2=np.asarray([ndi.convolve(v.astype(np.float64),k,mode='reflect') for k in self.kernels])
            c=(np.asarray(self.magnitude)/np.sqrt(np.maximum(n2,0)+1e-12)).astype(np.float32)
            self.banks[tag]=dict(M=self.M,theta=self.theta,C=c,N2=n2.astype(np.float32))
        return self.banks[tag]

    def candidates(self,tag,high):
        key=(tag,float(high))
        if key not in self.chains:
            c=self.bank(tag)['C'];cand=[]
            for s in range(3):
                ok=self.keep[s]&(c[s]>=.6*high)
                cand.append(dict(yx=np.column_stack(np.nonzero(ok)),score=c[s][ok],theta=self.theta[s][ok]))
            self.chains[key]=(cand,fast_chains(cand))
        return self.chains[key]

    def edge(self,family,high):
        if family=='Canny':
            u8=np.round(np.clip(self.image,0,1)*255).astype(np.uint8)
            smooth=cv2.GaussianBlur(u8,(11,11),1.4,borderType=cv2.BORDER_REFLECT)
            return cv2.Canny(smooth,int(high*.5),int(high),apertureSize=3,L2gradient=True)>0
        if family=='Fixed_s2.8': return self.keep[2]&(self.bank('A0')['C'][2]>=high)
        tag=family[:2]
        if family=='A3_detect': tag='A0'
        cand,chains=self.candidates(tag,high)
        out=np.zeros(self.shape,bool)
        for chain in chains:
            strong=[(s,j) for s,j in chain if cand[s]['score'][j]>=high]
            if not strong: continue
            ds,dj=min(strong)
            ls,lj=ds,dj
            if family=='A1_conf':
                choices=[(s,j) for s,j in chain if s<=ds and cand[s]['score'][j]>=max(.6*high,.8*high)]
                ls,lj=min(choices) if choices else (ds,dj)
            y,x=cand[ls]['yx'][lj];out[y,x]=True
        return out

def verify(image,gt):
    start=time.perf_counter();det=Detector(image);checks=[]
    for tag in ['A0','A1','A2']:
        var=np.maximum({'A0':det.V,'A1':det.V+det.D,'A2':1.5*det.V}[tag],(1/255)**2)
        bank=core.build_scale_bank(image,var,SCALES,1e-12)
        for k in ['M','theta','C','N2']:
            if not np.array_equal(bank[k],det.bank(tag)[k]):
                raise AssertionError((tag,k,float(np.max(abs(bank[k]-det.bank(tag)[k])))))
        for high in [3.,5.,10.]:
            cand=core.make_candidates(bank,.6*high)
            chains=core.build_chains(cand,SCALES,20)
            c2,ch2=det.candidates(tag,high)
            if chains!=ch2: raise AssertionError(('chains',tag,high))
            old=core.select_chains(cand,chains,SCALES,high,image.shape,False)['edge']
            if not np.array_equal(old,det.edge(tag+'_detect',high)): raise AssertionError(('edge',tag,high))
            checks.append(tag+':'+str(high))
    old=match_boundaries(det.edge('A1_detect',5),gt)
    new=fast_match(det.edge('A1_detect',5),gt)
    if old[1:]!=new[1:] or len(old[0])!=len(new[0]) or not np.isclose(old[0].sum(),new[0].sum(),atol=1e-8):
        raise AssertionError(('matching',len(old[0]),len(new[0]),old[0].sum(),new[0].sum()))
    return dict(passed=True,array_equality=True,checks=checks,matching_count_cost_equal=True,elapsed_s=time.perf_counter()-start)
