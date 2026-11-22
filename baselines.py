from pathlib import Path
import os,sys,time
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parent
class Baselines:
 def __init__(self,deep=True):
  cv2.setNumThreads(1)
  # OpenCV's FileStorage on Windows cannot open a non-ASCII absolute path.
  # The ASCII sidecar is a byte-for-byte copy of the frozen model.
  sf_path=ROOT/'models/structured_forest.yml.gz'
  sidecar=ROOT/'models_ascii/sf.yml.gz'
  # A relative ASCII path avoids OpenCV's Unicode conversion bug on Windows.
  if sidecar.exists(): sf_path=Path('models_ascii/sf.yml.gz')
  self.sf=cv2.ximgproc.createStructuredEdgeDetection(str(sf_path))
  self.deep=deep and os.environ.get('LESSON5_SKIP_DEXINED','0')!='1'
  self.device='unavailable'
  if self.deep:
   try:
    import torch
    sys.path.insert(0,str(ROOT/'models'));from dexined_model import DexiNed
    torch.set_num_threads(4);self.torch=torch;self.device='mps' if torch.backends.mps.is_available() else 'cpu'
    self.net=DexiNed();state=torch.load(ROOT/'models/dexined.pth',map_location='cpu',weights_only=True);self.net.load_state_dict(state);self.net.to(self.device).eval()
   except ImportError:
    self.deep=False
 def nms(self,e):
  e=np.ascontiguousarray(e,dtype=np.float32);o=self.sf.computeOrientation(e);return self.sf.edgesNms(e,o,r=2,s=0,m=1,isParallel=False)
 def responses(self,image):
  im=np.clip(image,0,1);rgb=np.repeat(im[:,:,None],3,axis=2).astype(np.float32)
  start=time.perf_counter();sf=self.nms(self.sf.detectEdges(rgb));times={'StructuredForests':time.perf_counter()-start};out={'StructuredForests':sf}
  if self.deep:
   torch=self.torch;start=time.perf_counter();x=cv2.resize(rgb*255,(512,512))-np.array([103.939,116.779,123.68],np.float32)
   x=torch.from_numpy(x.transpose(2,0,1).copy()).unsqueeze(0).to(self.device)
   with torch.inference_mode():e=torch.sigmoid(self.net(x)[-1])[0,0].cpu().numpy()
   # Match author display scaling but retain floating point, white=edge; add declared common NMS.
   e=(e-e.min())/(e.max()-e.min()+1e-12);e=cv2.resize(e,(image.shape[1],image.shape[0]));out['DexiNed']=self.nms(e);times['DexiNed']=time.perf_counter()-start
  return out,times

def canny(image,high,clahe=False):
 u=np.uint8(np.clip(image,0,1)*255)
 if clahe:u=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(u)
 u=cv2.GaussianBlur(u,(0,0),1.)
 return cv2.Canny(u,float(high)*.4,float(high),L2gradient=True)>0
