from pathlib import Path
import numpy as np,json
ROOT=Path(__file__).resolve().parent;LG=ROOT.parent.parent
manifest={}
def readbin(p,dtype):
 with open(p,'rb') as f:n,d=np.fromfile(f,np.uint32,2);return np.fromfile(f,dtype,count=int(n)*int(d)).reshape(int(n),int(d))
def writebin(p,a):
 with open(p,'wb') as f:np.array(a.shape,np.uint32).tofile(f);np.asarray(a).tofile(f)
for ds,query,gt,idsbase in [('sift','data/sift/sift_query.fbin','data/sift/gt100.bin','experiments/sift1m/training_data_l100'),('deep','data/deep1M_query.fbin','data/deep1M_gt100.bin','experiments/deep1m/queries')]:
 qs=readbin(LG/query,np.float32);truth=readbin(LG/gt,np.uint32)
 splits={s:np.load(LG/idsbase/s/'query_ids.npy') for s in ['train','validation','test']}
 selected=np.sort(np.random.default_rng(20261014).choice(splits['train'],1000,replace=False))
 splits['train']=selected
 for s,ids in splits.items():
  dest=ROOT/ds/s;dest.mkdir(parents=True,exist_ok=True);writebin(dest/'query.fbin',qs[ids]);writebin(dest/'gt.bin',truth[ids]);np.save(dest/'query_ids.npy',ids)
 assert not any(set(splits[a])&set(splits[b]) for a,b in [('train','validation'),('train','test'),('validation','test')])
 manifest[ds]={s:ids.tolist() for s,ids in splits.items()}
(ROOT/'query_splits.json').write_text(json.dumps(manifest,indent=2)+'\n')
