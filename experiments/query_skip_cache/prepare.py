from pathlib import Path
import os,time,json,hashlib
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
ROOT=Path(__file__).resolve().parent;LG=ROOT.parents[1];SOURCE=ROOT.parent/'io_predictability'
def matrix(path,dt):
 with open(path,'rb') as f:n,d=np.fromfile(f,'<u4',2);return np.fromfile(f,dt,count=int(n)*int(d)).reshape(int(n),int(d))
def write(path,a):
 with open(path,'wb') as f:np.array(a.shape,'<u4').tofile(f);np.asarray(a).tofile(f)
def groundtruth(base,query):
 with open(base,'rb') as f:n,d=np.fromfile(f,'<u4',2)
 x=np.memmap(base,mode='r',offset=8,dtype='<f4',shape=(int(n),int(d)));q=query.astype(np.float64);qn=(q*q).sum(1)
 best=np.full((len(q),10),np.inf);ids=np.zeros((len(q),10),np.uint32)
 for start in range(0,len(x),8192):
  b=x[start:start+8192].astype(np.float64);dist=qn[:,None]+(b*b).sum(1)[None,:]-2*q@b.T
  ix=np.argpartition(dist,9,axis=1)[:,:10];dv=np.take_along_axis(dist,ix,axis=1);di=(ix+start).astype(np.uint32)
  dv=np.concatenate([best,dv],axis=1);di=np.concatenate([ids,di],axis=1);keep=np.argsort(dv,axis=1,kind='stable')[:,:10];best=np.take_along_axis(dv,keep,axis=1);ids=np.take_along_axis(di,keep,axis=1)
 return ids
manifest={}
for ds in ['sift','deep']:
 dst=ROOT/ds;dst.mkdir(exist_ok=True);queries=matrix(SOURCE/ds/'train/query.fbin','<f4');ids=np.load(SOURCE/ds/'train/query_ids.npy');rng=np.random.default_rng(20261017);order=rng.permutation(len(queries));base=LG/'data'/('sift/sift_base.fbin' if ds=='sift' else 'deep1M_base.fbin');manifest[ds]={}
 for split,parent in [('validation',order[:100]),('test',order[100:200])]:
  folder=dst/('near_'+split);folder.mkdir(exist_ok=True);query=[];parents=[];levels=[]
  for level in [.01,.05]:
   noise=rng.normal(size=queries[parent].shape);noise*=np.linalg.norm(queries[parent].astype(float),axis=1)[:,None]*level/np.linalg.norm(noise,axis=1)[:,None]
   query.append((queries[parent]+noise).astype('<f4'));parents.extend(ids[parent].tolist());levels.extend([level]*len(parent))
  q=np.concatenate(query);write(folder/'query.fbin',q)
  meta={'parent_query_ids':parents,'relative_l2_levels':levels,'queries':len(q),'exact_gt':'float64 blocked squared L2 over all 1M base vectors','query_sha256':hashlib.sha256(q.tobytes()).hexdigest()}
  if not(folder/'COMPLETE.json').exists():
   print(ds,split,'exact GT start',flush=True);start=time.time();gt=groundtruth(base,q);write(folder/'gt.bin',gt);meta['gt_seconds']=time.time()-start;(folder/'COMPLETE.json').write_text(json.dumps(meta,indent=2));print(ds,split,'exact GT complete',meta['gt_seconds'],flush=True)
  manifest[ds][split]=meta
 repeat=dst/'repeat';repeat.mkdir(exist_ok=True);subset=order[200:250];write(repeat/'query.fbin',queries[subset]);write(repeat/'gt.bin',matrix(SOURCE/ds/'train/gt.bin','<u4')[subset]);np.save(repeat/'donor_rows.npy',subset)
 sets=[set(np.load(SOURCE/ds/s/'query_ids.npy').tolist()) for s in ['train','validation','test']];assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]);assert not set(manifest[ds]['validation']['parent_query_ids'])&set(manifest[ds]['test']['parent_query_ids'])
(ROOT/'workloads.json').write_text(json.dumps(manifest,indent=2))
