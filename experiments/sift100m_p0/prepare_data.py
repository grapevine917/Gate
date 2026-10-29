"""Freeze query splits and verify the supplied 100M uint8 benchmark."""
from pathlib import Path
import hashlib,json,struct
import numpy as np
R=Path(__file__).resolve().parent; ROOT=R.parents[1]; D=ROOT/'data/sift100m'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if (R/'DATA_COMPLETE.json').exists():raise SystemExit(0)
base=np.memmap(D/'base.100M.u8bin','u1','r',offset=8,shape=(100000000,128))
with (D/'query.public.10K.u8bin').open('rb') as f:
 assert struct.unpack('<II',f.read(8))==(10000,128);q=np.fromfile(f,'u1').reshape(10000,128)
with (D/'groundtruth.public.10K.ibin').open('rb') as f:
 assert struct.unpack('<II',f.read(8))==(10000,100);gt=np.fromfile(f,'<u4',1000000).reshape(10000,100);dist=np.fromfile(f,'<f4').reshape(10000,100)
assert (D/'base.100M.u8bin').stat().st_size==12800000008
with (D/'base.100M.u8bin').open('rb') as f:assert struct.unpack('<II',f.read(8))==(100000000,128)
assert gt.max()<100000000 and np.all(np.diff(dist,axis=1)>=0)
audit=np.linspace(0,9999,16,dtype=int)
for i in audit:
 delta=base[gt[i]].astype('i4')-q[i].astype('i4');np.testing.assert_array_equal((delta*delta).sum(1),dist[i])
perm=np.random.default_rng(42).permutation(10000);pool=perm[:8000]
ids={'train':np.sort(np.random.default_rng(20261030).choice(pool,1200,False)),'validation':np.sort(perm[8000:9000]),'test':np.sort(perm[9000:])}
hold=np.random.default_rng(20261031).choice(ids['train'],200,False)
for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not set(ids[a])&set(ids[b])
np.save(R/'data/internal_holdout_ids.npy',hold)
for split,rows in ids.items():
 np.save(R/'data'/f'{split}_ids.npy',rows)
 for arr,ext in [(q,'u8bin'),(q.astype('<f4'),'fbin'),(gt,'gt')]:
  with (R/'data'/f'{split}.{ext}').open('wb') as f:np.array([len(rows),arr.shape[1]],'<u4').tofile(f);arr[rows].tofile(f)
dump(R/'splits.json',{s:a.tolist() for s,a in ids.items()}|{'internal_holdout':hold.tolist()})
hashes={str(p):sha(p) for p in D.iterdir() if p.is_file()}
dump(R/'DATA_COMPLETE.json',dict(passed=True,base_shape=[100000000,128],dtype='uint8',GT_format='uint32[n,100] then float32 squared-L2[n,100]',GT_distance_audit_query_ids=audit.tolist(),GT_distance_audit_neighbors_each=100,query_disjoint=True,split_seed=42,sha256=hashes,limits='Distance audit verifies supplied neighbor distances, not global exact top100 membership.'))
print('DATA COMPLETE',flush=True)
