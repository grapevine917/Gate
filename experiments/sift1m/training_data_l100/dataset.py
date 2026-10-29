#!/usr/bin/env python3
"""Memory-mapped expansion dataset. Only explicit feature matrices enter batch()."""
import argparse,json
from pathlib import Path
import numpy as np
class ExpansionDataset:
 def __init__(self,root=None,split='train',state_view='pre_io',normalize=True):
  self.root=Path(root) if root else Path(__file__).resolve().parent
  assert json.loads((self.root/'COMPLETE.json').read_text())['complete'],'Dataset is incomplete'
  self.manifest=json.loads((self.root/'manifest.json').read_text())
  if split not in ['train','validation','test']: raise ValueError('Unknown split')
  if state_view not in ['pre_io','post_score']: raise ValueError('Unknown state view')
  folder=self.root/split; self.split=split; self.state_view=state_view; self.normalize=normalize
  self._meta=np.load(folder/'metadata.npy',mmap_mode='r'); self.y=np.load(folder/'y.npy',mmap_mode='r')
  self.state=np.load(folder/(state_view+'_state.npy'),mmap_mode='r')
  self.query_ids=np.load(folder/'query_ids.npy',mmap_mode='r'); self.queries=np.load(folder/'query_vectors.npy',mmap_mode='r')
  self.query_expansion_offsets=np.load(folder/'query_expansion_offsets.npy',mmap_mode='r')
  base=self.manifest['base_vectors']; self.base=np.memmap(base['path'],mode='r',dtype=base['dtype'],offset=base['offset_bytes'],shape=tuple(base['shape']))
  self.feature_names=self.manifest[state_view+'_state_columns']
  stats=np.load(self.root/(state_view+'_state_train_normalization.npz'))
  self.mean=stats['mean']; self.scale=stats['scale']
  self._neighbors=None; self._safe_neighbors=None
 def __len__(self): return len(self.y)
 def batch(self,indices,include_vectors=True):
  ix=np.atleast_1d(np.asarray(indices,dtype=np.int64))
  if np.any(ix<0) or np.any(ix>=len(self)): raise IndexError('Sample index out of range')
  m=self._meta[ix]; s=self.state[ix].copy()
  if self.normalize: s=(s-self.mean)/self.scale
  b={'query_id':m['query_id'],'node_id':m['node_id'],'sample_id':m['sample_id'],'state':s,'y':self.y[ix].copy(),'q':self.queries[m['query_row']]}
  if include_vectors: b['v']=self.base[m['node_id']]
  return b
 def iter_batches(self,batch_size=4096,shuffle=False,seed=42,include_vectors=True):
  order=np.arange(len(self))
  if shuffle: np.random.default_rng(seed).shuffle(order)
  for i in range(0,len(self),batch_size): yield self.batch(order[i:i+batch_size],include_vectors)
 def neighbors(self,query_id,safe_only=False):
  row=int(np.searchsorted(self.query_ids,query_id))
  if row>=len(self.query_ids) or int(self.query_ids[row])!=query_id: raise KeyError('query belongs to another split')
  folder=self.root/self.split
  if safe_only:
   if self._safe_neighbors is None: self._safe_neighbors=np.load(folder/'safe_skip_neighbors.npy',mmap_mode='r')
   data=self._safe_neighbors; offsets=np.load(folder/'query_safe_neighbor_offsets.npy',mmap_mode='r')
  else:
   if self._neighbors is None: self._neighbors=np.load(folder/'neighbors.npy',mmap_mode='r')
   data=self._neighbors; offsets=np.load(folder/'query_neighbor_offsets.npy',mmap_mode='r')
  return data[int(offsets[row]):int(offsets[row+1])]
if __name__=='__main__':
 a=argparse.ArgumentParser(); a.add_argument('--split',default='train'); a.add_argument('--state-view',default='pre_io'); args=a.parse_args()
 ds=ExpansionDataset(split=args.split,state_view=args.state_view); b=ds.batch(np.arange(min(4,len(ds))))
 print(json.dumps({'split':ds.split,'samples':len(ds),'feature_names':ds.feature_names,'batch_shapes':{k:list(v.shape) for k,v in b.items()},'batch_labels':b['y'].tolist()},indent=2))
