"""Convert native DECTR001 chunks to analysis NPZ; reject incomplete/corrupt traces."""
from pathlib import Path
import argparse,json,struct
import numpy as np
DT=np.dtype([('query_id','<u4'),('node_id','<u4'),('decision_step','<u4'),('search_step','<u4'),('state_features','<f4',(17,)),('model_score','<f4'),('cache_hit','u1'),('model_invoked','u1'),('model_decision','u1'),('reserved','u1')])
def convert(manifest,output):
 m=json.loads(Path(manifest).read_text());assert m['complete'] and m['schema']=='DECTR001' and DT.itemsize==92
 ids=np.fromfile(m['query_ids'],dtype='<u4');v=np.fromfile(m['query_vectors'],dtype='<f4',offset=8).reshape(m['query_count'],m['dimension'])
 assert len(ids)==m['query_count'] and len(np.unique(ids))==len(ids)
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(output/'queries.npz',query_ids=ids,query_vectors=v)
 for item in m['records']:
  chunks=[]
  with open(item['path'],'rb') as f:
   assert f.read(8)==b'DECTR001';n,size=struct.unpack('<II',f.read(8));assert n==len(ids) and size==92
   for row in range(n):
    qid,qrow,count=struct.unpack('<III',f.read(12));assert qid==ids[row] and qrow==row
    a=np.frombuffer(f.read(count*size),dtype=DT).copy();assert len(a)==count
    assert np.all(a['query_id']==qid) and np.array_equal(a['decision_step'],np.arange(count))
    assert np.all(a['cache_hit']==(a['state_features'][:,10]!=0))
    assert np.all(a['search_step']==a['state_features'][:,8]) and np.all(a['model_invoked']==1)
    chunks.append(a)
   assert not f.read(1)
  a=np.concatenate(chunks);np.savez_compressed(output/f"L{item['L']}.npz",**{k:a[k] for k in DT.names if k!='reserved'},all_query_ids=ids)
 return m
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('output');a=p.parse_args();convert(a.manifest,a.output)
