#!/usr/bin/env python3
"""Validate DEEP1M binary inputs and independently check sampled exact L2 GT."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
from pathlib import Path
import hashlib,json,time
import numpy as np
from threadpoolctl import threadpool_limits
E=Path(__file__).resolve().parent;ROOT=E.parents[1];D=ROOT/'data'
def vectors(path):
    with path.open('rb') as f:n,d=map(int,np.fromfile(f,'<u4',2))
    assert path.stat().st_size==8+n*d*4
    return np.memmap(path,dtype='<f4',mode='r',offset=8,shape=(n,d))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    base=vectors(D/'deep1M_base.fbin');query=vectors(D/'deep1M_query.fbin')
    assert base.shape==(1000000,96) and query.shape==(10000,96)
    for a in (base,query):
        for i in range(0,len(a),65536):assert np.isfinite(a[i:i+65536]).all()
    path=D/'deep1M_groundtruth.ivecs'
    assert path.stat().st_size==10000*101*4
    records=np.fromfile(path,'<i4').reshape(10000,101);assert np.all(records[:,0]==100)
    gt=records[:,1:];assert np.all((gt>=0)&(gt<len(base)))
    assert np.all(np.diff(np.sort(gt,axis=1),axis=1)>0)
    chosen=np.linspace(0,9999,32,dtype=int);q=query[chosen].astype('float64')
    best_d=np.full((len(q),100),np.inf);best_id=np.zeros((len(q),100),dtype='int64');start=time.perf_counter()
    with threadpool_limits(limits=4):
        for first in range(0,len(base),65536):
            b=base[first:first+65536].astype('float64')
            d=np.einsum('ij,ij->i',q,q)[:,None]+np.einsum('ij,ij->i',b,b)[None,:]-2*q@b.T
            ix=np.argpartition(d,99,axis=1)[:,:100];local=np.take_along_axis(d,ix,axis=1)
            cand_d=np.concatenate((best_d,local),axis=1);cand_id=np.concatenate((best_id,ix+first),axis=1)
            ix=np.argpartition(cand_d,99,axis=1)[:,:100]
            best_d=np.take_along_axis(cand_d,ix,axis=1);best_id=np.take_along_axis(cand_id,ix,axis=1)
    order=np.argsort(best_d,axis=1);best_d=np.take_along_axis(best_d,order,axis=1);best_id=np.take_along_axis(best_id,order,axis=1)
    supplied_d=np.sum((base[gt[chosen]].astype('float64')-q[:,None,:])**2,axis=2)
    assert np.all(np.diff(supplied_d,axis=1)>=-1e-6)
    assert np.all(supplied_d[:,:10]<=best_d[:,9:10]+1e-6)
    assert np.all(supplied_d<=best_d[:,-1:]+1e-6)
    equal=np.array([set(a)==set(b) for a,b in zip(gt[chosen,:10],best_id[:,:10])]);assert equal.all()
    with (D/'deep1M_gt100.bin').open('wb') as f:
        np.array(gt.shape,dtype='<u4').tofile(f);gt.astype('<u4').tofile(f)
    info=dict(passed=True,base_shape=list(base.shape),query_shape=list(query.shape),groundtruth_shape=list(gt.shape),metric='squared Euclidean L2; original vectors unchanged',finite_vectors=True,valid_unique_gt_ids=True,checked_queries=chosen.tolist(),sample_exact_top10_sets_identical=True,sample_top100_membership_valid=True,sample_gt_sorted_by_distance=True,exact_check='float64 exhaustive blocked distances against all 1M base vectors for 32 predetermined queries',seconds=time.perf_counter()-start,sha256={name:sha(D/name) for name in ('deep1M_base.fbin','deep1M_query.fbin','deep1M_groundtruth.ivecs')})
    (E/'data_validation.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
if __name__=='__main__':main()
