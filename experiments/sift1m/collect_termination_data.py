#!/usr/bin/env python3
"""Capture real beam-boundary state and final-baseline-top10 prefix labels."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from run_online_m3 import E,ROOT,BIN,execute,mat,save_mat
from train_state_baselines import dump
OUT=E/'learned_termination';DATA=E/'training_data_l100'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-trace',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True);(OUT/'DATA_COMPLETE.json').unlink(missing_ok=True)
    trace=OUT/'baseline_boundaries.f32'
    print('Collecting beam-boundary states for all 10,000 baseline queries',flush=True)
    if not args.reuse_trace:
        execute('collect_baseline',ROOT/'data/sift/sift_query.fbin',ROOT/'data/sift/gt100.bin',
                {'DISKANN_ET_TRACE':str(trace)},output_dir=OUT)
    for suffix,dtype in (('idx_uint32','<u4'),('dists_float','<f4')):
        np.testing.assert_array_equal(mat(OUT/f'collect_baseline_100_{suffix}.bin',dtype),
                                      mat(E/f'baseline_100_{suffix}.bin',dtype))
    a=np.fromfile(trace,dtype='<f4').reshape(-1,21)
    assert np.isfinite(a).all() and len(np.unique(a[:,0]))==10000
    a=a[np.lexsort((a[:,10],a[:,0]))] # query, then state[beam_step=8]
    assert np.all(a[:,9]>=10) and np.all(a[:,11]==0) # state completed=7, beam_position=9
    # Independently rebuild safe-stop labels from prior expansion metadata and baseline IDs.
    # Oracle membership uses only final baseline top10, never GT as model input.
    baseline=mat(E/'baseline_100_idx_uint32.bin','<u4')
    query_data={}
    for split in ('train','validation','test'):
        m=np.load(DATA/split/'metadata.npy',mmap_mode='r')
        x=np.load(DATA/split/'pre_io_state.npy',mmap_mode='r')
        order=np.lexsort((x[:,9],x[:,8],m['query_id']))
        qids=m['query_id'][order];starts=np.flatnonzero(np.r_[True,qids[1:]!=qids[:-1]])
        ends=np.r_[starts[1:],len(order)]
        for start,end in zip(starts,ends):
            ix=order[start:end];query=int(qids[start])
            # Prefix counts at beam boundaries are independent of within-beam exact-score order.
            last_final_beam=int(x[ix[np.isin(m['node_id'][ix],baseline[query])],8].max())
            query_data[query]=last_final_beam
    expected=np.array([int(row[10])>query_data[int(row[0])] for row in a],dtype='uint8')
    np.testing.assert_array_equal(a[:,19],expected)
    counts={}
    for split in ('train','validation','test'):
        ids=np.load(DATA/split/'query_ids.npy');folder=OUT/'data'/split;folder.mkdir(parents=True,exist_ok=True)
        subset=a[np.isin(a[:,0],ids)]
        assert len(np.unique(subset[:,0]))==len(ids)
        original_meta=np.load(DATA/split/'metadata.npy',mmap_mode='r')
        original_state=np.load(DATA/split/'pre_io_state.npy',mmap_mode='r')
        keys=original_meta['query_id'].astype('uint64')*1000000+original_meta['node_id']
        order=np.argsort(keys);wanted=subset[:,0].astype('uint64')*1000000+subset[:,1].astype('uint64')
        positions=order[np.searchsorted(keys[order],wanted)]
        np.testing.assert_array_equal(keys[positions],wanted)
        np.testing.assert_array_equal(subset[:,2:19],original_state[positions])
        np.save(folder/'state.npy',subset[:,2:19]);np.save(folder/'y_safe_stop.npy',subset[:,19].astype('uint8'))
        np.save(folder/'query_ids.npy',ids);np.save(folder/'sample_query_ids.npy',subset[:,0].astype('uint32'))
        np.save(folder/'candidate_node_ids.npy',subset[:,1].astype('uint32'))
        counts[split]=dict(queries=len(ids),boundaries=len(subset),safe=int(subset[:,19].sum()),
                           unsafe=int((subset[:,19]==0).sum()))
    # Reuse held-out test files and prepare validation files in original validation ID order.
    ids=np.load(DATA/'validation/query_ids.npy')
    save_mat(OUT/'validation_query.fbin',mat(ROOT/'data/sift/sift_query.fbin','<f4')[ids])
    gt=ROOT/'data/sift/gt100.bin'
    with gt.open('rb') as f:
        n,d=np.frombuffer(f.read(8),dtype='<u4');g=np.fromfile(f,dtype='<u4',count=int(n)*int(d)).reshape(int(n),int(d))
        raw_distances=np.fromfile(f,dtype='<f4')
        distances=raw_distances.reshape(int(n),int(d)) if len(raw_distances) else None
    with (OUT/'validation_gt.bin').open('wb') as f:
        np.array([len(ids),d],dtype='<u4').tofile(f);g[ids].tofile(f)
        if distances is not None:distances[ids].tofile(f)
    names=json.loads((DATA/'manifest.json').read_text())['pre_io_state_columns']
    dump(OUT/'data_manifest.json',dict(L=100,K=10,beam=4,state_columns=names,splits=counts,
        sample_unit='before next beam; only when >=K exact-scored candidates and queue nonempty',
        safe_to_stop_label='1 iff all full L100 baseline final top10 IDs have already been exact-scored',
        independent_label_check='last beam containing baseline final top10 from existing expansion metadata',
        original_query_split=True,labels_monotonic_along_each_query=True,
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(BIN.read_bytes()).hexdigest()))
    dump(OUT/'DATA_COMPLETE.json',dict(complete=True,baseline_parity=True,independent_labels_verified=True))
    print('ET DATA COMPLETE',counts,flush=True)

if __name__=='__main__':main()
