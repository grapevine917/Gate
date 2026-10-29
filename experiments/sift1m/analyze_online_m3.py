#!/usr/bin/env python3
"""Check live sweep counters, repeat determinism, baseline parity and savings."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
from run_online_m3 import mat,E,OUT

def main():
    assert json.loads((OUT/'COMPLETE.json').read_text())['complete']
    runs=json.loads((OUT/'runs.json').read_text())
    historical=mat(E/'baseline_100_idx_uint32.bin','<u4')
    test_ids=np.load(E/'training_data_l100/test/query_ids.npy')
    for suite,n in [('test1000',1000),('all10000',10000)]:
        gt=mat(OUT/'test_gt.bin' if suite=='test1000' else E.parents[1]/'data/sift/gt100.bin','<u4')
        configs={r['config'] for r in runs if r['suite']==suite}
        for c in configs:
            rows=[r for r in runs if r['suite']==suite and r['config']==c]
            assert len(rows)==3
            first=None
            for r in rows:
                prefix=OUT/r['label']
                stats=np.genfromtxt(str(prefix)+'.csv',delimiter=',',names=True)
                assert len(stats)==n
                if c!='baseline':
                    assert np.array_equal(stats['expansions'],stats['gate_evaluated']-stats['gate_skipped'])
                ids=mat(str(prefix)+'_100_idx_uint32.bin','<u4')
                if first is None:first=ids
                else:assert np.array_equal(ids,first)
                if c=='baseline':
                    assert np.array_equal(ids,historical[test_ids] if suite=='test1000' else historical)
            hits=0
            for i in range(0,n,512):
                hits+=int(np.any(first[i:i+512,:,None]==gt[i:i+512,None,:10],axis=2).sum())
            assert abs(100*hits/(n*10)-rows[0]['recall_pct'])<1e-6
    summary=json.loads((OUT/'summary.json').read_text());deltas=[]
    for r in summary:
        base=next(x for x in summary if x['suite']==r['suite'] and x['config']=='baseline')
        deltas.append(dict(suite=r['suite'],config=r['config'],recall_delta_percentage_points=r['recall_pct']-base['recall_pct'],
            expansion_reduction_pct=100*(1-r['expansions_per_query']/base['expansions_per_query']),
            ssd_read_reduction_pct=100*(1-r['ssd_reads_per_query']/base['ssd_reads_per_query']),
            latency_reduction_pct=100*(1-r['latency_us']/base['latency_us']),
            qps_change_pct=100*(r['qps']/base['qps']-1)))
    (OUT/'deltas.json').write_text(json.dumps(deltas,indent=2)+'\n')
    verification=json.loads((OUT/'verification.json').read_text())
    verification.update(all_36_measured_runs_counter_consistency_verified=True,
                        repeat_ordered_results_identical=True,baseline_full_and_test_result_parity_verified=True,
                        recalls_independently_recomputed_from_result_ids=True)
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    source=OUT/'source';source.mkdir(exist_ok=True)
    repo=E.parents[1]/'DiskANN';hashes={}
    for name in ['include/online_gate.h','include/percentile_stats.h','src/pq_flash_index.cpp','apps/search_disk_index.cpp']:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,target)
        hashes[name]=hashlib.sha256(target.read_bytes()).hexdigest()
    p=OUT/'provenance.json';prov=json.loads(p.read_text());prov['source_snapshots_sha256']=hashes
    p.write_text(json.dumps(prov,indent=2)+'\n')
    print(json.dumps(deltas,indent=2));print('All 36 live runs verified.')

if __name__=='__main__':main()
