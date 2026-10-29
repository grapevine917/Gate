#!/usr/bin/env python3
"""Live M3 pre-I/O gate sweep. Thresholds from validation, no baseline mask replay."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import csv,json,hashlib,subprocess,time
from pathlib import Path
import joblib,numpy as np
from train_state_baselines import threshold_from_validation
from predict_model3 import predict_joint
E=Path(__file__).resolve().parent; ROOT=E.parents[1]; OUT=E/'online_m3'; DATA=E/'training_data_l100'
BIN=ROOT/'DiskANN/build/apps/search_disk_index'
TARGETS=[0.001,0.0025,0.005,0.01,0.02]

def mat(path,dtype):
    with open(path,'rb') as f:
        n,d=np.frombuffer(f.read(8),dtype='<u4')
        return np.fromfile(f,dtype=dtype,count=int(n)*int(d)).reshape(int(n),int(d))
def save_mat(path,a):
    with open(path,'wb') as f:
        np.array(a.shape,dtype='<u4').tofile(f); a.tofile(f)
def execute(label,query,gt,env_extra,search_l=100,output_dir=None):
    OUT=Path(output_dir) if output_dir is not None else E/'online_m3'
    prefix=OUT/label
    args=[str(BIN),'--data_type','float','--dist_fn','l2','--index_path_prefix',str(E/'index'),
          '--query_file',str(query),'--gt_file',str(gt),'-K','10','-L',str(search_l),'--result_path',str(prefix),
          '-W','4','-T','4','--num_nodes_to_cache','10000']
    env=os.environ.copy()
    for k in list(env):
        if k.startswith('DISKANN_'): env.pop(k)
    env.update(env_extra); env['DISKANN_QUERY_STATS']=str(prefix)+'.csv'
    (OUT/(label+'.command.json')).write_text(json.dumps(dict(args=args,env={k:v for k,v in env.items() if k.startswith('DISKANN_') or k in ('OMP_NUM_THREADS','MKL_NUM_THREADS')}),indent=2))
    with (OUT/(label+'.log')).open('w') as log:
        subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    summary=next(x for x in (OUT/(label+'.log')).read_text().splitlines() if x.startswith('ONLINE_SUMMARY ')).split()
    stats=np.genfromtxt(str(prefix)+'.csv',delimiter=',',names=True)
    return dict(label=label,recall_pct=float(summary[5]),qps=float(summary[2]),latency_us=float(summary[3]),
                latency_p50_us=float(np.median(stats['latency_us'])),latency_p95_us=float(np.percentile(stats['latency_us'],95)),
                ssd_reads_per_query=float(stats['ssd_reads'].mean()),expansions_per_query=float(stats['expansions'].mean()),
                gate_evaluations_per_query=float(stats['gate_evaluated'].mean()),skips_per_query=float(stats['gate_skipped'].mean()))

def main():
    OUT.mkdir(exist_ok=True); (OUT/'COMPLETE.json').unlink(missing_ok=True)
    bundle=joblib.load(E/'model3/model.joblib')
    model_file=OUT/'model3.bin'
    with model_file.open('wb') as f:
        f.write(b'M3GATE01')
        bundle['mean'].astype('<f4').tofile(f); bundle['scale'].astype('<f4').tofile(f)
        for w,b in zip(bundle['model'].coefs_,bundle['model'].intercepts_):
            np.ascontiguousarray(w,dtype='<f4').tofile(f); b.astype('<f4').tofile(f)
    y=np.load(DATA/'validation/y.npy'); vp=np.load(E/'model3/validation_p_retain.npy')
    chosen={f'fnr_{risk*100:g}pct':threshold_from_validation(y,vp,risk) for risk in TARGETS}
    for c in chosen.values():
        c['decision']='skip iff p(retain|q,v,interactions,live_pre_io_state) < threshold'
    (OUT/'thresholds.json').write_text(json.dumps(chosen,indent=2))
    base=ROOT/'data/sift/sift_base.fbin'; allq=ROOT/'data/sift/sift_query.fbin'; allgt=ROOT/'data/sift/gt100.bin'
    test_ids=np.load(DATA/'test/query_ids.npy'); queries=mat(allq,'<f4')
    save_mat(OUT/'test_query.fbin',queries[test_ids])
    with allgt.open('rb') as f:
        n,d=np.frombuffer(f.read(8),dtype='<u4'); ids=np.fromfile(f,dtype='<u4',count=int(n)*int(d)).reshape(int(n),int(d))
        raw_distances=np.fromfile(f,dtype='<f4',count=int(n)*int(d))
        distances=raw_distances.reshape(int(n),int(d)) if len(raw_distances) else None
    with (OUT/'test_gt.bin').open('wb') as f:
        np.array([len(test_ids),d],dtype='<u4').tofile(f); ids[test_ids].tofile(f)
        if distances is not None: distances[test_ids].tofile(f)
    common={'DISKANN_M3_MODEL':str(model_file),'DISKANN_M3_BASE':str(base)}
    # tau=0 executes gate on the real trajectory but cannot skip, checking exact search parity.
    print('Verifying live tau=0 trajectory, state reconstruction, and C++ inference',flush=True)
    zero=execute('audit_tau0',allq,allgt,{**common,'DISKANN_M3_TAU':'0','DISKANN_M3_AUDIT':str(OUT/'audit_q0.bin')})
    actual_ids=mat(OUT/'audit_tau0_100_idx_uint32.bin','<u4')
    actual_dists=mat(OUT/'audit_tau0_100_dists_float.bin','<f4')
    assert np.array_equal(actual_ids,mat(E/'baseline_100_idx_uint32.bin','<u4'))
    assert np.array_equal(actual_dists,mat(E/'baseline_100_dists_float.bin','<f4'))
    assert abs(zero['recall_pct']-99.653)<1e-6
    audit=np.fromfile(OUT/'audit_q0.bin',dtype='<f4').reshape(-1,19)
    q0meta=None
    for split in ('train','validation','test'):
        m=np.load(DATA/split/'metadata.npy',mmap_mode='r'); ix=np.flatnonzero(m['query_id']==0)
        if len(ix):
            q0meta=m[ix]; reference=np.load(DATA/split/'pre_io_state.npy',mmap_mode='r')[ix];break
    by_node={int(m['node_id']):s for m,s in zip(q0meta,reference)}
    np.testing.assert_array_equal(audit[:,1:18],np.array([by_node[int(row[0])] for row in audit]))
    v=np.memmap(base,mode='r',dtype='<f4',offset=8,shape=(1000000,128))[audit[:,0].astype('uint32')]
    python_scores=predict_joint(np.repeat(queries[0:1],len(audit),axis=0),v,audit[:,1:18],bundle)
    np.testing.assert_allclose(audit[:,18],python_scores,atol=2e-6,rtol=2e-5)
    check=dict(tau0_exact_ordered_top10_and_distances_all_10000_queries=True,
               q0_online_pre_io_state_exactly_matches_offline=True,
               cpp_python_max_probability_difference=float(abs(audit[:,18]-python_scores).max()))
    (OUT/'verification.json').write_text(json.dumps(check,indent=2))
    print('Verification passed',check,flush=True)
    rng=np.random.default_rng(42); runs=[]
    configs={'baseline':None,**{k:v['threshold'] for k,v in chosen.items()}}
    for suite,q,g in [('test1000',OUT/'test_query.fbin',OUT/'test_gt.bin'),('all10000',allq,allgt)]:
        # One unreported baseline pass warms index pages; no privileged cache dropping.
        execute(suite+'_warmup',q,g,{})
        for repeat in range(3):
            for name in rng.permutation(list(configs)):
                tau=configs[name]; env={} if tau is None else {**common,'DISKANN_M3_TAU':str(tau)}
                label=f'{suite}_r{repeat}_{name}'
                row=execute(label,q,g,env); row.update(suite=suite,repeat=repeat,config=name,threshold=tau)
                runs.append(row); (OUT/'runs.json').write_text(json.dumps(runs,indent=2))
                print(label, 'Recall',row['recall_pct'],'Exp',row['expansions_per_query'],'Reads',row['ssd_reads_per_query'],'QPS',row['qps'],flush=True)
    summary=[]
    for suite in ('test1000','all10000'):
        for name,tau in configs.items():
            rows=[r for r in runs if r['suite']==suite and r['config']==name]
            assert len(rows)==3
            assert len({r['recall_pct'] for r in rows})==1
            a=dict(suite=suite,config=name,threshold=tau)
            for key in ('recall_pct','expansions_per_query','ssd_reads_per_query','latency_us','latency_p50_us','latency_p95_us','qps','gate_evaluations_per_query','skips_per_query'):
                a[key]=float(np.median([r[key] for r in rows]))
            a['qps_min']=min(r['qps'] for r in rows);a['qps_max']=max(r['qps'] for r in rows)
            summary.append(a)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    provenance=dict(model='M3',L=100,K=10,beam_width=4,threads=4,node_cache=10000,
        threshold_selection='validation only, fixed across live trajectories',repeats=3,order_seed=42,
        timing='median of three randomized-order measured passes after baseline warmup; host/OS cache uncontrolled',
        full_v_storage='512000000 bytes preloaded into RAM; model/base load outside timed search',
        latency_includes='per-candidate feature construction and real C++ MLP inference',
        ssd_reads='actual DiskANN index read requests; not hardware physical disk counters',
        primary_suite='1000 held-out test queries',all10000='includes train and validation; historical baseline comparison only',
        model_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(BIN.read_bytes()).hexdigest())
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    lines=['# M3 online L=100 sweep','','阈值来自 validation；每次候选选择时用当前在线 state、q/v 和交互项实时推理，在读盘前决定是否跳过，保留原 beam slot。没有复用 baseline 的跳过掩码。',
        'v 全表常驻 RAM（512 MB）；gate 特征构造和推理计入 Latency/QPS，模型/向量加载不计入。SSD Reads 是索引发出的读请求数。',
        '固定 K=10、L=100、beam=4、threads=4、cache=10000；各配置随机顺序重复三次，报告中位数。计时是预热后的当前主机结果，不是冷盘实验。','']
    for suite in ('test1000','all10000'):
        lines += ['## '+suite,'','| FNR target | tau | Recall@10 % | Expansion/query | SSD Reads/query | Mean latency us | QPS |','|---|---:|---:|---:|---:|---:|---:|']
        for a in summary:
            if a['suite']!=suite:continue
            tau='—' if a['threshold'] is None else f'{a["threshold"]:.9g}'
            lines.append(f'| {a["config"]} | {tau} | {a["recall_pct"]:.3f} | {a["expansions_per_query"]:.3f} | {a["ssd_reads_per_query"]:.3f} | {a["latency_us"]:.2f} | {a["qps"]:.2f} |')
    lines += ['','test1000 为未参与训练的 query；all10000 含训练/验证 query，仅用于对齐历史 Recall@10=99.653%。',
        'FNR target 是离线验证标签的阈值校准目标，不是在线召回损失保证。在线轨迹改变后，state 分布和候选集合也会改变。',
        'tau=0 全量搜索逐项复现历史结果；q0 的在线 pre_io_state 与离线完全一致，C++/Python 概率误差记录在 verification.json。',
        '每次搜索的命令、日志、结果 ID/距离和逐 query 统计均保留。重现：`.venv/bin/python experiments/sift1m/run_online_m3.py`。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(complete=True,live_gate=True,thresholds=5,repeats=3,heldout_test_reported=True),indent=2))
    print('ONLINE M3 COMPLETE',flush=True)

if __name__=='__main__':main()
