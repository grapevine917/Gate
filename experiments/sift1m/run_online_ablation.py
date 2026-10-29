#!/usr/bin/env python3
"""Real live Top1/Top3/All17 pruning, with interleaved baseline timing."""
import csv,hashlib,json
import joblib,numpy as np
from run_online_m3 import E,BIN,execute,mat
from train_model1 import predict
from train_state_baselines import dump
OUT=E/'feature_ablation/online'
MODELS=['Top1','Top3','All17']

def main():
    OUT.mkdir(exist_ok=True);(OUT/'COMPLETE.json').unlink(missing_ok=True)
    assert json.loads((E/'feature_ablation/COMPLETE.json').read_text())['complete']
    q=E/'online_m3/test_query.fbin';gt=E/'online_m3/test_gt.bin'
    qids=np.load(E/'training_data_l100/test/query_ids.npy')
    reference=mat(E/'baseline_100_idx_uint32.bin','<u4')[qids]
    reference_d=mat(E/'baseline_100_dists_float.bin','<f4')[qids]
    meta=np.load(E/'training_data_l100/test/metadata.npy',mmap_mode='r')
    idx=np.flatnonzero(meta['query_id']==qids[0])
    state=np.load(E/'training_data_l100/test/pre_io_state.npy',mmap_mode='r')[idx]
    by_node={int(m['node_id']):s for m,s in zip(meta[idx],state)}
    verification={};configs=[dict(model='DiskANN',key='baseline',target_fnr_pct=0,threshold=None,env={})]
    for label in MODELS:
        folder=E/'feature_ablation'/label;bundle=joblib.load(folder/'model.joblib')
        env={'DISKANN_M3_MODEL':str(folder/'model.bin')}
        audit=OUT/f'audit_{label}.bin'
        row=execute(f'audit_{label}',q,gt,{**env,'DISKANN_M3_TAU':'0',
            'DISKANN_M3_AUDIT':str(audit)},output_dir=OUT)
        np.testing.assert_array_equal(mat(OUT/f'audit_{label}_100_idx_uint32.bin','<u4'),reference)
        np.testing.assert_array_equal(mat(OUT/f'audit_{label}_100_dists_float.bin','<f4'),reference_d)
        a=np.fromfile(audit,dtype='<f4').reshape(-1,19)
        np.testing.assert_array_equal(a[:,1:18],np.array([by_node[int(r[0])] for r in a]))
        x=np.array(a[:,1:18][:,bundle['feature_indices']],dtype='float32',order='C')
        x-=bundle['mean'];x/=bundle['scale'];p=predict(bundle['model'],x)
        np.testing.assert_allclose(a[:,18],p,rtol=2e-5,atol=2e-6)
        assert row['skips_per_query']==0
        verification[label]=dict(tau0_ordered_ids_and_distances_match_all_test_queries=True,
            live_state_matches_offline=True,cpp_python_max_probability_error=float(abs(a[:,18]-p).max()),
            mapped_features=bundle['feature_names'])
        for op,c in json.loads((folder/'thresholds.json').read_text()).items():
            configs.append(dict(model=label,key=label+'_'+op,target_fnr_pct=c['target_retain_false_skip_rate']*100,
                threshold=c['threshold'],env={**env,'DISKANN_M3_TAU':str(c['threshold'])}))
    dump(OUT/'verification.json',verification)
    execute('warmup',q,gt,{},output_dir=OUT)
    rng=np.random.default_rng(42);runs=[]
    for repeat in range(3):
        for i in rng.permutation(len(configs)):
            c=configs[i];label=f'r{repeat}_{c["key"]}'
            row=execute(label,q,gt,c['env'],output_dir=OUT)
            row.update({k:v for k,v in c.items() if k!='env'});row['repeat']=repeat;runs.append(row)
            dump(OUT/'runs.json',runs)
            print(label,'Recall',row['recall_pct'],'Reads',row['ssd_reads_per_query'],
                  'Exp',row['expansions_per_query'],'Latency',row['latency_us'],flush=True)
    gtids=mat(gt,'<u4');summary=[]
    for c in configs:
        rows=[r for r in runs if r['key']==c['key']];assert len(rows)==3
        result={k:v for k,v in c.items() if k!='env'};expected=None
        for r in rows:
            ids=mat(OUT/(r['label']+'_100_idx_uint32.bin'),'<u4')
            if expected is None:expected=ids
            else:np.testing.assert_array_equal(ids,expected)
            recall=np.any(ids[:,:,None]==gtids[:,None,:10],axis=2).sum()/100
            assert abs(recall-r['recall_pct'])<1e-6
            stats=np.genfromtxt(OUT/(r['label']+'.csv'),delimiter=',',names=True)
            assert len(stats)==1000
            if r['model']=='DiskANN':assert not stats['gate_evaluated'].any()
            else:np.testing.assert_array_equal(stats['expansions'],stats['gate_evaluated']-stats['gate_skipped'])
            assert stats['ssd_reads'].sum()<=stats['expansions'].sum()
        for k in ['recall_pct','ssd_reads_per_query','expansions_per_query','latency_us','qps',
                  'latency_p50_us','latency_p95_us','gate_evaluations_per_query','skips_per_query']:
            result[k]=float(np.median([r[k] for r in rows]))
        result['latency_min_us']=min(r['latency_us'] for r in rows)
        result['latency_max_us']=max(r['latency_us'] for r in rows)
        summary.append(result)
    dump(OUT/'summary.json',summary)
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    prior=json.loads((E/'online_curves_state/joint_summary.json').read_text())
    base=summary[0]; old=next(r for r in prior if r['key']=='L100')
    for k in ['recall_pct','expansions_per_query','ssd_reads_per_query']:assert base[k]==old[k]
    # Fresh All17 retraining should reproduce the established state MLP.
    for r in summary:
        if r['model']!='All17':continue
        old=next(t for t in prior if t['key']==f'state_fnr_{r["target_fnr_pct"]:g}pct')
        for k in ['recall_pct','expansions_per_query','ssd_reads_per_query']:assert r[k]==old[k]
    verification['sweep']=dict(runs=len(runs),repeats=3,recalls_independently_recomputed=True,
        repeat_ordered_ids_identical=True,gate_counters_verified=True,
        baseline_and_all17_match_existing_behavior=True)
    dump(OUT/'verification.json',verification)
    dump(OUT/'provenance.json',dict(L=100,K=10,beam=4,threads=4,node_cache=10000,
        query_ids=str(E/'training_data_l100/test/query_ids.npy'),queries=1000,repeats=3,
        order_seed=42,thresholds='validation FNR targets .1%,.25%,.5%,1%,2%; fixed before online test',
        timing='median of three interleaved randomized passes after warmup; CPU inference included',
        storage='no full q/v feature table loaded',
        state_construction='same live 17-field reconstruction then mapped subset fed to MLP',
        ssd_reads='actual index read requests, not physical device counters',
        binary_sha256=hashlib.sha256(BIN.read_bytes()).hexdigest()))
    lines=['# Top1 / Top3 / All17 live DiskANN test','',
        '相同1000 held-out query；L=100、K=10、beam=4、threads=4、cache=10000。Top1/Top3/All17每次候选展开前使用当前live state推理；跳过保留原beam slot。不使用离线掩码，不加载完整v表。',
        '阈值只用validation；目标FNR不是在线召回损失保证。所有16配置交错随机重复3次，报告中位数，含特征构造和C++推理延迟。主机/OS缓存为预热状态。',
        '当前实现统一重建17维live state后按映射选择模型输入，尚未逐组优化state构造；Latency反映这一实现。',
        '', '| Model | val FNR target | tau | Recall@10 | SSD Reads/query | Expansions/query | Latency ms | QPS |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        tau='—' if r['threshold'] is None else f'{r["threshold"]:.9g}'
        lines.append(f'| {r["model"]} | {r["target_fnr_pct"]:g}% | {tau} | {r["recall_pct"]:.3f}% | {r["ssd_reads_per_query"]:.3f} | {r["expansions_per_query"]:.3f} | {r["latency_us"]/1000:.3f} | {r["qps"]:.1f} |')
    lines+=['','tau=0全部test query逐项复现baseline ID和距离；第一条test query的live state和Python/C++概率核对通过。全部48次测量重复ID一致，Recall独立重算、gate计数核对通过。新All17与既有state模型在线行为一致。',
        '所有命令、日志、逐query计数、结果ID/距离、概率审计均保留。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(OUT/'COMPLETE.json',dict(complete=True,live_gate=True,runs=len(runs),models=3,repeats=3))
    print('ONLINE ABLATION COMPLETE',flush=True)

if __name__=='__main__':main()
