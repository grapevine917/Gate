#!/usr/bin/env python3
"""Joint live heuristic and learned gate comparison on fixed test queries."""
import csv,hashlib,json
import numpy as np
from run_online_m3 import E,BIN,execute,mat
from train_state_baselines import dump
OUT=E/'top3_heuristic/online'

def heuristic_env(c):
    return {'DISKANN_HEURISTIC_R':str(c['r']),'DISKANN_HEURISTIC_E':str(c['e'])}

def main():
    OUT.mkdir(exist_ok=True);(OUT/'COMPLETE.json').unlink(missing_ok=True)
    assert json.loads((OUT.parent/'CALIBRATED.json').read_text())['complete']
    selected=json.loads((OUT.parent/'selected_parameters.json').read_text())
    q=E/'online_m3/test_query.fbin';gt=E/'online_m3/test_gt.bin'
    qids=np.load(E/'training_data_l100/test/query_ids.npy')
    reference=mat(E/'baseline_100_idx_uint32.bin','<u4')[qids]
    reference_d=mat(E/'baseline_100_dists_float.bin','<f4')[qids]
    meta=np.load(E/'training_data_l100/test/metadata.npy',mmap_mode='r')
    ix=np.flatnonzero(meta['query_id']==qids[0])
    state=np.load(E/'training_data_l100/test/pre_io_state.npy',mmap_mode='r')[ix]
    by_node={int(m['node_id']):s for m,s in zip(meta[ix],state)}
    # No-skip path verifies rank is captured at the same point as learned/offline state.
    audit=OUT/'audit_disabled.bin'
    row=execute('audit_disabled',q,gt,{'DISKANN_HEURISTIC_R':'99','DISKANN_M3_AUDIT':str(audit)},output_dir=OUT)
    np.testing.assert_array_equal(mat(OUT/'audit_disabled_100_idx_uint32.bin','<u4'),reference)
    np.testing.assert_array_equal(mat(OUT/'audit_disabled_100_dists_float.bin','<f4'),reference_d)
    a=np.fromfile(audit,dtype='<f4').reshape(-1,19)
    np.testing.assert_array_equal(a[:,1:18],np.array([by_node[int(r[0])] for r in a]))
    assert row['skips_per_query']==0 and np.all(a[:,18]==1)
    verification=dict(disabled_rule_exact_baseline_ids_distances_all_test=True,
        disabled_rule_live_state_matches_offline=True,rule_decision_audits={})
    audited={}
    for method in ('Rank','Rank+Progress'):
        c=next(v for v in selected if v['method']==method and v['target_fnr_pct']==1)
        audit=OUT/(c['key']+'_audit.bin');label='audit_'+c['key']
        execute(label,q,gt,{**heuristic_env(c),'DISKANN_M3_AUDIT':str(audit)},output_dir=OUT)
        a=np.fromfile(audit,dtype='<f4').reshape(-1,19)
        skip=(a[:,4]>c['r'])&(a[:,8]>c['e']) # record=node,state17,decision; state rank offset3 and count7
        np.testing.assert_array_equal(a[:,18],(~skip).astype('float32'))
        assert np.all(np.diff(a[:,8])>=0)
        verification['rule_decision_audits'][method]=dict(r=c['r'],e=c['e'],
            cpp_matches_python_on_live_state=True,evaluations=len(a),skipped=int(skip.sum()))
        audited[c['key']]=mat(OUT/(label+'_100_idx_uint32.bin'),'<u4')
    dump(OUT/'verification.json',verification)
    configs=[dict(method='DiskANN',key='baseline',target_fnr_pct=0,r=None,e=None,threshold=None,env={})]
    for c in selected:
        configs.append(dict(method=c['method'],key=c['key'],target_fnr_pct=c['target_fnr_pct'],
            r=c['r'],e=c['e'],threshold=None,env=heuristic_env(c)))
    for model in ('Top3','All17'):
        folder=E/'feature_ablation'/model
        for op,c in json.loads((folder/'thresholds.json').read_text()).items():
            configs.append(dict(method='Learned '+model,key=model+'_'+op,
                target_fnr_pct=c['target_retain_false_skip_rate']*100,r=None,e=None,
                threshold=c['threshold'],env={'DISKANN_M3_MODEL':str(folder/'model.bin'),
                'DISKANN_M3_TAU':str(c['threshold'])}))
    dump(OUT/'configs.json',[{k:v for k,v in c.items() if k!='env'} for c in configs])
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
    old=json.loads((E/'feature_ablation/online/summary.json').read_text())
    for c in configs:
        trials=[r for r in runs if r['key']==c['key']];assert len(trials)==3
        result={k:v for k,v in c.items() if k!='env'};expected=None
        for r in trials:
            ids=mat(OUT/(r['label']+'_100_idx_uint32.bin'),'<u4')
            if expected is None:expected=ids
            else:np.testing.assert_array_equal(ids,expected)
            recall=np.any(ids[:,:,None]==gtids[:,None,:10],axis=2).sum()/100
            assert abs(recall-r['recall_pct'])<1e-6
            stats=np.genfromtxt(OUT/(r['label']+'.csv'),delimiter=',',names=True)
            assert len(stats)==1000
            if c['method']=='DiskANN':assert not stats['gate_evaluated'].any()
            else:np.testing.assert_array_equal(stats['expansions'],stats['gate_evaluated']-stats['gate_skipped'])
            assert np.all(stats['ssd_reads']<=stats['expansions'])
        if c['key'] in audited:np.testing.assert_array_equal(expected,audited[c['key']])
        if c['method']=='DiskANN':np.testing.assert_array_equal(expected,reference)
        for key in ('recall_pct','ssd_reads_per_query','expansions_per_query','latency_us','qps',
                    'latency_p50_us','latency_p95_us','gate_evaluations_per_query','skips_per_query'):
            result[key]=float(np.median([r[key] for r in trials]))
        result['latency_min_us']=min(r['latency_us'] for r in trials)
        result['latency_max_us']=max(r['latency_us'] for r in trials)
        if c['method']=='DiskANN' or c['method'].startswith('Learned'):
            previous=next(r for r in old if r['key']==c['key'])
            for key in ('recall_pct','ssd_reads_per_query','expansions_per_query'):assert result[key]==previous[key]
        summary.append(result)
    dump(OUT/'summary.json',summary)
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    verification.update(measured_runs=len(runs),repeat_ids_identical=True,recalls_independently_recomputed=True,
        gate_counters_verified=True,audit_and_non_audit_results_identical=True,
        baseline_and_learned_behavior_match_previous=True)
    dump(OUT/'verification.json',verification)
    dump(OUT/'provenance.json',dict(L=100,K=10,beam=4,threads=4,node_cache=10000,
        query_ids=str(E/'training_data_l100/test/query_ids.npy'),queries=1000,repeats=3,order_seed=42,
        timing='median of three joint randomized passes after warmup; decisions included',
        heuristics='direct integer comparisons; no model or vector feature table loaded',
        rule_state='live zero-based queue rank before mark-expanded; completed exact-scored expansion count',
        skip_semantics='same original beam slot consumed as learned expansion pruning',
        learned_state='common 17-state reconstruction then Top3/All17 MLP; inference included',
        selection='offline validation grid: max SkipRate under same conservative retain FNR targets as learned',
        ssd_reads='actual index read requests, not physical device counters',
        binary_sha256=hashlib.sha256(BIN.read_bytes()).hexdigest()))
    lines=['# Heuristic vs learned live pruning','',
        '固定L=100、K=10、beam=4、threads=4、cache=10000；相同1000条test query。所有21配置重新交错随机重复3次，取中位数。推理或规则判断计入Latency；主机/OS缓存为预热状态。',
        'Rank: rank>r；Rank+Progress: rank>r AND completed_expansions>e。rank为零起始队列位置，completed只计已精确评分展开。两规则使用Top3中的前两个特征，不使用visited_count。',
        '参数在离线validation baseline状态上穷举，最大化SkipRate且满足与learned相同的保守retain FNR目标0.1%、0.25%、0.5%、1%、2%。未在test选择r/e或tau；真实在线轨迹会改变，因此FNR目标不是召回保证。',
        '规则只做必要的rank查找和整数比较；learned保留原有live state构造及MLP推理。所有方法跳过时消耗原beam slot。',
        '', '| Method | val FNR target | r | e | tau | Recall@10 | SSD Reads/query | Expansions/query | Latency ms | QPS |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        tau='—' if r['threshold'] is None else f'{r["threshold"]:.9g}'
        rank='—' if r['r'] is None else str(r['r']);exp='—' if r['e'] is None or r['method']=='Rank' else str(r['e'])
        lines.append(f'| {r["method"]} | {r["target_fnr_pct"]:g}% | {rank} | {exp} | {tau} | {r["recall_pct"]:.3f}% | {r["ssd_reads_per_query"]:.3f} | {r["expansions_per_query"]:.3f} | {r["latency_us"]/1000:.3f} | {r["qps"]:.1f} |')
    lines+=['','关闭规则时逐项复现baseline；live rule决策与Python核对；audit开关不改变结果；63次测量重复ID一致、Recall独立重算和gate计数核对通过。learned模型与上次在线搜索行为一致。',
        'SSD Reads指索引读取请求；Latency是当前主机预热结果，不是冷盘性能。查询原本用于feature importance/ablation，属于探索性对照。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(OUT/'COMPLETE.json',dict(complete=True,live_rules=True,learned_models=2,rules=2,runs=len(runs),repeats=3))
    print('ONLINE HEURISTIC COMPARISON COMPLETE',flush=True)

if __name__=='__main__':main()
