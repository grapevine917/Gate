#!/usr/bin/env python3
"""Joint baseline/budget ET/learned ET/Top3 expansion-pruning real search sweep."""
import csv,hashlib,json
import numpy as np
from run_online_m3 import E,BIN,execute,mat
from train_state_baselines import dump
ROOT=E/'learned_termination';OUT=ROOT/'online'
BUDGETS=[20,40,60,70,80,85,90,95,100,105,110,120,140]

def measured(label,q,gt,env):
    row=execute(label,q,gt,env,output_dir=OUT)
    stats=np.genfromtxt(OUT/(label+'.csv'),delimiter=',',names=True)
    row.update(et_evaluations_per_query=float(stats['et_evaluated'].mean()),
               et_stopped_query_rate=float(stats['et_stopped'].mean()))
    return row

def main():
    assert json.loads((ROOT/'CALIBRATION_COMPLETE.json').read_text())['complete']
    OUT.mkdir(exist_ok=True);(OUT/'COMPLETE.json').unlink(missing_ok=True)
    q=E/'online_m3/test_query.fbin';gt=E/'online_m3/test_gt.bin'
    qids=np.load(E/'training_data_l100/test/query_ids.npy')
    baseline_ids=mat(E/'baseline_100_idx_uint32.bin','<u4')[qids]
    baseline_d=mat(E/'baseline_100_dists_float.bin','<f4')[qids]
    raw=np.load(ROOT/'data/test/state.npy');y=np.load(ROOT/'data/test/y_safe_stop.npy')
    sample_qids=np.load(ROOT/'data/test/sample_query_ids.npy')
    configs=[dict(method='DiskANN',key='baseline',parameter=0,threshold=None,budget=None,
                  risk_kind=None,env={})]
    for budget in BUDGETS:
        configs.append(dict(method='Budget Early Termination',key=f'budget{budget}',parameter=budget,
            threshold=None,budget=budget,risk_kind=None,env={'DISKANN_EARLY_IO_LIMIT':str(budget)}))
    native={};verification={}
    baseline_stats=np.genfromtxt(E/'feature_ablation/online/r0_baseline.csv',delimiter=',',names=True)
    for model in ('Top3','All17'):
        folder=ROOT/model;trace=OUT/f'{model}_test_native.f32'
        row=measured('native_'+model,q,gt,{'DISKANN_ET_MODEL':str(folder/'model.bin'),
            'DISKANN_ET_TAU':'1.1','DISKANN_ET_TRACE':str(trace)})
        np.testing.assert_array_equal(mat(OUT/f'native_{model}_100_idx_uint32.bin','<u4'),baseline_ids)
        np.testing.assert_array_equal(mat(OUT/f'native_{model}_100_dists_float.bin','<f4'),baseline_d)
        assert row['et_stopped_query_rate']==0 and row['et_evaluations_per_query']>0
        a=np.fromfile(trace,dtype='<f4').reshape(-1,21);local=a[:,0].astype('uint32');original=qids[local]
        order=np.lexsort((a[:,10],original));a=a[order];local=local[order];original=original[order]
        np.testing.assert_array_equal(original,sample_qids)
        np.testing.assert_array_equal(a[:,2:19],raw);np.testing.assert_array_equal(a[:,19],y)
        p=a[:,20];python=np.load(folder/'test_p_safe_stop.npy')
        np.testing.assert_allclose(p,python,atol=2e-6,rtol=2e-5)
        np.save(folder/'test_native_p_safe_stop.npy',p)
        native[model]=dict(a=a,p=p,local=local)
        verification[model]=dict(disabled_et_all_test_ids_and_distances_match_baseline=True,
            test_native_state_and_labels_match_collected_data=True,
            cpp_python_max_probability_error=float(abs(p-python).max()))
        for op,c in json.loads((folder/'thresholds.json').read_text()).items():
            configs.append(dict(method='Learned ET '+model,key=model+'_'+op,
                parameter=c['target_unsafe_stop_query_rate']*100,threshold=c['threshold'],budget=None,
                risk_kind='validation unsafe-stop queries %',env={'DISKANN_ET_MODEL':str(folder/'model.bin'),
                'DISKANN_ET_TAU':str(c['threshold'])}))
    folder=E/'feature_ablation/Top3'
    for op,c in json.loads((folder/'thresholds.json').read_text()).items():
        configs.append(dict(method='Learned Top3 Expansion Pruning',key='prune_'+op,
            parameter=c['target_retain_false_skip_rate']*100,threshold=c['threshold'],budget=None,
            risk_kind='validation retain candidate FNR %',env={'DISKANN_M3_MODEL':str(folder/'model.bin'),
            'DISKANN_M3_TAU':str(c['threshold'])}))
    dump(OUT/'configs.json',[{k:v for k,v in c.items() if k!='env'} for c in configs])
    dump(OUT/'verification.json',verification)
    measured('warmup',q,gt,{})
    rng=np.random.default_rng(42);runs=[]
    for repeat in range(3):
        for i in rng.permutation(len(configs)):
            c=configs[i];label=f'r{repeat}_{c["key"]}'
            row=measured(label,q,gt,c['env']);row.update({k:v for k,v in c.items() if k!='env'})
            row['repeat']=repeat;runs.append(row);dump(OUT/'runs.json',runs)
            print(label,'Recall',row['recall_pct'],'Reads',row['ssd_reads_per_query'],
                  'Exp',row['expansions_per_query'],'Latency',row['latency_us'],flush=True)
    gtids=mat(gt,'<u4');summary=[]
    previous=json.loads((E/'feature_ablation/online/summary.json').read_text())
    for c in configs:
        trials=[r for r in runs if r['key']==c['key']];assert len(trials)==3
        result={k:v for k,v in c.items() if k!='env'};expected=None;et_expected=None
        if c['method'].startswith('Learned ET'):
            model=c['method'].split()[-1];a=native[model]['a'];p=native[model]['p'];local=native[model]['local']
            counts=baseline_stats['expansions'].copy();stopped=np.zeros(1000,dtype='uint8')
            unsafe=np.zeros(1000,dtype='uint8');evals=np.zeros(1000,dtype='uint32')
            for query in range(1000):
                ix=np.flatnonzero(local==query);hit=np.flatnonzero(p[ix]>=c['threshold'])
                if len(hit):
                    first=ix[hit[0]];counts[query]=a[first,9];stopped[query]=1;unsafe[query]=1-int(y[first])
                    evals[query]=hit[0]+1
                else:evals[query]=len(ix)
            et_expected=dict(counts=counts,stopped=stopped,unsafe=unsafe,evals=evals)
            result.update(test_unsafe_stop_query_rate=float(unsafe.mean()),test_safe_stop_query_rate=float((stopped-unsafe).mean()),
                          test_stopped_queries=int(stopped.sum()),test_unsafe_stopped_queries=int(unsafe.sum()))
        else:
            result.update(test_unsafe_stop_query_rate=None,test_safe_stop_query_rate=None,
                          test_stopped_queries=None,test_unsafe_stopped_queries=None)
        for r in trials:
            ids=mat(OUT/(r['label']+'_100_idx_uint32.bin'),'<u4')
            dists=mat(OUT/(r['label']+'_100_dists_float.bin'),'<f4')
            if expected is None:expected=ids
            else:np.testing.assert_array_equal(ids,expected)
            recall=np.any(ids[:,:,None]==gtids[:,None,:10],axis=2).sum()/100
            assert abs(recall-r['recall_pct'])<1e-6
            stats=np.genfromtxt(OUT/(r['label']+'.csv'),delimiter=',',names=True);assert len(stats)==1000
            assert np.all(stats['ssd_reads']<=stats['expansions'])
            if c['method']=='DiskANN':
                assert not stats['gate_evaluated'].any() and not stats['et_evaluated'].any()
                np.testing.assert_array_equal(ids,baseline_ids)
            elif c['method']=='Budget Early Termination':
                assert stats['ssd_reads'].max()<=c['budget']+3
                assert not stats['et_evaluated'].any() and not stats['gate_evaluated'].any()
                if c['budget']==140:np.testing.assert_array_equal(ids,baseline_ids)
            elif et_expected:
                assert not stats['gate_evaluated'].any()
                np.testing.assert_array_equal(stats['expansions'],et_expected['counts'])
                np.testing.assert_array_equal(stats['et_stopped'],et_expected['stopped'])
                np.testing.assert_array_equal(stats['et_evaluated'],et_expected['evals'])
                safe=~et_expected['unsafe'].astype(bool)
                np.testing.assert_array_equal(ids[safe],baseline_ids[safe])
                np.testing.assert_array_equal(dists[safe],baseline_d[safe])
                changed=np.any(ids!=baseline_ids,axis=1)
                np.testing.assert_array_equal(changed,et_expected['unsafe'].astype(bool))
            else:
                np.testing.assert_array_equal(stats['expansions'],stats['gate_evaluated']-stats['gate_skipped'])
                assert not stats['et_evaluated'].any()
        for key in ('recall_pct','ssd_reads_per_query','expansions_per_query','latency_us','qps',
                    'latency_p50_us','latency_p95_us','gate_evaluations_per_query','skips_per_query',
                    'et_evaluations_per_query','et_stopped_query_rate'):
            result[key]=float(np.median([r[key] for r in trials]))
        result['latency_min_us']=min(r['latency_us'] for r in trials)
        result['latency_max_us']=max(r['latency_us'] for r in trials)
        if c['method']=='DiskANN' or c['method']=='Learned Top3 Expansion Pruning':
            oldkey='baseline' if c['method']=='DiskANN' else 'Top3_'+c['key'].removeprefix('prune_')
            old=next(t for t in previous if t['key']==oldkey)
            for key in ('recall_pct','ssd_reads_per_query','expansions_per_query'):assert result[key]==old[key]
        summary.append(result)
    dump(OUT/'summary.json',summary)
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    verification.update(measured_runs=len(runs),repeat_ids_identical=True,recalls_independently_recomputed=True,
        budget_max_read_overshoot=3,et_first_stop_prediction_matches_actual_expansions_and_evaluations=True,
        all_safe_terminations_preserve_exact_baseline_ids_and_distances=True,
        unsafe_terminations_exactly_match_changed_baseline_ids=True,
        baseline_and_top3_pruning_match_previous_behavior=True)
    dump(OUT/'verification.json',verification)
    dump(OUT/'provenance.json',dict(dataset='SIFT1M',L=100,K=10,beam=4,threads=4,node_cache=10000,
        query_ids=str(E/'training_data_l100/test/query_ids.npy'),queries=1000,repeats=3,order_seed=42,
        budget_et='existing SSD I/O budget cutoff at beam boundary, at most3 additional reads',
        learned_et='whole-search break before next beam, only after >=K exact-scored candidates',
        et_label='baseline final top10 already scored',et_risk='fraction of validation queries whose first predicted stop is unsafe',
        pruning_risk='fraction of validation conservative retain candidate labels skipped',
        risk_targets_are_not_identical_metrics=True,selection='all models and thresholds selected on train/validation only',
        timing='three joint randomized passes, median warm-host latency; state construction and inference included',
        ssd_reads='index requests, not physical device counters',
        binary_sha256=hashlib.sha256(BIN.read_bytes()).hexdigest()))
    dump(OUT/'COMPLETE.json',dict(complete=True,learned_et_models=2,repeats=3,configs=len(configs),runs=len(runs)))
    print('LEARNED TERMINATION ONLINE COMPARISON COMPLETE',flush=True)

if __name__=='__main__':main()
