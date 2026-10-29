#!/usr/bin/env python3
"""Fixed validation-selected ET+pruning configurations and freshly timed controls."""
import csv,hashlib,json
import joblib,numpy as np
from run_online_m3 import E,BIN,mat
from train_model1 import predict
from train_state_baselines import dump
from calibrate_et_pruning import OUT,ET_MODEL,PRUNE_MODEL,env_for,measured,first_stops,verify_path,hits,model_hashes
ONLINE=OUT/'online'
BUDGETS=[20,40,60,70,80,85,90,95,100,105,110,120,140]

def ids_dists(folder,label):
    return mat(folder/(label+'_100_idx_uint32.bin'),'<u4'),mat(folder/(label+'_100_dists_float.bin'),'<f4')

def main():
    assert json.loads((OUT/'CALIBRATION_COMPLETE.json').read_text())['complete']
    provenance=json.loads((OUT/'calibration_provenance.json').read_text());assert model_hashes()==provenance['model_sha256']
    selected=json.loads((OUT/'selected_joint_configs.json').read_text())
    ONLINE.mkdir(exist_ok=True);(ONLINE/'COMPLETE.json').unlink(missing_ok=True)
    q=E/'online_m3/test_query.fbin';gt=E/'online_m3/test_gt.bin';gtids=mat(gt,'<u4')
    qids=np.load(E/'training_data_l100/test/query_ids.npy')
    base_ids=mat(E/'baseline_100_idx_uint32.bin','<u4')[qids]
    base_d=mat(E/'baseline_100_dists_float.bin','<f4')[qids]
    bundle=joblib.load(E/'learned_termination/All17/model.joblib');verification={}
    # Both disabled -> baseline; candidate pruning probability audit checks independent model wiring.
    audit=ONLINE/'both_disabled_pruning_audit.f32'
    row,stats=measured('both_disabled',q,gt,{**env_for(0,1.1),'DISKANN_M3_AUDIT':str(audit)},ONLINE)
    actual,d=ids_dists(ONLINE,'both_disabled')
    np.testing.assert_array_equal(actual,base_ids);np.testing.assert_array_equal(d,base_d)
    assert not stats['gate_skipped'].any() and not stats['et_stopped'].any()
    a=np.fromfile(audit,dtype='<f4').reshape(-1,19)
    prune_bundle=joblib.load(E/'feature_ablation/Top3/model.joblib')
    x=np.array(a[:,1:18][:,prune_bundle['feature_indices']],dtype='float32',order='C')
    x-=prune_bundle['mean'];x/=prune_bundle['scale'];p=predict(prune_bundle['model'],x)
    np.testing.assert_allclose(a[:,18],p,atol=2e-6,rtol=2e-5)
    verification['both_disabled']=dict(all_test_ids_and_distances_match_baseline=True,
        both_models_evaluate_without_actions=True,pruning_cpp_python_probability_max_error=float(abs(a[:,18]-p).max()))
    # Pruning disabled -> an existing real ET operating point, with identical behavior.
    et_params=json.loads((E/'learned_termination/All17/thresholds.json').read_text())
    c=et_params['unsafe_query_0.25pct']
    row,_=measured('pruning_disabled',q,gt,env_for(0,c['threshold']),ONLINE)
    old_folder=E/'learned_termination/online';old_label='r0_All17_unsafe_query_0.25pct'
    actual,d=ids_dists(ONLINE,'pruning_disabled');ref,ref_d=ids_dists(old_folder,old_label)
    np.testing.assert_array_equal(actual,ref);np.testing.assert_array_equal(d,ref_d)
    verification['pruning_disabled']=dict(all_test_ids_distances_match_standalone_et=True)
    native={};prune_params=json.loads((E/'feature_ablation/Top3/thresholds.json').read_text())
    # Freeze selected combos before native test diagnostics. No test selection is performed.
    for prune_key in sorted(set(c['prune_key'] for c in selected)):
        tau=prune_params[prune_key]['threshold'];label='native_'+prune_key;trace=ONLINE/(label+'.f32')
        full,full_stats=measured(label,q,gt,{**env_for(tau,1.1),'DISKANN_ET_TRACE':str(trace)},ONLINE)
        full_ids,full_d=ids_dists(ONLINE,label)
        old_label='r0_Top3_'+prune_key;old_folder=E/'feature_ablation/online'
        ref,ref_d=ids_dists(old_folder,old_label)
        np.testing.assert_array_equal(full_ids,ref);np.testing.assert_array_equal(full_d,ref_d)
        old_stats=np.genfromtxt(old_folder/(old_label+'.csv'),delimiter=',',names=True)
        for column in ('expansions','ssd_reads','gate_evaluated','gate_skipped'):
            np.testing.assert_array_equal(full_stats[column],old_stats[column])
        assert not full_stats['et_stopped'].any()
        a=np.fromfile(trace,dtype='<f4').reshape(-1,21);a=a[np.lexsort((a[:,10],a[:,0]))]
        x=np.array(a[:,2:19][:,bundle['feature_indices']],dtype='float32',order='C');x-=bundle['mean'];x/=bundle['scale']
        p=predict(bundle['model'],x);np.testing.assert_allclose(a[:,20],p,atol=2e-6,rtol=2e-5)
        native[prune_key]=dict(a=a,stats=full_stats,ids=full_ids,d=full_d)
        verification[prune_key]=dict(disabled_et_matches_standalone_pruning_ids_distances_and_counters=True,
            et_cpp_python_probability_max_error=float(abs(a[:,20]-p).max()))
    configs=[dict(method='DiskANN',key='baseline',parameter=0,et_tau=None,prune_tau=None,
        budget=None,prune_key=None,selected_for_recall_budgets_pp=[],env={})]
    for budget in BUDGETS:
        configs.append(dict(method='Budget Early Termination',key=f'budget{budget}',parameter=budget,
            et_tau=None,prune_tau=None,budget=budget,prune_key=None,selected_for_recall_budgets_pp=[],
            env={'DISKANN_EARLY_IO_LIMIT':str(budget)}))
    for op,c in et_params.items():
        configs.append(dict(method='Learned ET All17',key='et_'+op,parameter=c['target_unsafe_stop_query_rate']*100,
            et_tau=c['threshold'],prune_tau=None,budget=None,prune_key=None,
            selected_for_recall_budgets_pp=[],env=env_for(None,c['threshold'])))
    for op,c in prune_params.items():
        configs.append(dict(method='Learned Top3 Expansion Pruning',key='prune_'+op,
            parameter=c['target_retain_false_skip_rate']*100,et_tau=None,prune_tau=c['threshold'],
            budget=None,prune_key=op,selected_for_recall_budgets_pp=[],env=env_for(c['threshold'],None)))
    for c in selected:
        configs.append(dict(method='ET + Expansion Pruning',key='joint_'+c['key'],
            parameter=min(c['selected_for_recall_budgets_pp']),et_tau=c['et_tau'],prune_tau=c['prune_tau'],
            budget=None,prune_key=c['prune_key'],nominal_prune_fnr_pct=c['nominal_prune_fnr_pct'],
            conditional_et_query_target_pct=c['conditional_et_query_target_pct'],
            selected_for_recall_budgets_pp=c['selected_for_recall_budgets_pp'],env=env_for(c['prune_tau'],c['et_tau'])))
    dump(ONLINE/'configs.json',[{k:v for k,v in c.items() if k!='env'} for c in configs])
    dump(ONLINE/'verification.json',verification)
    measured('warmup',q,gt,{},ONLINE);rng=np.random.default_rng(42);runs=[]
    for repeat in range(3):
        for i in rng.permutation(len(configs)):
            c=configs[i];label=f'r{repeat}_{c["key"]}'
            row,_=measured(label,q,gt,c['env'],ONLINE)
            row.update({k:v for k,v in c.items() if k!='env'});row['repeat']=repeat;runs.append(row)
            dump(ONLINE/'runs.json',runs)
            print(label,'Recall',row['recall_pct'],'Reads',row['ssd_reads_per_query'],
                  'Exp',row['expansions_per_query'],'Latency',row['latency_us'],flush=True)
    old=json.loads((E/'learned_termination/online/summary.json').read_text());summary=[]
    for c in configs:
        trials=[r for r in runs if r['key']==c['key']];assert len(trials)==3
        result={k:v for k,v in c.items() if k!='env'};expected=None;pred=None
        if c['method']=='ET + Expansion Pruning':
            parent=native[c['prune_key']];pred=first_stops(parent['a'],c['et_tau'],parent['stats']['expansions'])
            result.update(test_conditional_et_unsafe_queries=int(pred['unsafe'].sum()),
                test_conditional_et_unsafe_query_rate=float(pred['unsafe'].mean()))
        for r in trials:
            ids,d=ids_dists(ONLINE,r['label'])
            if expected is None:expected=ids
            else:np.testing.assert_array_equal(ids,expected)
            assert abs(hits(ids,gtids)/100-r['recall_pct'])<1e-6
            stats=np.genfromtxt(ONLINE/(r['label']+'.csv'),delimiter=',',names=True)
            assert len(stats)==1000 and np.all(stats['ssd_reads']<=stats['expansions'])
            if c['method']=='ET + Expansion Pruning':verify_path(stats,ids,d,parent['ids'],parent['d'],pred)
            elif c['method']=='Learned Top3 Expansion Pruning':
                np.testing.assert_array_equal(stats['expansions'],stats['gate_evaluated']-stats['gate_skipped'])
                assert not stats['et_evaluated'].any()
            elif c['method']=='Learned ET All17':
                assert not stats['gate_evaluated'].any() and np.all(stats['et_stopped']<=1)
            else:
                assert not stats['et_evaluated'].any() and not stats['gate_evaluated'].any()
                if c['budget'] is not None:assert stats['ssd_reads'].max()<=c['budget']+3
                else:np.testing.assert_array_equal(ids,base_ids)
        result['test_changed_baseline_queries']=int(np.any(expected!=base_ids,axis=1).sum())
        for key in ('recall_pct','ssd_reads_per_query','expansions_per_query','latency_us','qps',
                    'latency_p50_us','latency_p95_us','gate_evaluations_per_query','skips_per_query',
                    'et_evaluations_per_query','et_stopped_query_rate'):
            result[key]=float(np.median([r[key] for r in trials]))
        result['latency_min_us']=min(r['latency_us'] for r in trials);result['latency_max_us']=max(r['latency_us'] for r in trials)
        if c['method']!='ET + Expansion Pruning':
            oldkey=('All17_'+c['key'].removeprefix('et_')) if c['method']=='Learned ET All17' else c['key']
            previous=next(r for r in old if r['key']==oldkey)
            for key in ('recall_pct','ssd_reads_per_query','expansions_per_query'):assert result[key]==previous[key]
        summary.append(result)
    dump(ONLINE/'summary.json',summary)
    with (ONLINE/'summary.csv').open('w',newline='') as f:
        fields=list(dict.fromkeys(k for r in summary for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary)
    verification.update(measured_runs=len(runs),repeat_ids_identical=True,recalls_independently_recomputed=True,
        joint_first_stop_expansion_counts_and_et_evaluations_match_native_pruned_paths=True,
        safe_et_preserves_exact_pruning_only_endpoint=True,gate_counter_invariant_verified=True,
        controls_match_previous_search_behavior=True)
    dump(ONLINE/'verification.json',verification)
    dump(ONLINE/'provenance.json',dict(SIFT1M=True,L=100,K=10,beam=4,threads=4,node_cache=10000,queries=1000,
        query_ids=str(E/'training_data_l100/test/query_ids.npy'),repeats=3,order_seed=42,
        model_sha256=model_hashes(),selection='validation-only conditional ET calibration + real validation Recall/read grid selection',
        semantics='ET at beam boundary, pruning at each candidate before IO; skipped candidate consumes original beam slot',
        timing='all configurations jointly randomized, median warm-host timings, both model inferences and state construction included',
        ssd_reads='index read requests, not physical device counters',
        binary_sha256=hashlib.sha256(BIN.read_bytes()).hexdigest()))
    dump(ONLINE/'COMPLETE.json',dict(complete=True,configs=len(configs),measured_runs=len(runs),
        joint_points=len(selected),validation_selected=True,repeats=3))
    print('ET + PRUNING ONLINE COMPLETE',flush=True)

if __name__=='__main__':main()
