#!/usr/bin/env python3
"""Recalibrate ET on pruned validation paths; select joint thresholds by real recall/I/O."""
import csv,hashlib,json
import joblib,numpy as np
from run_online_m3 import E,BIN,execute,mat
from train_model1 import predict
from train_state_baselines import dump
OUT=E/'et_pruning';VAL=OUT/'validation'
ET_MODEL=E/'learned_termination/All17/model.bin'
PRUNE_MODEL=E/'feature_ablation/Top3/model.bin'
RISKS=[0,.001,.0025,.005,.01,.02,.05,.10]
RECALL_BUDGETS=[0,.01,.025,.05,.1,.2,.3,.5,.75,1,2]

def model_hashes():
    return dict(et=hashlib.sha256(ET_MODEL.read_bytes()).hexdigest(),
                pruning=hashlib.sha256(PRUNE_MODEL.read_bytes()).hexdigest())

def env_for(prune_tau=None,et_tau=None):
    env={}
    if prune_tau is not None:env.update(DISKANN_M3_MODEL=str(PRUNE_MODEL),DISKANN_M3_TAU=str(prune_tau))
    if et_tau is not None:env.update(DISKANN_ET_MODEL=str(ET_MODEL),DISKANN_ET_TAU=str(et_tau))
    return env

def measured(label,q,gt,env,out):
    row=execute(label,q,gt,env,output_dir=out)
    stats=np.genfromtxt(out/(label+'.csv'),delimiter=',',names=True)
    row.update(et_evaluations_per_query=float(stats['et_evaluated'].mean()),
               et_stopped_query_rate=float(stats['et_stopped'].mean()))
    return row,stats

def first_stops(a,tau,full_counts):
    local=a[:,0].astype('uint32');p=a[:,20];y=a[:,19]
    counts=full_counts.copy();stopped=np.zeros(1000,dtype='uint8')
    unsafe=np.zeros(1000,dtype='uint8');evals=np.zeros(1000,dtype='uint32')
    for query in range(1000):
        ix=np.flatnonzero(local==query);hit=np.flatnonzero(p[ix]>=tau)
        if len(hit):
            first=ix[hit[0]];counts[query]=a[first,9];stopped[query]=1;unsafe[query]=1-int(y[first])
            evals[query]=hit[0]+1
        else:evals[query]=len(ix)
    return dict(counts=counts,stopped=stopped,unsafe=unsafe,evals=evals)

def verify_path(stats,ids,dists,full_ids,full_dists,pred):
    np.testing.assert_array_equal(stats['expansions'],pred['counts'])
    np.testing.assert_array_equal(stats['et_stopped'],pred['stopped'])
    np.testing.assert_array_equal(stats['et_evaluated'],pred['evals'])
    np.testing.assert_array_equal(stats['expansions'],stats['gate_evaluated']-stats['gate_skipped'])
    safe=~pred['unsafe'].astype(bool)
    np.testing.assert_array_equal(ids[safe],full_ids[safe]);np.testing.assert_array_equal(dists[safe],full_dists[safe])
    np.testing.assert_array_equal(np.any(ids!=full_ids,axis=1),pred['unsafe'].astype(bool))

def hits(ids,gt):
    return int(np.any(ids[:,:,None]==gt[:,None,:10],axis=2).sum())

def main():
    OUT.mkdir(exist_ok=True);VAL.mkdir(exist_ok=True);(OUT/'CALIBRATION_COMPLETE.json').unlink(missing_ok=True)
    q=E/'learned_termination/validation_query.fbin';gt=E/'learned_termination/validation_gt.bin';gtids=mat(gt,'<u4')
    base,base_stats=measured('baseline',q,gt,{},VAL)
    baseids=mat(VAL/'baseline_100_idx_uint32.bin','<u4');basehits=hits(baseids,gtids)
    assert abs(basehits/100-base['recall_pct'])<1e-6
    prior_ids=np.load(E/'training_data_l100/validation/query_ids.npy')
    np.testing.assert_array_equal(baseids,mat(E/'baseline_100_idx_uint32.bin','<u4')[prior_ids])
    pruning=json.loads((E/'feature_ablation/Top3/thresholds.json').read_text())
    bundle=joblib.load(E/'learned_termination/All17/model.joblib')
    conditional={};grid=[];verification={}
    for prune_key,c in pruning.items():
        tau=c['threshold'];trace=VAL/(prune_key+'_native.f32');label='native_'+prune_key
        full,full_stats=measured(label,q,gt,{**env_for(tau,1.1),'DISKANN_ET_TRACE':str(trace)},VAL)
        full_ids=mat(VAL/(label+'_100_idx_uint32.bin'),'<u4');full_d=mat(VAL/(label+'_100_dists_float.bin'),'<f4')
        only,only_stats=measured('prune_only_'+prune_key,q,gt,env_for(tau,None),VAL)
        for suffix,dtype,expected in [('idx_uint32','<u4',full_ids),('dists_float','<f4',full_d)]:
            np.testing.assert_array_equal(mat(VAL/f'prune_only_{prune_key}_100_{suffix}.bin',dtype),expected)
        np.testing.assert_array_equal(full_stats['expansions'],only_stats['expansions'])
        np.testing.assert_array_equal(full_stats['ssd_reads'],only_stats['ssd_reads'])
        np.testing.assert_array_equal(full_stats['expansions'],full_stats['gate_evaluated']-full_stats['gate_skipped'])
        assert not full_stats['et_stopped'].any()
        a=np.fromfile(trace,dtype='<f4').reshape(-1,21);a=a[np.lexsort((a[:,10],a[:,0]))]
        assert np.isfinite(a).all() and np.all(a[:,9]>=10)
        same=a[1:,0]==a[:-1,0];assert np.all(np.diff(a[:,19])[same]>=0)
        local=a[:,0].astype('uint32');p=a[:,20];y=a[:,19]
        x=np.array(a[:,2:19][:,bundle['feature_indices']],dtype='float32',order='C');x-=bundle['mean'];x/=bundle['scale']
        python=predict(bundle['model'],x);np.testing.assert_allclose(p,python,atol=2e-6,rtol=2e-5)
        unsafe_max=np.full(1000,-1,dtype='float32');np.maximum.at(unsafe_max,local[y==0],p[y==0])
        sorted_max=np.sort(unsafe_max)[::-1];choices={}
        for risk in RISKS:
            budget=int(np.floor(risk*1000));et_tau=max(0,float(np.nextafter(sorted_max[budget],np.float32(np.inf))))
            et_key=f'et_query_{risk*100:g}pct';key=prune_key+'_'+et_key
            pred=first_stops(a,et_tau,full_stats['expansions'])
            assert pred['unsafe'].sum()<=budget
            choice=dict(key=key,prune_key=prune_key,prune_tau=tau,
                nominal_prune_fnr_pct=c['target_retain_false_skip_rate']*100,
                et_tau=et_tau,conditional_et_query_target_pct=risk*100,
                validation_conditional_unsafe_query_rate=float(pred['unsafe'].mean()),
                calibration_reference='full pruning-only result at this fixed pruning threshold')
            choices[et_key]=choice
        conditional[prune_key]=choices
        # Choices are frozen from native validation probabilities before evaluating their live recall.
        dump(OUT/'conditional_thresholds.json',conditional)
        for et_key,choice in choices.items():
            label='grid_'+choice['key'];row,stats=measured(label,q,gt,env_for(tau,choice['et_tau']),VAL)
            ids=mat(VAL/(label+'_100_idx_uint32.bin'),'<u4');d=mat(VAL/(label+'_100_dists_float.bin'),'<f4')
            pred=first_stops(a,choice['et_tau'],full_stats['expansions'])
            verify_path(stats,ids,d,full_ids,full_d,pred)
            nhits=hits(ids,gtids);assert abs(nhits/100-row['recall_pct'])<1e-6
            row.update(choice);row.update(validation_hits=nhits,validation_recall_loss_pp=(basehits-nhits)/100,
                validation_conditional_unsafe_stops=int(pred['unsafe'].sum()),
                validation_changed_baseline_queries=int(np.any(ids!=baseids,axis=1).sum()))
            grid.append(row);dump(OUT/'validation_grid.json',grid)
            print('GRID',choice['key'],'Recall',row['recall_pct'],'Reads',row['ssd_reads_per_query'],
                  'Exp',row['expansions_per_query'],'ET unsafe relative to pruning-only',pred['unsafe'].sum(),flush=True)
        verification[prune_key]=dict(disabled_et_matches_pruning_only=True,
            native_et_probability_max_error=float(abs(p-python).max()),
            safe_labels_monotonic_on_pruned_paths=True,live_first_stop_and_safe_endpoint_verified=True)
    selected=[];budgets=[]
    for budget in RECALL_BUDGETS:
        required=basehits-int(np.floor(budget*100+1e-9))
        feasible=[r for r in grid if r['validation_hits']>=required]
        if not feasible:
            budgets.append(dict(recall_loss_budget_pp=budget,required_hits=required,selected_key=None));continue
        best=min(feasible,key=lambda r:(r['ssd_reads_per_query'],r['expansions_per_query'],-r['validation_hits'],
                                      r['nominal_prune_fnr_pct'],r['conditional_et_query_target_pct']))
        budgets.append(dict(recall_loss_budget_pp=budget,required_hits=required,selected_key=best['key']))
        existing=next((r for r in selected if r['key']==best['key']),None)
        if existing is None:
            existing={k:best[k] for k in ['key','prune_key','prune_tau','nominal_prune_fnr_pct','et_tau',
                'conditional_et_query_target_pct','validation_hits','validation_recall_loss_pp',
                'recall_pct','ssd_reads_per_query','expansions_per_query']}
            existing['selected_for_recall_budgets_pp']=[];selected.append(existing)
        existing['selected_for_recall_budgets_pp'].append(budget)
    assert selected
    dump(OUT/'selected_joint_configs.json',selected);dump(OUT/'selection_budgets.json',budgets)
    with (OUT/'validation_grid.csv').open('w',newline='') as f:
        fields=list(dict.fromkeys(k for r in grid for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(grid)
    dump(OUT/'validation_verification.json',verification)
    dump(OUT/'calibration_provenance.json',dict(
        models='frozen All17 ET + frozen Top3 expansion pruning',model_sha256=model_hashes(),
        original_pruning_thresholds='five previous validation FNR targets .1%,.25%,.5%,1%,2%',
        conditional_et_calibration='native full pruning-only validation trajectory; retain that pruning-only endpoint',
        et_conditional_query_targets_pct=[risk*100 for risk in RISKS],
        joint_selection='min actual validation SSD reads, then expansions, under actual validation Recall loss budget',
        recall_loss_budgets_pp=RECALL_BUDGETS,validation_baseline=base,validation_baseline_hits=basehits,
        test_used_for_thresholds_or_selection=False,no_model_retraining=True,
        label_limitation='conditional ET safety preserves pruning-only endpoint; it does not undo pruning losses',
        risk_metrics_not_added=True,binary_sha256=hashlib.sha256(BIN.read_bytes()).hexdigest()))
    dump(OUT/'CALIBRATION_COMPLETE.json',dict(complete=True,validation_grid=40,selected_joint_points=len(selected),
        thresholds_and_selection_locked_before_test=True))
    print('JOINT VALIDATION COMPLETE',selected,flush=True)

if __name__=='__main__':main()
