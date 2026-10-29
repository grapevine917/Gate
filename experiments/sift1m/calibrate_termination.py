#!/usr/bin/env python3
"""Calibrate whole-query unsafe termination budgets on native validation scores."""
import csv,json
import numpy as np
from run_online_m3 import E,execute,mat
from train_state_baselines import dump
OUT=E/'learned_termination'
RISKS=[0,.001,.0025,.005,.01,.02,.05,.10]

def main():
    assert json.loads((OUT/'TRAINING_COMPLETE.json').read_text())['complete']
    ids=np.load(E/'training_data_l100/validation/query_ids.npy')
    raw=np.load(OUT/'data/validation/state.npy');y=np.load(OUT/'data/validation/y_safe_stop.npy')
    qids=np.load(OUT/'data/validation/sample_query_ids.npy')
    summary=[];verification={}
    for model in ('Top3','All17'):
        folder=OUT/model;trace=OUT/f'{model}_validation_native.f32'
        execute('validation_native_'+model,OUT/'validation_query.fbin',OUT/'validation_gt.bin',
            {'DISKANN_ET_MODEL':str(folder/'model.bin'),'DISKANN_ET_TAU':'1.1',
             'DISKANN_ET_TRACE':str(trace)},output_dir=OUT)
        for suffix,dtype in (('idx_uint32','<u4'),('dists_float','<f4')):
            np.testing.assert_array_equal(mat(OUT/f'validation_native_{model}_100_{suffix}.bin',dtype),
                                          mat(E/f'baseline_100_{suffix}.bin',dtype)[ids])
        a=np.fromfile(trace,dtype='<f4').reshape(-1,21)
        local=a[:,0].astype('uint32');original=ids[local]
        order=np.lexsort((a[:,10],original));a=a[order];local=local[order];original=original[order]
        np.testing.assert_array_equal(original,qids)
        np.testing.assert_array_equal(a[:,2:19],raw)
        np.testing.assert_array_equal(a[:,19],y)
        p=a[:,20];python=np.load(folder/'validation_p_safe_stop.npy')
        np.testing.assert_allclose(p,python,atol=2e-6,rtol=2e-5)
        np.save(folder/'validation_native_p_safe_stop.npy',p)
        unsafe_max=np.full(len(ids),-1,dtype='float32')
        np.maximum.at(unsafe_max,local[y==0],p[y==0])
        sorted_max=np.sort(unsafe_max)[::-1];thresholds={}
        for risk in RISKS:
            budget=int(np.floor(risk*len(ids)))
            # Native float32 scores and a float32-nextafter threshold prevent rounding across the tie.
            tau=float(np.nextafter(sorted_max[budget],np.float32(np.inf)))
            if tau<0:tau=0.0
            stop=(p>=tau);unsafe=unsafe_max>=tau
            assert unsafe.sum()<=budget
            stop_q=np.unique(local[stop]);first=[]
            for query in range(len(ids)):
                ix=np.flatnonzero((local==query)&stop)
                if len(ix):first.append(int(ix[0]))
            first=np.array(first,dtype=int)
            assert int((y[first]==0).sum())==int(unsafe.sum())
            key=f'unsafe_query_{risk*100:g}pct'
            c=dict(threshold=tau,selection_split='validation_native',target_unsafe_stop_query_rate=risk,
                query_budget=budget,validation_unsafe_stops=int(unsafe.sum()),
                validation_unsafe_query_rate=float(unsafe.mean()),validation_stopped_queries=len(stop_q),
                validation_stop_query_rate=len(stop_q)/len(ids),
                validation_mean_expansions_at_predicted_stop=float(a[first,9].mean()) if len(first) else None,
                decision='stop entire search iff P(safe-to-stop) >= threshold at beam boundary after >=K exact scores')
            thresholds[key]=c;summary.append(dict(model=model,key=key,**c))
        dump(folder/'thresholds.json',thresholds)
        verification[model]=dict(disabled_et_exact_baseline_results_validation=True,
            native_state_and_labels_match_collected_data=True,
            cpp_python_max_probability_error=float(abs(p-python).max()),
            calibration_uses_native_float32_scores=True,
            first_stop_unsafe_counts_match_max_unsafe_probability_rule=True)
        print('ET THRESHOLDS',model,thresholds,flush=True)
    dump(OUT/'calibration_summary.json',summary)
    dump(OUT/'calibration_verification.json',verification)
    dump(OUT/'CALIBRATION_COMPLETE.json',dict(complete=True,models=2,validation_only=True,
        query_level_risk=True,unsafe_query_targets_pct=[r*100 for r in RISKS]))

if __name__=='__main__':main()
