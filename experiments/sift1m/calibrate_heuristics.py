#!/usr/bin/env python3
"""Exhaustive validation-only calibration of rank and rank/progress rules."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from train_state_baselines import dump,operating
E=Path(__file__).resolve().parent;DATA=E/'training_data_l100';OUT=E/'top3_heuristic'
TARGETS=[.001,.0025,.005,.01,.02]

def suffix(a):
    s=a[::-1,::-1].cumsum(0).cumsum(1)[::-1,::-1]
    return np.pad(s,((0,1),(0,1)))

def main():
    OUT.mkdir(exist_ok=True);(OUT/'CALIBRATED.json').unlink(missing_ok=True)
    manifest=json.loads((DATA/'manifest.json').read_text());names=manifest['pre_io_state_columns']
    ri=names.index('candidate_queue_rank');ei=names.index('completed_expansions')
    x=np.load(DATA/'validation/pre_io_state.npy',mmap_mode='r');y=np.load(DATA/'validation/y.npy')
    rank=x[:,ri].astype(int);completed=x[:,ei].astype(int)
    np.testing.assert_array_equal(x[:,ri],rank);np.testing.assert_array_equal(x[:,ei],completed)
    assert rank.min()>=0 and rank.max()<100 and completed.min()>=0
    shape=(100,int(completed.max())+1);total=np.zeros(shape,dtype='int64');positive=np.zeros(shape,dtype='int64')
    np.add.at(total,(rank,completed),1);np.add.at(positive,(rank[y==1],completed[y==1]),1)
    skip_counts=suffix(total);bad_counts=suffix(positive)
    grids={'Rank':[],'Rank+Progress':[]};npositive=int((y==1).sum())
    for r in range(-1,100):
        for e in range(-1,shape[1]):
            n=int(skip_counts[r+1,e+1]);bad=int(bad_counts[r+1,e+1])
            row=dict(r=r,e=e,validation_skipped=n,validation_false_skips=bad,
                     validation_skip_rate=n/len(y),validation_retain_fnr=bad/npositive)
            grids['Rank+Progress'].append(row)
            if e==-1:grids['Rank'].append(row)
    selected=[]
    for method,grid in grids.items():
        filename='rank' if method=='Rank' else 'rank_progress'
        with (OUT/(filename+'_validation_grid.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(grid[0]));w.writeheader();w.writerows(grid)
        for risk in TARGETS:
            budget=int(np.floor(risk*npositive))
            feasible=[g for g in grid if g['validation_false_skips']<=budget]
            best=max(feasible,key=lambda g:(g['validation_skipped'],-g['validation_false_skips'],g['r'],g['e']))
            actual=(rank>best['r'])&(completed>best['e'])
            assert actual.sum()==best['validation_skipped']
            assert (actual&(y==1)).sum()==best['validation_false_skips']
            selected.append(dict(method=method,key=filename+f'_fnr_{risk*100:g}pct',target_fnr_pct=risk*100,
                validation_retain_budget=budget,**best))
    # Lock choices before loading or reporting test labels/states.
    dump(OUT/'selected_parameters.json',selected)
    summary=[]
    for split in ('validation','test'):
        x=np.load(DATA/split/'pre_io_state.npy',mmap_mode='r');y=np.load(DATA/split/'y.npy')
        meta=np.load(DATA/split/'metadata.npy',mmap_mode='r')
        for c in selected:
            skip=(x[:,ri]>c['r'])&(x[:,ei]>c['e'])
            m=operating(y,(~skip).astype('float64'),.5,meta['query_id'])
            summary.append(dict(method=c['method'],key=c['key'],split=split,target_fnr_pct=c['target_fnr_pct'],
                r=c['r'],e=c['e'],skip_rate=m['skip_rate'],retain_fnr=m['retain_false_skip_rate'],
                skipped=m['skipped'],false_skips=m['retain_mislabeled_skipped'],
                affected_queries=m['queries_with_retain_false_skip']))
    dump(OUT/'offline_summary.json',summary)
    with (OUT/'offline_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    dump(OUT/'calibration_provenance.json',dict(
        rules={'Rank':'rank > r','Rank+Progress':'rank > r AND completed_expansions > e'},
        rank_zero_based=True,decision='before candidate is marked expanded; completed counts only exact-scored expansions',
        parameter_grid=dict(r=[-1,99],e=[-1,shape[1]-1],integer_step=1),
        objective='maximize validation sample-weighted SkipRate under conservative retain FNR budget',
        tie_break='minimum false skips, then larger r, then larger e',
        selection_split='validation',selection_online=False,
        learned_comparison='same offline validation FNR targets; actual online trajectories rerun',
        no_test_parameter_selection=True,validation_queries=1000,test_queries=1000,
        manifest_sha256=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),
        limitation='The two requested rules use rank and completed_expansions; visited_count is not used.'))
    dump(OUT/'CALIBRATED.json',dict(complete=True,rules=2,selected_points=len(selected),validation_only=True))
    for c in selected:print(c,flush=True)

if __name__=='__main__':main()
