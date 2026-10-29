"""Verify exact replay and report paired query tails without selecting new policies."""
from pathlib import Path
import csv, json, hashlib
import numpy as np

D=Path(__file__).resolve().parent
R=D.parent/'sift100m_p0'
manifest=json.loads((D/'manifest.json').read_text())
keys=list(dict.fromkeys(s['key'] for s in manifest['selected']))+['DiskANN_L500_p0']
def read(p):return np.genfromtxt(p,delimiter=',',names=True)
def save(name,rows):
    with (D/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def scalar(x):return float(x)
def med(a,k):return np.median(np.stack([v[k] for v in a]),axis=0)
def corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
data={};summary=[];checks=[];cohorts=[];paired=[]
for key in keys:
    modes=['profile','full']+([] if key.startswith('DiskANN') else ['replay'])
    data[key]={}
    original=R/'runs/test'/f'{key}_r0'
    original_ids=original.with_suffix('.ids').read_bytes()
    old=read(original.with_suffix('.csv'))
    for mode in modes:
        rows=[];metas=[]
        for rep in range(3):
            prefix=D/'runs'/f'{key}_r{rep}_{mode}'
            a=read(prefix.with_suffix('.csv'));m=json.loads(prefix.with_suffix('.meta.json').read_text())
            assert len(a)==1000 and prefix.with_suffix('.ids').read_bytes()==original_ids,(key,mode,rep,'IDs')
            for k in ['hits','reads','expansions','skips','gate_evaluated']:
                np.testing.assert_array_equal(a[k],old[k],err_msg=f'{key}/{mode}/{rep}/{k}')
            assert m['reader_bytes']==m['linux_read_bytes']==int(a['reads'].sum())*4096
            assert m['swap_before_kib']==m['swap_after_kib']==0
            assert np.all(a['io_us']<=a['latency_us']+5)
            rows.append(a);metas.append(m)
        data[key][mode]=rows
        lat=med(rows,'latency_us')
        summary.append(dict(key=key,mode=mode,recall=float(old['hits'].sum()/100),p95_ms=float(np.median([np.percentile(a['latency_us'],95) for a in rows])/1000),qps=float(np.median([m['qps'] for m in metas])),mean_ms=float(np.mean(lat)/1000),reads=float(old['reads'].mean()),io_rounds=float(med(rows,'io_rounds').mean()),reads_per_io_round=float(old['reads'].sum()/med(rows,'io_rounds').sum()),rounds=float(med(rows,'rounds').mean()),empty_rounds=float(med(rows,'empty_rounds').mean()),queue_slots=float(med(rows,'queue_slots').mean()),skips=float(old['skips'].mean()),io_ms=float(med(rows,'io_us').mean()/1000),rank_us=float(med(rows,'rank_us').mean()),feature_us=float(med(rows,'feature_us').mean()),score_us=float(med(rows,'score_us').mean()),p95_min_ms=float(min(np.percentile(a['latency_us'],95) for a in rows)/1000),p95_max_ms=float(max(np.percentile(a['latency_us'],95) for a in rows)/1000)))
        checks.append(dict(key=key,mode=mode,repeats=3,queries=1000,ids_and_counts_equal_frozen=True,physical_io_equal=True,no_swap=True))
    full=data[key]['full'];profile=data[key]['profile']
    tail=np.argsort(med(full,'latency_us'))[-50:]
    for cohort,ix in [('all',np.arange(1000)),('own_slowest_5pct',tail)]:
        p={k:med(profile,k)[ix] for k in profile[0].dtype.names}
        lat=med(full,'latency_us')[ix]
        row=dict(key=key,cohort=cohort,queries=len(ix),full_mean_ms=float(lat.mean()/1000),io_mean_ms=float(med(full,'io_us')[ix].mean()/1000),reads=float(p['reads'].mean()),io_rounds=float(p['io_rounds'].mean()),reads_per_io_round=float(p['reads'].sum()/p['io_rounds'].sum()),rounds=float(p['rounds'].mean()),empty_rounds=float(p['empty_rounds'].mean()),skips=float(p['skips'].mean()),rank_us=float(p['rank_us'].mean()),feature_us=float(p['feature_us'].mean()),score_us=float(p['score_us'].mean()),profile_mean_ms=float(p['latency_us'].mean()/1000))
        if 'replay' in data[key]:
            replay=med(data[key]['replay'],'latency_us')[ix]
            full_residual=lat-med(full,'io_us')[ix]
            replay_residual=replay-med(data[key]['replay'],'io_us')[ix]
            row.update(replay_mean_ms=float(replay.mean()/1000),full_minus_replay_ms=float((lat-replay).mean()/1000),full_non_reader_ms=float(full_residual.mean()/1000),replay_non_reader_ms=float(replay_residual.mean()/1000),full_minus_replay_non_reader_ms=float((full_residual-replay_residual).mean()/1000))
        else:row.update(replay_mean_ms=None,full_minus_replay_ms=None,full_non_reader_ms=None,replay_non_reader_ms=None,full_minus_replay_non_reader_ms=None)
        cohorts.append(row)
save('summary.csv',summary);save('cohorts.csv',cohorts)
distribution=[]
for key in keys:
    a=data[key]['full'];lat=med(a,'latency_us');reads=med(a,'reads');rounds=med(a,'io_rounds')
    distribution.append(dict(key=key,reads_min=float(reads.min()),reads_p25=float(np.percentile(reads,25)),reads_p50=float(np.percentile(reads,50)),reads_p75=float(np.percentile(reads,75)),reads_p95=float(np.percentile(reads,95)),reads_max=float(reads.max()),latency_reads_correlation=corr(lat,reads),latency_io_rounds_correlation=corr(lat,rounds)))
save('query_distribution.csv',distribution)
for target in [99.5,99.7]:
    choices={s['family']:s['key'] for s in manifest['selected'] if s['target']==target}
    lr=choices['LR'];heur=choices['Heuristic'];baseline=choices['DiskANN']
    x=data[lr]['full'];y=data[heur]['full']
    tails={'LR_tail':np.argsort(med(x,'latency_us'))[-50:],'Heuristic_tail':np.argsort(med(y,'latency_us'))[-50:],'all':np.arange(1000)}
    for cohort,ix in tails.items():
        for family,key in [('LR',lr),('Heuristic',heur),('DiskANN',baseline),('same_L_DiskANN','DiskANN_L500_p0')]:
            a=data[key]['full'];p=data[key]['profile'];f=med(a,'latency_us')[ix]
            paired.append(dict(target=target,cohort=cohort,family=family,key=key,queries=len(ix),mean_ms=float(f.mean()/1000),io_ms=float(med(a,'io_us')[ix].mean()/1000),non_io_ms=float((f-med(a,'io_us')[ix]).mean()/1000),reads=float(med(a,'reads')[ix].mean()),io_rounds=float(med(a,'io_rounds')[ix].mean()),reads_per_round=float(med(a,'reads')[ix].sum()/med(a,'io_rounds')[ix].sum()),empty_rounds=float(med(a,'empty_rounds')[ix].mean()),rank_us=float(med(p,'rank_us')[ix].mean()),feature_us=float(med(p,'feature_us')[ix].mean()),score_us=float(med(p,'score_us')[ix].mean()),hits=float(med(a,'hits')[ix].mean())))
    qrows=[]
    for qi in range(1000):
        row=dict(query=qi,LR_tail=bool(qi in tails['LR_tail']),Heuristic_tail=bool(qi in tails['Heuristic_tail']))
        for family,key in [('LR',lr),('Heuristic',heur),('DiskANN',baseline)]:
            for k in ['latency_us','io_us','io_rounds','rounds','empty_rounds','reads','skips','hits','comparisons']:
                row[f'{family}_{k}']=float(med(data[key]['full'],k)[qi])
        row['LR_minus_Heuristic_us']=row['LR_latency_us']-row['Heuristic_latency_us']
        qrows.append(row)
    save(f'paired_queries_{target}.csv',qrows)
save('paired_cohorts.csv',paired)
# Same-L traces distinguish alternate routes from merely shorter baseline prefixes.
def traces(key):
    a=np.fromfile(D/'runs'/f'{key}_r0_profile.trace.u32',dtype='<u4').reshape(-1,3)
    assert np.all(a[:,0]<1000) and np.all(a[:,1]<100000000)
    return [(set(a[(a[:,0]==q)&(a[:,2]==0),1].tolist()),set(a[(a[:,0]==q)&(a[:,2]==1),1].tolist())) for q in range(1000)]
base_trace=traces('DiskANN_L500_p0');route_rows=[]
for key in keys:
    if key.startswith('DiskANN'):continue
    trace=traces(key);a=data[key]['full'];b=data['DiskANN_L500_p0']['full']
    for qi,((expanded,skipped),(base_expanded,_)) in enumerate(zip(trace,base_trace)):
        route_rows.append(dict(key=key,query=qi,expanded_nodes=len(expanded),skipped_nodes=len(skipped),expanded_outside_same_L_baseline=len(expanded-base_expanded),baseline_expansions_skipped=len(skipped&base_expanded),baseline_nodes_not_expanded=len(base_expanded-expanded),hits_delta=float(med(a,'hits')[qi]-med(b,'hits')[qi]),io_rounds_delta=float(med(a,'io_rounds')[qi]-med(b,'io_rounds')[qi])))
save('route_comparison.csv',route_rows)
(D/'verification.json').write_text(json.dumps(dict(passed=True,checks=checks,original_input_hashes_unchanged={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest['input_hashes'].items()}),indent=2)+'\n')
print('VERIFIED',len(checks),'configuration/mode combinations; tables written')
for r in summary:
    if r['mode'] in ['full','replay']:print(r)
