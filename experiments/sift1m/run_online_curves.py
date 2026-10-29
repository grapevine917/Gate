#!/usr/bin/env python3
"""Joint randomized measurements for three recall/cost curves on held-out queries."""
import json,hashlib,csv,argparse
import numpy as np
from run_online_m3 import E,ROOT,BIN,execute
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--with-state',action='store_true')
args=parser.parse_args()
OUT=E/('online_curves_state' if args.with_state else 'online_curves');OUT.mkdir(exist_ok=True)
q=E/'online_m3/test_query.fbin';gt=E/'online_m3/test_gt.bin'
configs=[]
for l in [20,40,50,60,70,80,90,100,120,150]:
    configs.append(dict(method='DiskANN varying L',key=f'L{l}',L=l,parameter=l,env={}))
for budget in [20,40,60,70,80,85,90,95,100,105,110,120,140]:
    configs.append(dict(method='Early Termination',key=f'io{budget}',L=100,parameter=budget,
                        env={'DISKANN_EARLY_IO_LIMIT':str(budget)}))
thresholds=json.loads((E/'online_m3/thresholds.json').read_text())
for key,c in thresholds.items():
    configs.append(dict(method='Learned Expansion Pruning',key=key,L=100,
        parameter=c['target_retain_false_skip_rate']*100,threshold=c['threshold'],env={
        'DISKANN_M3_MODEL':str(E/'online_m3/model3.bin'),
        'DISKANN_M3_BASE':str(ROOT/'data/sift/sift_base.fbin'),
        'DISKANN_M3_TAU':str(c['threshold'])}))
if args.with_state:
    for key,c in json.loads((E/'online_state_mlp/thresholds.json').read_text()).items():
        configs.append(dict(method='State-only Expansion Pruning',key='state_'+key,L=100,
            parameter=c['target_retain_false_skip_rate']*100,threshold=c['threshold'],env={
            'DISKANN_M3_MODEL':str(E/'online_state_mlp/model.bin'),
            'DISKANN_M3_TAU':str(c['threshold'])}))
execute('joint_warmup',q,gt,{},output_dir=OUT)
rng=np.random.default_rng(42);runs=[]
for repeat in range(3):
    for index in rng.permutation(len(configs)):
        c=configs[index]; label=f'joint_r{repeat}_{c["key"]}'
        row=execute(label,q,gt,c['env'],search_l=c['L'],output_dir=OUT)
        row.update({k:v for k,v in c.items() if k!='env'});row['repeat']=repeat;runs.append(row)
        (OUT/'joint_runs.json').write_text(json.dumps(runs,indent=2))
        print(label,'Recall',row['recall_pct'],'Reads',row['ssd_reads_per_query'],'Latency',row['latency_us'],flush=True)
summary=[]
for c in configs:
    rows=[r for r in runs if r['key']==c['key']]
    assert len(rows)==3 and len({r['recall_pct'] for r in rows})==1
    result={k:v for k,v in c.items() if k!='env'}
    for k in ['recall_pct','ssd_reads_per_query','expansions_per_query','latency_us','qps']:
        result[k]=float(np.median([r[k] for r in rows]))
    result['latency_min_us']=min(r['latency_us'] for r in rows)
    result['latency_max_us']=max(r['latency_us'] for r in rows)
    summary.append(result)
(OUT/'joint_summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'curve_data.csv').open('w',newline='') as f:
    fields=list(dict.fromkeys(k for r in summary for k in r))
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary)
(OUT/'provenance.json').write_text(json.dumps(dict(queries='held-out test1000',query_ids=str(E/'training_data_l100/test/query_ids.npy'),
    K=10,threads=4,beam=4,node_cache=10000,repeats=3,randomized_joint_order_seed=42,
    early_termination='existing DiskANN io_limit; L=100; stop at beam boundary; <=3 reads possible overshoot',
    learned_pruning='M3 and state-only MLP' if args.with_state else 'M3',
    model_details='L=100, validation-selected thresholds; M3 uses all v resident in RAM; state-only never loads q/v vectors; includes inference latency',
    timing='three interleaved randomized passes, median warm-host timings',
    binary_sha256=hashlib.sha256(BIN.read_bytes()).hexdigest()),indent=2))
print('JOINT CURVES COMPLETE',flush=True)
