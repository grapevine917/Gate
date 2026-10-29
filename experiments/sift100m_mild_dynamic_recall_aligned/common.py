"""Shared protocol and strictly verified native measurements; no training."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import csv,hashlib,json,random,subprocess,time,shutil
import numpy as np
D=Path(__file__).resolve().parent;A=D;W=D.parent/'sift100m_p0';ROOT=D.parents[1]
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def save(p,rows):
    if not rows:return
    with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
P=json.loads((D/'PROTOCOL.json').read_text()) if (D/'PROTOCOL.json').exists() else dict(seed=20261024)
for p in [D/'logs',D/'runs/validation',D/'runs/test']:p.mkdir(parents=True,exist_ok=True)
exec(compile((D/'measurement.py').read_text(),str(D/'measurement.py'),'exec'),globals())
def aggregate(runs):
    out=[]
    for key in sorted({r['key'] for r in runs}):
        rs=[r for r in runs if r['key']==key];r=rs[0].copy();r.pop('repeat');r['repeats']=len(rs)
        for k in ['total_hits','physical_bytes','reads','expansions','skips','extra_skips','eligible','model_calls','ids_sha256']:assert len({a[k] for a in rs})==1,(key,k)
        for k in ['p95_ms','p50_ms','p99_ms','qps','mean_ms','io_ms']:r[k]=float(np.median([a[k] for a in rs]))
        r['p95_min_ms']=min(a['p95_ms'] for a in rs);r['p95_max_ms']=max(a['p95_ms'] for a in rs);out.append(r)
    return out
def choose_alignment(rows,validation_bykey,nominal_hits=9960):
    """Keep each static/dynamic pair's early threshold identical; no interpolation.

    Candidate membership uses the supplied split's hit counts. Timing rank always
    uses validation cost, including when rows are the frozen descriptive test grid.
    """
    pools={}
    for family in ['DiskANN','Heuristic']:
        pools[family]=[dict(keys=[r['key']],lo=r['total_hits'],hi=r['total_hits'],cost=validation_bykey[r['key']]['p95_ms']) for r in rows if r['family']==family and (family=='DiskANN' or r['skips']>0)]
    for family in ['MLP','LR']:
        ss=[r for r in rows if r['family']==family and r['kind'] in [2,3] and r['skips']>0];ds=[r for r in rows if r['family']==family and r['kind'] in [4,5] and r['multiplier']>1 and r['extra_skips']>0]
        pools[family]=[]
        for s in ss:
            for d in ds:
                if s['base_scale']==d['base_scale']:
                    assert s['tau']==d['tau']
                    pools[family].append(dict(keys=[s['key'],d['key']],lo=min(s['total_hits'],d['total_hits']),hi=max(s['total_hits'],d['total_hits']),cost=validation_bykey[s['key']]['p95_ms']+validation_bykey[d['key']]['p95_ms']))
    if any(not v for v in pools.values()):return dict(available=False,reason='No active static/dynamic pairs for one family')
    exact=set.intersection(*[{o['lo'] for o in options if o['lo']==o['hi']} for options in pools.values()])
    if exact:
        h=min(exact,key=lambda h:(abs(h-nominal_hits),-h));options={f:min([o for o in os if o['lo']==h and o['hi']==h],key=lambda o:(o['cost'],o['keys'])) for f,os in pools.items()};lo=hi=h
    else:
        # Minimum empirical spread is reported explicitly as approximate, never exact.
        candidates=sorted({o['lo'] for os in pools.values() for o in os});best=None
        for lo0 in candidates:
            selected={}
            for f,os in pools.items():
                valid=[o for o in os if o['lo']>=lo0]
                if not valid:break
                selected[f]=min(valid,key=lambda o:(o['hi'],o['cost'],o['keys']))
            if len(selected)!=4:continue
            hi0=max(o['hi'] for o in selected.values());score=(hi0-lo0,abs((hi0+lo0)/2-nominal_hits),-(hi0+lo0)/2,sum(o['cost'] for o in selected.values()))
            if best is None or score<best[0]:best=(score,lo0,hi0,selected)
        assert best is not None;_,lo,hi,options=best
    keys=options['DiskANN']['keys']+options['Heuristic']['keys']+[options['MLP']['keys'][0],options['LR']['keys'][0],options['MLP']['keys'][1],options['LR']['keys'][1]]
    return dict(available=True,mode='exact' if lo==hi else 'approximate',hits_min=lo,hits_max=hi,recall_min=lo/100,recall_max=hi/100,spread_pp=(hi-lo)/100,nominal_recall=nominal_hits/100,keys=keys,static_dynamic_base_tau_matched=True,cost_selection='validation P95 only; test times never used for choosing matching configs')
