"""Frozen validation choices, isolated profiling and exact decision-replay ablation."""
from pathlib import Path
import json, subprocess, os, time, hashlib

D=Path(__file__).resolve().parent
R=D.parent/'sift100m_p0'
configs={c['key']:c for c in json.loads((R/'configs.json').read_text())}
selected=[s for s in json.loads((R/'SELECTION_LOCK.json').read_text()) if s['study']=='p0_b_lr' and s['target'] in [99.5,99.7] and s['objective']=='physical_bytes' and s['family'] in ['DiskANN','Heuristic','LR']]
keys=list(dict.fromkeys(s['key'] for s in selected))+['DiskANN_L500_p0']
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
manifest=dict(selected=selected,additional_control='DiskANN_L500_p0: same L as all four pruning configurations',repeats=3,queries=1000,threads=4,modes=['profile: clocks plus record decision tape','full: regular gate, no per-candidate clocks or tape recording','replay: exact recorded decisions, no features or model'],purpose='diagnostic only; do not replace frozen formal results',input_hashes={str(p):sha(p) for p in [R/'native/search',R/'models/lr.bin',R/'data/test.u8bin',R/'data/test.gt',D/'native/search']})
(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
env=os.environ.copy()
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[name]='1'
out=D/'runs';out.mkdir(exist_ok=True)
def complete(key,rep,mode):
    prefix=out/f'{key}_r{rep}_{mode}'
    try:
        return (json.loads(prefix.with_suffix('.meta.json').read_text())['queries']==1000
                and prefix.with_suffix('.ids').stat().st_size==80000
                and len(prefix.with_suffix('.csv').read_text().splitlines())==1001)
    except (OSError,ValueError,KeyError):return False
for rep in range(3):
    order=keys[rep:]+keys[:rep]
    for key in order:
        c=configs[key]
        cp=D/f'{key}.configs'
        cp.write_text(f'{key} {c["kind"]} {c["L"]} {c["tau"]:.17g} {c["et_tau"]:.17g} {c["rank"]} {c["exp"]}\n')
        expected=['profile','full']+([] if c['kind']==0 else ['replay'])
        if all(complete(key,rep,mode) for mode in expected):continue
        cmd=[str(D/'native/search'),str(R/'index/ann'),str(R/'data/test.u8bin'),str(R/'data/test.gt'),str(R/'models'),str(R/'data/vectors.cvec'),str(cp),str(out),str(rep)]
        state=dict(key=key,repeat=rep,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),argv=cmd)
        (D/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
        print(time.strftime('%H:%M:%S'),'RUN',rep,key,flush=True)
        with (D/f'{key}_r{rep}.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
print('ALL RUNS COMPLETE',flush=True)
