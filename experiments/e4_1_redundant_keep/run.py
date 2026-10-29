import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,struct,subprocess,shlex,hashlib,datetime
import numpy as np
R=Path(__file__).resolve().parent; ROOT=R.parents[1]; E1=R.parent/'e1_similarity'; E2=R.parent/'e2_state_reuse'; D=ROOT/'DiskANN'
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
t=dict(np.load(E1/'MLP_test_trace.npz'));h=dict(np.load(E1/'MLP_history_trace.npz'))
keep=np.flatnonzero(t['model_invoked'] & ~t['model_decision'].astype(bool));splits=json.loads((E1/'query_splits.json').read_text());qrow={int(q):i for i,q in enumerate(splits['test'])};tau=json.loads((E1/'SELECTION_LOCK.json').read_text())['selected']['MLP']
protocol=dict(stage='E4-1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),population='ALL actual MLP Keep cache-miss positions in frozen E1 test trace; cached nodes are protected and not model predictions',queries=1000,keep_positions=len(keep),tau=tau,L=100,k=10,beam=4,threads=4,cache_nodes=10000,intervention='one forced Skip at exact ordinal/node/byte-identical 17D state, fresh original MLP thereafter',delta_recall='(after_hits-baseline_hits)/10',delta_io='baseline_reads-after_reads',A='delta_hits>=0 and delta_io>0',B='delta_hits>=0 and delta_io<=0',C='delta_hits<0 and delta_io>0',D='delta_hits<0 and delta_io<=0',selection='exhaustive; no outcome selection',difficulty='post-hoc baseline Recall; not an online feature',frequency='history model-invoked node occurrences only',confidence='1000 query-cluster bootstrap replicates, seed 20261019',limits=['Previously observed test set','Single interventions cannot be summed as simultaneous pruning benefit','SSD requests are not physical device reads','Cache-hit candidates are excluded because model not invoked'])
if (R/'PROTOCOL.json').exists():assert json.loads((R/'PROTOCOL.json').read_text())['keep_positions']==len(keep)
else:dump(R/'PROTOCOL.json',protocol)
np.savez_compressed(R/'keep_positions.npz',record=keep,**{k:v[keep] for k,v in t.items()})
requests=R/'interventions.bin'
with requests.open('wb') as f:
 f.write(struct.pack('<I',len(keep)))
 for i,j in enumerate(keep):f.write(struct.pack('<5I18f',i,qrow[int(t['query_id'][j])],int(t['search_step'][j]),int(t['node_id'][j]),1,float(t['model_score'][j]),*t['state_features'][j]))
# Sham Keep controls sampled without looking at intervention outcomes.
rng=np.random.default_rng(20261019);sham=rng.choice(keep,min(1000,len(keep)),replace=False)
with (R/'sham.bin').open('wb') as f:
 f.write(struct.pack('<I',len(sham)))
 for i,j in enumerate(sham):f.write(struct.pack('<5I18f',i,qrow[int(t['query_id'][j])],int(t['search_step'][j]),int(t['node_id'][j]),0,float(t['model_score'][j]),*t['state_features'][j]))
exe=R/'intervene'
if not exe.exists():
 flags=(D/'build/src/CMakeFiles/diskann.dir/flags.make').read_text();opts=[]
 for key in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']:opts+=shlex.split(next(l.split('=',1)[1] for l in flags.splitlines() if l.startswith(key+' =')))
 link=shlex.split((D/'build/apps/CMakeFiles/collect_io_labels.dir/link.txt').read_text());libs=link[link.index('-L/opt/intel/oneapi/mkl/latest/lib/intel64'):];libs=[str(E2/'native/libdiskann.a') if s=='../src/libdiskann.a' else s for s in libs]
 cmd=['g++','-I'+str(E2/'native'),*opts,str(R/'intervene.cpp'),'-o',str(exe),*libs,'-latomic'];dump(R/'build.command.json',cmd)
 with (R/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
for name in ['sham','intervention']:
 inp=R/('sham.bin' if name=='sham' else 'interventions.bin');prefix=R/name
 inputs={str(p):sha(p) for p in [exe,inp,E1/'data/test.fbin',E1/'data/test.gt',R/'intervene.cpp',E2/'native/libdiskann.a',R.parent/'sift1m/online_state_mlp/model.bin']}
 done=R/(name+'.done.json')
 if done.exists():assert json.loads(done.read_text())['inputs']==inputs;continue
 cmd=[str(exe),str(R.parent/'sift1m/index'),str(E1/'data/test.fbin'),str(E1/'data/test.gt'),str(R.parent/'sift1m/online_state_mlp/model.bin'),str(tau),str(inp),str(prefix)];dump(R/(name+'.command.json'),dict(argv=cmd,env={k:os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 print('START',name,len(keep) if name=='intervention' else len(sham),flush=True)
 with (R/(name+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 assert sha(R/(name+'.baseline.ids'))==sha(E1/'runs_main/test_MLP_r0_time.ids')
 assert sha(R/(name+'.cache.bin'))==sha(E1/'runs_main/test_MLP_trace.cache.bin')
 dump(done,dict(inputs=inputs,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline_ids_equal_e1=True,cache_equal_e1=True))
subprocess.run([str(ROOT/'.venv/bin/python'),str(R/'analyze.py')],check=True)
