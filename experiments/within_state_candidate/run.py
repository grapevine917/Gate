from common import *
import subprocess,sys
for marker,script in [('PREPARED.json','prepare.py'),('native/intervene','build.py')]:
 if not (R/marker).exists():subprocess.run([sys.executable,str(R/script)],check=True)
model=FG/'models/Full17_s42/model.bin';inputs={str(p):sha(p) for p in [R/'native/intervene',R/'native/libdiskann.a',R/'override.h',R/'intervene.cpp',R/'PROTOCOL.json',R/'interventions.bin',R/'sham.bin',model,E1/'data/test.fbin',E1/'data/test.gt']}
if (R/'INPUT_LOCK.json').exists():assert json.loads((R/'INPUT_LOCK.json').read_text())==inputs
else:dump(R/'INPUT_LOCK.json',inputs)
for name in ['sham','intervention']:
 done=R/(name+'.done.json')
 if done.exists():assert json.loads(done.read_text())['inputs']==inputs;continue
 req=R/('sham.bin' if name=='sham' else 'interventions.bin');prefix=R/name;cmd=['taskset','-c',','.join(map(str,CPUS)),str(R/'native/intervene'),str(S/'index'),str(E1/'data/test.fbin'),str(E1/'data/test.gt'),str(model),'0',str(req),str(prefix)]
 env=os.environ.copy();env.update(OMP_PROC_BIND='true',OMP_PLACES='cores');dump(R/(name+'.command.json'),dict(argv=cmd,CPU_affinity=CPUS,env={k:env[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_PROC_BIND','OMP_PLACES']},utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));log('START '+name)
 with (R/(name+'.log')).open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 assert sha(R/(name+'.baseline.ids'))==sha(FG/'audit/DiskANN.ids');assert sha(R/(name+'.cache.bin'))==sha(FG/'audit/DiskANN.cache.bin')
 dump(done,dict(inputs=inputs,baseline_ids_equal_DiskANN=True,cache_equal=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
if not (R/'logit_scores.npz').exists():subprocess.run([sys.executable,str(R/'score_logits.py')],check=True)
subprocess.run([sys.executable,str(R/'analyze.py')],check=True)
