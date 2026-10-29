#!/usr/bin/env python3
from common import *
import subprocess,sys,traceback
def main():
 log('E2 START/RESUME')
 if not (R/'PREPARED.json').exists():subprocess.run([sys.executable,str(R/'prepare.py')],check=True)
 if not (R/'native/intervene').exists():subprocess.run([sys.executable,str(R/'build_native.py')],check=True)
 p=R/'intervention';tau=json.loads((E1/'SELECTION_LOCK.json').read_text())['selected']['MLP']
 cmd=[str(R/'native/intervene'),str(S/'index'),str(E1/'data/test.fbin'),str(E1/'data/test.gt'),str(S/'online_state_mlp/model.bin'),str(tau),str(R/'interventions.bin'),str(p)]
 inputs={str(path):sha(path) for path in [R/'native/intervene',R/'interventions.bin',S/'online_state_mlp/model.bin',E1/'data/test.fbin',E1/'data/test.gt']}
 if (R/'INTERVENTION_DONE.json').exists():assert json.loads((R/'INTERVENTION_DONE.json').read_text())['inputs']==inputs,'Changed inputs require a new run'
 else:
  dump(R/'intervention.command.json',{'argv':cmd,'env':{k:os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  log('START '+str(json.loads((R/'PREPARED.json').read_text())['requests'])+' single-decision replacement searches')
  with (R/'intervention.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  assert sha(R/'intervention.baseline.ids')==sha(E1/'runs_main/test_MLP_r0_time.ids'),'Fresh E2 baseline must equal E1'
  assert sha(R/'intervention.cache.bin')==sha(E1/'runs_main/test_MLP_trace.cache.bin'),'Node cache must match E1'
  dump(R/'INTERVENTION_DONE.json',{'inputs':inputs,'csv_sha256':sha(R/'intervention.csv'),'baseline_ids_equal_e1':True,'cache_equal_e1':True})
  log('ALL REPLACEMENT SEARCHES COMPLETE; baseline/cache parity passed')
 subprocess.run([sys.executable,str(R/'analyze.py')],check=True)
 files=[R/'common.py',R/'prepare.py',R/'run.py',R/'build_native.py',R/'override.h',R/'intervene.cpp',R/'analyze.py',R/'plot_figures.py',R/'native/pq_flash_index.cpp',R/'native/intervene',S/'training_data_l100/train/pre_io_state.npy',S/'training_data_l100/train/query_ids.npy',S/'online_state_mlp/model.bin',E1/'SELECTION_LOCK.json',E1/'MLP_history_trace.npz',E1/'MLP_test_trace.npz']
 dump(R/'PROVENANCE.json',{'files':{str(f):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in files},'base_e1_provenance':str(E1/'PROVENANCE.json'),'normalization_train_only':True,'test_used_to_fit_cutoffs_or_predictors':False,'python':sys.version})
 artifacts=[f for f in R.rglob('*') if f.is_file() and f.suffix in ['.py','.h','.cpp','.npz','.csv','.png','.svg','.pdf','.json','.md','.bin','.ids'] and f.name not in ['artifact_manifest.json','COMPLETE.json']]
 dump(R/'artifact_manifest.json',{str(f.relative_to(R)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(artifacts)})
 dump(R/'COMPLETE.json',{'complete':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':{f.name:sha(f) for f in [R/'REPORT.md',R/'analysis.json',R/'intervention_summary.json',R/'verification.json',R/'artifact_manifest.json']}})
 log('E2 COMPLETE')
if __name__=='__main__':
 try:main()
 except Exception:log('FAILED\n'+traceback.format_exc());raise
