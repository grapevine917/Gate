"""Durable sequential continuation; waits for the already-running index job."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,subprocess,sys,time,traceback
R=Path(__file__).resolve().parent;PY=R.parents[1]/'.venv/bin/python';os.chdir(R.parents[1])
lock=(R/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(phase,**kw):
 info=dict(phase=phase,complete=False,pid=os.getpid(),updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
 tmp=R/'STATUS.tmp';tmp.write_text(json.dumps(info,indent=2)+'\n');tmp.replace(R/'STATUS.json');print(info,flush=True)
def run(script,log,phase):
 status(phase)
 with (R/log).open('a') as f:subprocess.run([str(PY),str(R/script)],stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 for marker in ['DATA_COMPLETE','NATIVE_COMPLETE','COMPRESSION_COMPLETE','EXACT_AUDIT_COMPLETE','PREFLIGHT_COMPLETE']:
  info=json.loads((R/f'{marker}.json').read_text());assert info.get('passed',info.get('complete',False)),(marker,info)
 status('waiting_for_index',current_log='index_build.log')
 while not (R/'INDEX_COMPLETE.json').exists():
  # The index job predates this runner; its failure status must remain visible.
  info=json.loads((R/'STATUS.json').read_text())
  if info['phase']=='index_build_failed':raise RuntimeError(f'Index failed: {info}')
  time.sleep(20)
 run('vector_store_smoke.py','vector_store_smoke.log','native_vector_store_verification')
 run('pilot_and_lock.py','pilot.log','validation_L_pilot_and_input_lock')
 run('collect_data.py','collection.log','trajectory_collection')
 if not (R/'TRAIN_COMPLETE.json').exists():run('train.py','train.log','model_training')
 pilot=json.loads((R/'L_PILOT_LOCK.json').read_text())
 io_check=json.loads((R/'IO_ACCOUNTING_VERIFIED.json').read_text()) if (R/'IO_ACCOUNTING_VERIFIED.json').exists() else {}
 if io_check.get('passed') and io_check.get('kernel')==os.uname().release and io_check.get('search_sha256')==hashlib.sha256((R/'native/search').read_bytes()).hexdigest():
  if not (R/'ONLINE_COMPLETE.json').exists():run('evaluate.py','evaluation.log','P0_A_B_C_evaluation')
  if not (R/'COMPLETE.json').exists():run('analyze.py','analysis.log','reports_and_figures')
  status('complete',complete_result=True,report=str(R/'REPORT.md'))
  info=json.loads((R/'STATUS.json').read_text());info['complete']=True;(R/'STATUS.json').write_text(json.dumps(info,indent=2)+'\n')
 else:
  status('blocked_evaluation_io_accounting',training_complete=True,error='Kernel physical IO accounting unavailable: /proc/self/io absent and getrusage ru_inblock remains zero. Recall pilot, collection and training completed; formal evaluation requires a compatible kernel. No physical IO result is claimed.')

except BaseException as e:
 alert=dict(failed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=str(e),traceback=traceback.format_exc(),previous_status=json.loads((R/'STATUS.json').read_text()),evaluation_log_tail=(R/'evaluation.log').read_text()[-8000:] if (R/'evaluation.log').exists() else '')
 (R/'ALERT.json').write_text(json.dumps(alert,indent=2,ensure_ascii=False)+'\n')
 print('EXPERIMENT ALERT: '+json.dumps(alert,ensure_ascii=False),flush=True)
 status('failed',error=str(e),traceback=alert['traceback']);raise
