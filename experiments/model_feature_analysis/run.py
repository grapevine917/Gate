"""Queue offline feature analysis after all expanded P0-B timing has ended."""
from pathlib import Path
import json,time,os,fcntl,subprocess,sys,hashlib
D=Path(__file__).resolve().parent;ROOT=D.parents[1];P0=ROOT/'experiments/p0_old_models_expanded'
def dump(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');q.replace(p)
def main():
 if (D/'COMPLETE.json').exists():return
 lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  try:done=json.loads((P0/'COMPLETE.json').read_text()).get('complete',False)
  except (OSError,ValueError):done=False
  idle=False
  if done:
   with (P0/'run.lock').open('a') as benchlock:
    try:fcntl.flock(benchlock,fcntl.LOCK_EX|fcntl.LOCK_NB);idle=True
    except BlockingIOError:pass
  if done and idle:break
  try:phase=json.loads((P0/'STATUS.json').read_text())
  except (OSError,ValueError):phase={}
  dump(D/'STATUS.json',dict(phase='waiting_for_expanded_P0B',updated=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid(),p0_phase=phase,reason='Do not overlap CPU inference/analysis with validation selection or formal search timing.'))
  time.sleep(30)
 dump(D/'STATUS.json',dict(phase='starting_analysis',updated=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid()))
 print(time.strftime('%F %T'),'P0-B complete; starting offline analysis',flush=True)
 subprocess.run([sys.executable,str(D/'analyze.py')],cwd=ROOT,check=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  dump(D/'FAILED.json',dict(error=repr(e),at=time.strftime('%F %T')));raise
