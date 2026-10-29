from pathlib import Path
import os,json,subprocess,fcntl,time
D=Path(__file__).resolve().parent;ROOT=D.parents[1]
with (D/'launch.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=D/'BACKGROUND_STATE.json'
 if state.exists():
  old=json.loads(state.read_text());pid=old.get('pid')
  try:
   args=Path(f'/proc/{pid}/cmdline').read_bytes()
   if str(D/'run.py').encode() in args:print(json.dumps(old));raise SystemExit
  except OSError:pass
 if (D/'COMPLETE.json').exists():print('Already complete');raise SystemExit
 with (D/'pipeline.log').open('ab',buffering=0) as log:
  p=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(D/'run.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 data=dict(pid=p.pid,state='launched',started_at=time.strftime('%Y-%m-%d %H:%M:%S'),log=str(D/'pipeline.log'),note='Check STATUS.json, COMPLETE.json and FAILED.json; launch state alone is not live status.')
 state.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
