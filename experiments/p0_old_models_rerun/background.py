"""Detached host supervisor; never overlaps benchmark processes."""
from pathlib import Path
import os,sys,json,time,subprocess,signal,fcntl
D=Path(__file__).resolve().parent;ROOT=D.parents[1]
def dump(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def active():
 out=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:a=(p/'cmdline').read_bytes().decode(errors='replace').split('\0');a=[s for s in a if s]
  except OSError:continue
  if not a:continue
  is_python=Path(a[0]).name.startswith('python')
  script=Path(a[1]) if is_python and len(a)>1 else None
  if script and str(script).endswith('experiments/p0_old_models_rerun/run.py'):out.append(dict(pid=int(p.name),role='pipeline'))
  elif script and str(script).endswith('experiments/p0_old_models_rerun/worker.py'):out.append(dict(pid=int(p.name),role='worker',args=a[2:]))
  elif str(D) in a[0] and a[0].endswith('/native/search'):out.append(dict(pid=int(p.name),role='native_search'))
 return out
def progress():
 try:s=json.loads((D/'STATUS.json').read_text())
 except (OSError,ValueError):s={}
 return dict(phase=s,completed_P0A=[ds for ds in ['sift1m','deep1m','sift100m'] if (D/ds/'P0A_COMPLETE.json').exists()],completed_P0B=[ds for ds in ['sift1m','deep1m','sift100m'] if (D/ds/'P0B_COMPLETE.json').exists()])
if '--launch' in sys.argv:
 p=D/'BACKGROUND_STATE.json'
 if p.exists():
  old=json.loads(p.read_text());pid=old.get('pid')
  if pid:
   try:
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
    if str(Path(__file__).resolve()).encode() in cmd and b'--daemon' in cmd:print(json.dumps(old,ensure_ascii=False));sys.exit(0)
   except OSError:pass
 with (D/'background.log').open('ab',buffering=0) as log:
  child=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--daemon'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 dump(p,dict(pid=child.pid,state='starting',log=str(D/'background.log'),launched_at=time.strftime('%Y-%m-%d %H:%M:%S')));print(json.dumps(dict(supervisor_pid=child.pid,state='launched',state_file=str(p)),ensure_ascii=False))
elif '--daemon' in sys.argv:
 signal.signal(signal.SIGHUP,signal.SIG_IGN);lock=(D/'background.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:sys.exit(0)
 child=None;started=time.strftime('%Y-%m-%d %H:%M:%S')
 while True:
  if (D/'COMPLETE.json').exists():
   dump(D/'BACKGROUND_STATE.json',dict(pid=os.getpid(),state='complete',finished_at=time.strftime('%Y-%m-%d %H:%M:%S'),**progress()));break
  processes=active()
  if child is not None and child.poll() is not None:
   if child.returncode!=0:
    dump(D/'BACKGROUND_STATE.json',dict(pid=os.getpid(),state='failed',returncode=child.returncode,log=str(D/'pipeline_background.log'),updated_at=time.strftime('%Y-%m-%d %H:%M:%S'),**progress()));break
   child=None
  if not processes:
   print(time.strftime('%Y-%m-%d %H:%M:%S'),'RESUMING PIPELINE',flush=True)
   log=(D/'pipeline_background.log').open('ab',buffering=0)
   child=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(D/'run.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True);log.close();processes=active()
  dump(D/'BACKGROUND_STATE.json',dict(pid=os.getpid(),state='running',started_at=started,updated_at=time.strftime('%Y-%m-%d %H:%M:%S'),log=str(D/'pipeline_background.log'),active_processes=processes,**progress()))
  time.sleep(20)
else:raise SystemExit('Use --launch or --daemon')
