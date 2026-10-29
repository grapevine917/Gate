from common import *
def stage(ds,mode,done):
 if (D/ds/done).exists():return
 print('STAGE',ds,mode,flush=True);subprocess.run([str(ROOT/'.venv/bin/python'),str(D/'worker.py'),ds,mode],check=True)
for ds in CATALOG:stage(ds,'smoke','SMOKE_COMPLETE.json')
for ds in CATALOG:stage(ds,'p0a','P0A_COMPLETE.json')
subprocess.run([str(ROOT/'.venv/bin/python'),str(D/'report.py'),'p0a'],check=True)
for ds in CATALOG:
 stage(ds,'p0b','P0B_COMPLETE.json')
 subprocess.run([str(ROOT/'.venv/bin/python'),str(D/'report.py'),ds],check=True)
subprocess.run([str(ROOT/'.venv/bin/python'),str(D/'report.py'),'all'],check=True)
