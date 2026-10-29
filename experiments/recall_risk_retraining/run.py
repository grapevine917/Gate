from common import *
for stage,done in [('collect_labels.py','LABELS_COMPLETE.json'),('train.py','TRAIN_COMPLETE.json'),('evaluate.py','ONLINE_COMPLETE.json'),('offline.py','OFFLINE_COMPLETE.json'),('report.py','COMPLETE.json')]:
    if (D/done).exists():continue
    print('STAGE',stage,flush=True);subprocess.run([str(ROOT/'.venv/bin/python'),str(D/stage)],check=True)
