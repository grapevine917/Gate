import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import csv,hashlib,json,random,subprocess,time,shutil
import numpy as np
D=Path('/home/lxyue/projects/learned-gate/experiments/late_multiplier_1p2/sift100m');A=Path('/home/lxyue/projects/learned-gate/experiments/sift100m_mild_dynamic_recall_aligned');W=Path('/home/lxyue/projects/learned-gate/experiments/sift100m_p0');ROOT=Path('/home/lxyue/projects/learned-gate')
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
P=json.loads((D/'PROTOCOL.json').read_text())
for p in [D/'logs',D/'runs/validation',D/'runs/test']:p.mkdir(parents=True,exist_ok=True)
exec(compile((D/'measurement.py').read_text(),str(D/'measurement.py'),'exec'),globals())
