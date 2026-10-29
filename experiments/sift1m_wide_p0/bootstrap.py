"""Build, collect widened trajectories and train; keep progress resumable."""
from pathlib import Path
import json,subprocess,time
D=Path(__file__).resolve().parent;ROOT=D.parents[1];PY=ROOT/'.venv/bin/python'
def run(cmd,log,phase):
    (D/'STATUS.json').write_text(json.dumps(dict(phase=phase,complete=False,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2)+'\n')
    print(phase,flush=True)
    with (D/log).open('a') as f:subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,check=True)
if not (D/'BUILD_COMPLETE.json').exists():run([PY,D/'prepare_build.py'],'build.log','build')
for split,ls,cap,prefix in [('train','L.txt',64,'mixed'),('validation','calibration_L.txt',100000,'calibration')]:
    marker=D/f'COLLECT_{split.upper()}_COMPLETE.json'
    if not marker.exists():
        cmd=[D/'native/collect',D.parent/'sift1m/index',D/'data'/f'{split}.fbin',D/'data'/f'{split}.gt',D/'models/dummy.bin',D/ls,D/'data'/prefix,str(cap)]
        run(cmd,f'collect_{split}.log',f'collect_{split}');marker.write_text(json.dumps(dict(complete=True,cap=cap,split=split))+'\n')
if not (D/'TRAIN_COMPLETE.json').exists():run([PY,D/'train.py'],'train.log','training')
run([D/'native/verify',D/'models'],'native_verify.log','native_export_verification')
errors=json.loads((D/'native_verify.log').read_text().splitlines()[-1]);assert max(errors[k] for k in ['LR','MLP','ET'])<2e-5
(D/'NATIVE_VERIFIED.json').write_text(json.dumps(dict(passed=True,errors=errors),indent=2)+'\n')
print('BOOTSTRAP COMPLETE',flush=True)
