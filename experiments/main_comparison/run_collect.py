from pathlib import Path
import subprocess
R=Path(__file__).resolve().parent;S=R.parent/'sift1m'
args=[str(R/'native/collect'),str(S/'index'),str(R/'data/train.fbin'),str(R/'data/train.gt'),str(S/'online_state_mlp/model.bin'),str(R/'L.txt'),str(R/'data/mixed'),"32"]
with (R/'collect.log').open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
print("MIXED COLLECTION COMPLETE")
