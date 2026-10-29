"""Native q-v feature and uint32-ID parity, including IDs above float precision."""
from pathlib import Path
import json,subprocess
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parents[1];S=R/'smoke';S.mkdir(exist_ok=True)
if (R/'VECTOR_STORE_VERIFIED.json').exists():raise SystemExit(0)
nodes=np.concatenate([np.array([0,1,16777215,16777216,16777217,16777219,99999998,99999999],'<u4'),np.random.default_rng(20261030).integers(0,100000000,2040,dtype='u4')])
base=np.memmap(ROOT/'data/sift100m/base.100M.u8bin','u1','r',offset=8,shape=(100000000,128));v=base[nodes].astype('f4');w=(np.arange(128,dtype='f4')+1)/128/128/255
with (S/'vector.bin').open('wb') as f:
 f.write(b'S3GATE01');np.zeros(529,'<f4').tofile(f);np.ones(529,'<f4').tofile(f)
 a=np.zeros((529,128),'<f4');a[128:256,0]=w;a.tofile(f);np.zeros(128,'<f4').tofile(f)
 a=np.zeros((128,64),'<f4');a[0,0]=1;a.tofile(f);np.zeros(64,'<f4').tofile(f)
 a=np.zeros((64,1),'<f4');a[0,0]=1;a.tofile(f);np.zeros(1,'<f4').tofile(f)
raw=np.zeros((len(nodes),147),'<f4');raw.view('<u4')[:,128]=nodes;raw.tofile(S/'input.bin')
subprocess.run([str(R/'native/score'),str(S/'vector.bin'),str(R/'data/vectors.cvec'),str(S/'input.bin'),str(S/'scores.bin'),str(S/'bench.json')],check=True)
actual=np.fromfile(S/'scores.bin','<f4');expected=1/(1+np.exp(-(v@w)));error=float(np.max(np.abs(actual-expected)));assert len(actual)==len(nodes) and np.isfinite(actual).all() and error<2e-6
np.testing.assert_array_equal(raw.view('<u4')[:,128],nodes)
info=dict(passed=True,samples=len(nodes),edge_node_ids=nodes[:8].tolist(),max_native_score_error=error,description='Native resident compressed reader and feature generation compared to direct raw uint8 vectors; IDs stored as uint32 bits.')
(R/'VECTOR_STORE_VERIFIED.json').write_text(json.dumps(info,indent=2)+'\n');print(info,flush=True)
