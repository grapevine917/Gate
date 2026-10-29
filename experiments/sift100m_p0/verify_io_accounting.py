"""Recheck physical I/O and disabled-gate parity before formal evaluation."""
from pathlib import Path
import json,subprocess,datetime,hashlib
import numpy as np
R=Path(__file__).resolve().parent;S=R/'preflight';out=S/'io_fix';out.mkdir(exist_ok=True)
marker=R/'IO_ACCOUNTING_VERIFIED.json'
if marker.exists():marker.unlink()
metas={}
for key in ['baseline','disabled']:
 with (out/f'{key}.log').open('w') as f:
  subprocess.run([str(R/'native/search'),str(S/'ann'),str(S/'query.u8bin'),str(S/'query.gt'),str(S/'models'),'unused',str(S/f'{key}.configs'),str(out),'0'],stdout=f,stderr=subprocess.STDOUT,check=True)
 meta=json.loads((out/f'{key}_r0.meta.json').read_text());a=np.genfromtxt(out/f'{key}_r0.csv',delimiter=',',names=True)
 assert meta['reader_bytes']==int(a['reads'].sum())*4096>0
 assert meta['reader_bytes']==meta['linux_read_bytes'], f"Kernel IO accounting unavailable or inconsistent: {meta}"
 assert meta['swap_before_kib']==meta['swap_after_kib']==0
 metas[key]=meta
assert (out/'baseline_r0.ids').read_bytes()==(out/'disabled_r0.ids').read_bytes()
a=np.genfromtxt(out/'baseline_r0.csv',delimiter=',',names=True);b=np.genfromtxt(out/'disabled_r0.csv',delimiter=',',names=True)
for key in ['hits','reads','expansions']:np.testing.assert_array_equal(a[key],b[key])
info=dict(passed=True,disabled_gate_ids_counts_equal=True,kernel_reader_bytes_equal=True,metas=metas,kernel=__import__('os').uname().release,search_sha256=hashlib.sha256((R/'native/search').read_bytes()).hexdigest(),verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
marker.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
