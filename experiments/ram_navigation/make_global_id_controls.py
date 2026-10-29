"""Generate the logically identical full-original-edge view for CSR parity."""
from pathlib import Path
import json,os,subprocess,shutil
ROOT=Path(__file__).resolve().parent;LG=ROOT.parent.parent
s=json.loads((ROOT/'summary.json').read_text())
for ds in ['sift','deep']:
 for method in ['B2','B3']:
  out=ROOT/ds/f'{method}_global_id_control';data=LG/'experiments/io_predictability'/ds/'test'
  budget=s[ds]['selected_navigation_budgets'][method]['navigation_budget']
  env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',OMP_PROC_BIND='close',OMP_PLACES='cores',RAM_USE_ORIGINAL_EDGES='1')
  args=[str(LG/'DiskANN/build/apps/search_ram_navigation'),'run',str(LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index'),str(ROOT/ds/method),str(data/'query.fbin'),str(data/'gt.bin'),str(out),str(budget),'1']
  with open(str(out)+'.log','w') as f:subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
