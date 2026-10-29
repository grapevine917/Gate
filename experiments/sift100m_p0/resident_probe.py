"""Verify resident loader on both failed q-v batches, without accepting timings."""
from pathlib import Path
import json,subprocess,time,csv,hashlib,datetime
R=Path(__file__).resolve().parent
OUT=R/'resident_probe';OUT.mkdir(exist_ok=True)
reports=[]
for tag in ['1ea6a49a7963','267a09862f3a','1ea6a49a7963']:
 folder=OUT/f'pass{len(reports)}';folder.mkdir(exist_ok=True)
 cmd=json.loads((R/f'logs/validation_r0_{tag}.command.json').read_text())['argv'];cmd[0]=str(R/'native/search_resident');cmd[-2]=str(folder)
 samples=[]
 with (folder/'search.log').open('w') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
  while p.poll() is None:
   try:
    v={l.split(':')[0]:int(l.split()[1]) for l in Path(f'/proc/{p.pid}/status').read_text().splitlines() if l.startswith(('VmRSS:','VmSwap:','VmHWM:'))}
    samples.append(v)
   except FileNotFoundError:pass
   time.sleep(.2)
 (folder/'memory_samples.json').write_text(json.dumps(samples)+'\n')
 assert p.returncode==0,(tag,p.returncode)
 metas=list(folder.glob('*.meta.json'));assert len(metas)==8
 for path in metas:
  m=json.loads(path.read_text());assert m['swap_before_kib']==m['swap_after_kib']==0
  assert m['reader_bytes']==m['linux_read_bytes']>0
  prefix=path.name.removesuffix('.meta.json')
  old=R/'memory_fix_probe'/f'{prefix}.ids'
  if old.exists():
   assert old.read_bytes()==(folder/f'{prefix}.ids').read_bytes()
   with old.with_suffix('.csv').open() as a,(folder/f'{prefix}.csv').open() as b:
    aa=list(csv.DictReader(a));bb=list(csv.DictReader(b));assert len(aa)==len(bb)==1000
    for x,y in zip(aa,bb):assert all(x[k]==y[k] for k in x if k!='latency_us')
 assert max(v.get('VmSwap',0) for v in samples)==0
 reports.append(dict(tag=tag,configs=8,peak_rss_kib=max(v.get('VmRSS',0) for v in samples),peak_swap_kib=0))
 print(json.dumps(reports[-1]),flush=True)
info=dict(passed=True,reports=reports,search_sha256=hashlib.sha256((R/'native/search_resident').read_bytes()).hexdigest(),at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(OUT/'VERIFIED.json').write_text(json.dumps(info,indent=2)+'\n')
