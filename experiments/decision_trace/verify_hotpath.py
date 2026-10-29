"""Compare compiled OMP query-loop instructions after relocation normalization."""
from pathlib import Path
import subprocess,re,json,hashlib
R=Path(__file__).resolve().parents[2];O=R/'experiments/decision_trace_acceptance'
def extract(exe,output):
 symbols=subprocess.check_output(['nm','-S','-C',str(exe)],text=True).splitlines()
 line=next(x for x in symbols if 'int search_disk_index<float, unsigned int' in x and '[clone ._omp_fn.1]' in x and ', true>' not in x)
 address,size=[int(x,16) for x in line.split()[:2]]
 raw=subprocess.check_output(['objdump','-d','--no-show-raw-insn',f'--start-address={address}',f'--stop-address={address+size}',str(exe)],text=True)
 tlsline=next(x for x in subprocess.check_output(['readelf','-lW',str(exe)],text=True).splitlines() if x.strip().startswith('TLS '))
 fields=tlsline.split();size_tls=int(fields[5],16);alignment=int(fields[-1],16);size_tls=(size_tls+alignment-1)//alignment*alignment
 tls={}
 for row in subprocess.check_output(['readelf','-sW',str(exe)],text=True).splitlines():
  fields=row.split()
  if len(fields)>=8 and fields[3]=='TLS':tls[(int(fields[1],16)-size_tls)%(1<<64)]=fields[-1]
 normalized=[]
 for line in raw.splitlines():
  m=re.match(r'\s*([0-9a-f]+):\s+(.*)',line)
  if not m:continue
  ins=m[2].split('#')[0].strip()
  ins=re.sub(r'-?0x[0-9a-f]+\(%rip\)','RIP(%rip)',ins)
  ins=re.sub(r'(?<![\w])([0-9a-f]+) <([^>]+)>',lambda m:('LOCAL+'+hex(int(m[1],16)-address)) if address<=int(m[1],16)<address+size else '<'+m[2]+'>',ins)
  ins=re.sub(r'%fs:0x([0-9a-f]+)',lambda m:'%fs:<'+tls.get(int(m[1],16),m[0])+'>',ins)
  normalized.append(ins)
 (O/output).write_text('\n'.join(normalized)+'\n')
 return dict(address=address,bytes=size,instructions=len(normalized),normalized_sha256=hashlib.sha256('\n'.join(normalized).encode()).hexdigest()),normalized
a,x=extract(O/'control','control_hotloop.asm');b,y=extract(R/'DiskANN/build/apps/search_disk_index','disabled_hotloop.asm')
result=dict(control=a,disabled=b,instructions_equal=x==y,normalization='Ignore code relocation addresses and RIP relative displacement only; retain immediates, stack offsets, call symbols, and relative local branch offsets; map ELF TLS displacements to their exact symbol names')
(O/'hotpath_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if x!=y:
 import difflib
 print('\n'.join(list(difflib.unified_diff(x,y))[:80]))
