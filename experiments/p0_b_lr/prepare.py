from pathlib import Path
import json,hashlib,subprocess
R=Path(__file__).resolve().parent;B=R.parent/'p0_b';A=R.parent/'p0_a_lr';M=R.parent/'main_comparison';S=R.parent/'sift1m'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
(R/'search.cpp').write_text((B/'search.cpp').read_text());c=json.loads((B/'build.command.json').read_text());c=[str(R/'search.cpp') if x==str(B/'search.cpp') else str(R/'search') if x==str(B/'search') else x for x in c];dump(R/'build.command.json',c)
with (R/'build.log').open('w') as f:subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,check=True)
cal=json.loads((A/'LR_CALIBRATION.json').read_text());taus=[x['tau'] for x in cal['points']];p=json.loads((B/'PROTOCOL.json').read_text());pairs=p['joint_threshold_index_pairs'];configs=json.loads((B/'configs.json').read_text())
for c in configs:
 if c['family']=='Ours':c['tau']=taus[c['preset']]
 if c['family']=='ET_Ours':c['tau']=taus[pairs[c['preset']][0]]
dump(R/'configs.json',configs)
p.update(experiment='P0-B LR replacement; original MLP run preserved',gate_model=str(M/'models/lr.bin'),gate='Same frozen P0-C mixed-L17D LR as P0-A LR rerun',gate_thresholds=taus,gate_threshold_risks=[x['risk'] for x in cal['points']],calibration=str(A/'LR_CALIBRATION.json'),limits=['Previously observed queries; exploratory','LR uses P0-C mixed-L weights, original P0-A/B MLP was L100-trained; rerun is not a matched-training MLP/LR ablation','ET+LR trajectory shift','Equal finite64 budget not global optimum; joint8 predeclared threshold pairs','Descriptive test frontier vs separately locked validation deployment choices; report actual Recall','Repeated finite timing still noisy'])
dump(R/'PROTOCOL.json',p)
inputs=[R/'search',R/'search.cpp',R/'PROTOCOL.json',R/'configs.json',A/'LR_CALIBRATION.json',M/'models/lr.bin',M/'models/et.bin',M/'native/libdiskann.a',S/'index_disk.index',S/'index_pq_pivots.bin',S/'index_pq_compressed.bin']+[M/f'data/{split}.{ext}' for split in ['validation','test'] for ext in ['fbin','gt']]
dump(R/'INPUT_LOCK.json',{str(x):sha(x) for x in inputs});print('P0-B LR GRID LOCKED 5x64',taus,flush=True)
