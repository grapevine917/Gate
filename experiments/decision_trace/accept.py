"""End-to-end native trace equivalence and randomized paired off-path benchmark."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import json,subprocess,time,hashlib,csv,random,sys
import numpy as np
from convert import convert
R=Path(__file__).resolve().parents[2];O=R/'experiments/decision_trace_acceptance';D=R/'experiments/e1_similarity/data';B=R/'DiskANN/build/apps/search_disk_index';CONTROL=O/'control';RUN=O/'runs';RUN.mkdir(exist_ok=True)
ENV={k:v for k,v in os.environ.items() if not k.startswith('DISKANN_')};ENV.update(DISKANN_M3_MODEL=str(R/'experiments/sift1m/online_state_mlp/model.bin'),DISKANN_M3_TAU='0.024352464824914932')
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def run(name,exe=B,extra=(),split='test',expect=0,model=True):
 out=RUN/name;env=ENV.copy();env['DISKANN_QUERY_STATS']=str(out)+'.csv'
 if not model:env.pop('DISKANN_M3_MODEL')
 cmd=[str(exe),'--data_type','float','--dist_fn','l2','--index_path_prefix',str(R/'experiments/sift1m/index'),'--query_file',str(D/(split+'.fbin')),'--gt_file',str(D/(split+'.gt')),'--result_path',str(out),'--recall_at','10','--search_list','100','--beamwidth','4','--num_threads','4','--num_nodes_to_cache','10000',*extra]
 if '--search_list' in extra:
  pos=cmd.index('--search_list');del cmd[pos:pos+2]
 dump(Path(str(out)+'.command.json'),dict(argv=cmd,env={k:v for k,v in env.items() if k.startswith('DISKANN_')},utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
 with open(str(out)+'.log','w') as f:p=subprocess.run(cmd,env=env,cwd=R,stdout=f,stderr=subprocess.STDOUT)
 if expect==0:assert p.returncode==0,(name,Path(str(out)+'.log').read_text()[-2000:])
 else:assert p.returncode!=0,name
 print(name,p.returncode,flush=True)
 if expect:return None
 rows=list(csv.DictReader(open(str(out)+'.csv')))
 latency=np.array([float(r['latency_us']) for r in rows]);summary=dict(mean_us=float(latency.mean()),p95_us=float(np.percentile(latency,95)))
 return out,rows,summary
KEYS=['query_row','expansions','ssd_reads','gate_evaluated','gate_skipped','et_evaluated','et_stopped']
def identical(a,b):
 filesa=sorted(a[0].parent.glob(a[0].name+'*bin'));filesb=sorted(b[0].parent.glob(b[0].name+'*bin'))
 fa={p.name[len(a[0].name):]:p for p in filesa};fb={p.name[len(b[0].name):]:p for p in filesb}
 assert fa.keys()==fb.keys() and fa
 for k in fa:assert fa[k].read_bytes()==fb[k].read_bytes(),k
 assert [[r[k] for k in KEYS] for r in a[1]]==[[r[k] for k in KEYS] for r in b[1]]
def main():
 for split in ['history','test']:
  ids=np.load(D/(split+'_query_ids.npy'));np.savetxt(O/(split+'_ids.txt'),ids,fmt='%d')
 control=run('functional_control',CONTROL)
 default=run('functional_default');identical(control,default)
 disabled_args=['--experiment_mode','decision_trace','--decision_trace_enabled','false','--decision_trace_prefix',str(O/'must_not_exist/log'),'--decision_trace_query_ids',str(O/'missing_ids')]
 disabled=run('functional_disabled',extra=disabled_args);identical(control,disabled);assert not (O/'must_not_exist').exists()
 for split in ['history','test']:
  args=['--experiment_mode','decision_trace','--decision_trace_prefix',str(O/split),'--decision_trace_query_ids',str(O/(split+'_ids.txt'))]
  enabled=run('functional_enabled_'+split,extra=args,split=split)
  base=control if split=='test' else run('functional_history_control',CONTROL,split=split)
  identical(base,enabled)
  convert(O/(split+'.manifest.json'),O/(split+'_npz'))
  trace=np.load(O/(split+'_npz/L100.npz'));rows=enabled[1]
  assert len(trace['query_id'])==sum(int(r['gate_evaluated']) for r in rows)
  assert int(trace['model_decision'].sum())==sum(int(r['gate_skipped']) for r in rows)
  assert np.all(trace['model_decision']==(trace['model_score']<float(ENV['DISKANN_M3_TAU'])))
  np.testing.assert_array_equal(np.load(O/(split+'_npz/queries.npz'))['query_vectors'],np.load(D/'query_vectors.npy')[np.load(D/(split+'_query_ids.npy'))])
  run('reject_overwrite_'+split,extra=args,split=split,expect=1)
 run('reject_bad_mode',extra=['--experiment_mode','bogus'],expect=1)
 run('reject_no_model',extra=['--experiment_mode','decision_trace'],expect=1,model=False)
 run('reject_missing_ids',extra=['--experiment_mode','decision_trace','--decision_trace_query_ids',str(O/'missing')],expect=1)
 # Warm filesystem/process dependencies before measurements; compile and analysis have ended.
 run('warm_control',CONTROL);run('warm_default');run('warm_disabled',extra=disabled_args)
 rounds=[];rng=random.Random(20261019)
 for i in range(9):
  kinds=['control','default','disabled'];rng.shuffle(kinds);row={}
  for kind in kinds:
   value=run(f'perf_{i}_{kind}',CONTROL if kind=='control' else B,disabled_args if kind=='disabled' else ())
   identical(control,value);row[kind]=value[2]
  rounds.append(row)
 ratios={};rngn=np.random.default_rng(20261019);take=rngn.integers(0,9,(10000,9))
 for mode in ['default','disabled']:
  ratios[mode]={}
  for metric in ['mean_us','p95_us']:
   v=np.array([r[mode][metric]/r['control'][metric] for r in rounds]);ci=np.percentile(np.median(v[take],axis=1),[2.5,97.5]);median=float(np.median(v))
   ratios[mode][metric]=dict(median_ratio=median,bootstrap_ci95=ci.tolist(),within_5pct=bool(.95<=median<=1.05))
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 dump(O/'verification.json',dict(functional_equivalence=True,ids_distances_expansions_reads_gate_counts_exact=True,disabled_ignores_invalid_trace_paths=True,trace_fields_checked=True,history_test_disjoint=True,negative_tests_passed=True,rounds=rounds,performance=ratios,core_source_sha256=sha(R/'DiskANN/src/pq_flash_index.cpp'),core_library_sha256=sha(R/'DiskANN/build/src/libdiskann.a')))
 print(json.dumps(ratios,indent=2),flush=True)
if __name__=='__main__':main()
