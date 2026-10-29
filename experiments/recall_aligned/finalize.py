from pathlib import Path
import csv,json,hashlib,struct
import numpy as np
ROOT=Path(__file__).resolve().parent;SOURCE=ROOT.parent/'io_predictability';OLD=ROOT.parent/'policy_comparison';MEMO=ROOT.parent/'query_skip_cache'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def token(x):return struct.pack('<f',float(np.float32(x))).hex()
results=json.loads((ROOT/'results.json').read_text());checks={};curves=[];aligned_rows=[];anchor_rows=[]
for ds,workloads in results.items():
 ids=[set(np.load(SOURCE/ds/s/'query_ids.npy').tolist()) for s in ['train','validation','test']];assert not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]);checks[ds]={}
 for work,v in workloads.items():
  folder=ROOT/ds/work;lock=json.loads((folder/'SELECTION_LOCK.json').read_text());assert lock['test_used_for_selection'] is False and lock['selected']==v['selected'];assert lock['validation_points']==v['validation']
  testdir=folder/'test/all_locked_points';assert (folder/'SELECTION_LOCK.json').stat().st_mtime_ns<=(testdir/'command.json').stat().st_mtime_ns
  # No accidental C++ model-map collisions when refined thresholds are rounded for lookup keys.
  native_keys={}
  for c in lock['configs']:
   if c['family'] in ['LR','Reuse']:
    nk=(c['model'],format(c['tau'],'.6f'))
    if nk in native_keys:assert native_keys[nk]==c['tau']
    native_keys[nk]=c['tau']
   if c['family']=='Reuse':
    signature=json.loads((Path(c['bank']).parent/'DONE.json').read_text());assert signature['tau']==c['tau'] and signature['model_sha256']==sha(c['model']) and signature['queries_sha256']==sha(SOURCE/ds/'train/query.fbin')
   if c['family']=='Mark':
    base=np.unpackbits(np.frombuffer((ROOT.parent/'node_mark_comparison'/ds/'Mark_m1_r1.bin').read_bytes()[12:],np.uint8),bitorder='little');bits=np.unpackbits(np.frombuffer(Path(c['mark']).read_bytes()[12:],np.uint8),bitorder='little');assert np.all(bits<=base)
    if c['parameter']==1:assert Path(c['mark']).read_bytes()==(ROOT.parent/'node_mark_comparison'/ds/'Mark_m1_r1.bin').read_bytes()
    if c['parameter']==0:assert not bits.any()
  # Unchanged fixed policies must reproduce prior results, not only rounded aggregate Recall.
  oldtau=json.loads((SOURCE/ds/'state_lr/thresholds.json').read_text())['unsafe_fpr_0.005']['retain_threshold'];oldprefix=OLD/ds/work/'test';newprefix=testdir/'run'
  mapping=[('DiskANN','DiskANN'),(v['full_mark']['key'],'Mark_naive'),('LR_'+token(oldtau),'LR'),('Reuse_'+token(oldtau),'LR_Reuse')]
  for new,old in mapping:
   for ext in ['ids','dists']:assert sha(f'{newprefix}_{new}_r0.{ext}')==sha(f'{oldprefix}_{old}_r0.{ext}')
  # Zero enabled marks and zero model threshold are genuine no-pruning controls.
  for c in lock['configs']:
   if (c['family']=='Mark' and c['parameter']==0) or (c['family']=='LR' and c['tau']==0):
    for ext in ['ids','dists']:assert sha(f'{newprefix}_{c["key"]}_r0.{ext}')==sha(f'{newprefix}_DiskANN_r0.{ext}')
  checks[ds][work]={'validation_only_selection':True,'lock_before_test':True,'independent_tau_banks':True,'native_threshold_map_no_collisions':True,'marks_subset_of_original':True,'full_mark_and_existing_LR_outputs_match':True,'no_pruning_controls_exact':True}
  for key,a in v['test'].items():curves.append({'dataset':ds,'workload':work,'family':a['family'],'key':key,'parameter':a['parameter'],**{k:a[k] for k in ['recall_pct','reads','calls','reused','latency_us','p95_us','qps','latency_min_us','latency_max_us']}})
  for entry in v['aligned']:
   for family,a in entry['methods'].items():
    if a is None:continue
    lo=v['test'][a['endpoint_low']];hi=v['test'][a['endpoint_high']];w=a['low_weight'];assert 0<=w<=1 and abs(w*lo['recall_pct']+(1-w)*hi['recall_pct']-entry['target_recall_pct'])<1e-7
    for metric in ['reads','calls','reused','latency_us']:assert abs(a[metric]-(w*lo[metric]+(1-w)*hi[metric]))<1e-7
    aligned_rows.append({'dataset':ds,'workload':work,'budget_pp':entry['budget_pp'],'family':family,**a})
  anchor=v['full_mark_anchor'];full=v['full_mark']
  for family in ['Mark','LR','Reuse']:
   a=anchor['methods'][family]
   if family=='Mark':a={'recall_pct':full['recall_pct'],'interpolated':False,'endpoint_low':full['key'],'endpoint_high':full['key'],'low_weight':1.,**{k:full[k] for k in ['reads','calls','reused','latency_us']}}
   if a is not None:anchor_rows.append({'dataset':ds,'workload':work,'family':family,**a})
with open(ROOT/'curve_points.csv','w') as f:
 writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
with open(ROOT/'aligned.csv','w') as f:
 writer=csv.DictWriter(f,fieldnames=list(aligned_rows[0]));writer.writeheader();writer.writerows(aligned_rows)
(ROOT/'verification.json').write_text(json.dumps(checks,indent=2));(ROOT/'full_mark_anchor.json').write_text(json.dumps(anchor_rows,indent=2))
labels={'Mark':'全局一次标记','LR':'LR pruning','Reuse':'LR+Reuse'}
lines=['# 对齐Recall后的三方案比较','','所有代价点来自本轮实测；三轮中位数。下表在同一Recall上用实测点作线性插值，标注为估计，不是三个上线单配置恰好达到相同Recall。验证集选参后的真实测试值另见REPORT.md。主优化目标是最少请求，表中延迟为同一I/O最优组合的估计，不独立优化延迟。','','## 与原始完整全局一次标记对齐','','Mark列始终使用f=1原表的真实结果；LR与Reuse对齐它的测试Recall，必要时插值。自然查询的该锚点损失超过2pp，仅作完整方案对照，不作为满足业务预算的结果。','','|数据集|workload|共同Recall%|方法|请求/q|平均延迟ms|请求相对Mark变化|插值估计|','|---|---|---:|---|---:|---:|---:|---|']
for ds in ['sift','deep']:
 for work in ['near','natural']:
  rr=[a for a in anchor_rows if a['dataset']==ds and a['workload']==work];mark=next(a for a in rr if a['family']=='Mark')
  for a in rr:lines.append(f"|{ds}|{work}|{a['recall_pct']:.3f}|{labels[a['family']]}|{a['reads']:.3f}|{a['latency_us']/1000:.4f}|{a['reads']/mark['reads']-1:+.2%}|{'是' if a['interpolated'] else '否'}|")
lines+=['','## 2pp预算内：共同可达的同Recall对照','','若完整标记损失不到2pp，目标取其更高的Recall。自然query则只启用部分原标记节点（固定哈希比例f），并非声称原完整标记已满足2pp。','','|数据集|workload|共同Recall%|方法|请求/q估计|平均延迟ms估计|相对DiskANN请求减少|','|---|---|---:|---|---:|---:|---:|']
for a in aligned_rows:
 if a['budget_pp']!=2:continue
 b=results[a['dataset']][a['workload']]['test']['DiskANN'];lines.append(f"|{a['dataset']}|{a['workload']}|{a['recall_pct']:.3f}|{labels[a['family']]}|{a['reads']:.3f}|{a['latency_us']/1000:.4f}|{1-a['reads']/b['reads']:.2%}|")
lines+=['','## 验证集对齐后的真实测试Recall','','参数只由验证集选择；下表如实保留测试Recall偏差，不能把它称为完全对齐。','','|数据集|workload|方法|验证共同目标%|验证实际Recall%|测试实际Recall%|参数f或tau|','|---|---|---|---:|---:|---:|---:|']
for ds,ws in results.items():
 for work,v in ws.items():
  for family,c in v['selected']['2.0'].items():lines.append(f"|{ds}|{work}|{labels[family]}|{c['target_recall_pct']:.3f}|{c['validation_recall_pct']:.3f}|{v['test'][c['key']]['recall_pct']:.3f}|{c['parameter']:.7g}|")
lines+=['','0.5/1/2pp全部目标、原始点与曲线估计见REPORT.md、curve_points.csv、aligned.csv。近似查询是历史query的扰动，不等同自然独立查询。每个LR阈值独立建立1000query的历史skip缓存，模型权重没有重训；Reuse固定rho<=0.01。已核验原完整Mark和旧LR/Reuse输出与前轮一致，空表/零阈值与DiskANN一致；三轮结果确定性由运行脚本核验。插值是描述性的平均成本曲线，不是假装已经测过的混合流量或测试选出的上线配置。']
(ROOT/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
complete=json.loads((ROOT/'COMPLETE.json').read_text());complete['independent_verification']=True
files=[ROOT/n for n in ['SUMMARY.md','verification.json','full_mark_anchor.json','curve_points.csv','aligned.csv','finalize.py','run.sh']]
for ds in results:
 for folder in (ROOT/ds/'banks').iterdir():files.extend([folder/'DONE.json',folder/'bank.bin'])
for p in files:complete['sha256'][str(p)]=sha(p)
(ROOT/'COMPLETE.json').write_text(json.dumps(complete,indent=2));print((ROOT/'SUMMARY.md').read_text())
