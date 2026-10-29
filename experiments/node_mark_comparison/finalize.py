"""Independent checks and concise review artifact; never reselect on test."""
from pathlib import Path
import json,hashlib,struct
import numpy as np
ROOT=Path(__file__).resolve().parent
results=json.loads((ROOT/'results.json').read_text());checks={}
for ds,v in results.items():
 counts=np.load(ROOT/ds/'training_counts.npz');n=counts['visits'];z=counts['low'];assert np.all(z<=n)
 for c in v['training']['rules']:
  path=Path(c['marks']);assert hashlib.sha256(path.read_bytes()).hexdigest()==c['marks_sha256']
  with open(path,'rb') as f:
   assert f.read(8)==b'NDMARK01' and struct.unpack('<I',f.read(4))[0]==1000000
   bits=np.unpackbits(np.frombuffer(f.read(),np.uint8),bitorder='little')
  np.testing.assert_array_equal(bits,(n>=c['min_support'])&(z>=c['low_rate']*n));assert int(bits.sum())==c['marked_nodes']
 lock=json.loads((ROOT/ds/'SELECTION_LOCK.json').read_text());assert lock['test_used_for_selection'] is False and lock['selected']==v['selected']
 assert hashlib.sha256((ROOT/ds/'validation.json').read_bytes()).hexdigest()==lock['validation_sha256']
 assert (ROOT/ds/'SELECTION_LOCK.json').stat().st_mtime<=min(p.stat().st_mtime for p in (ROOT/ds/'test').glob('*.command.json'))
 for r in v['selected']:
  base=v['validation']['DiskANN'];feasible=[a for a in v['validation'].values() if a['method']==r['method'] and base['recall_pct']-a['recall_pct']<=r['loss_budget_pp']+1e-9 and a[r['objective']]<base[r['objective']]]
  expected=min(feasible,key=lambda a:(a[r['objective']],-a['recall_pct'],a['key']))['key'] if feasible else 'DiskANN';assert expected==r['chosen_key']
 checks[ds]={'bitmaps_match_training_rule':True,'selection_matches_validation_only_rule':True,'selection_file_predates_test_runs':True,'native_controls':json.loads((ROOT/ds/'verification.json').read_text())}
lines=['# 非学习式标记与LR：结果摘要','','标记方案只统计历史事件并查冻结位图，不拟合学习模型。原始LR根据当前查询的17维状态评分。训练/验证/测试各1000查询、固定L100/beam4/K10/cache10000/4线程；测试重复3次，以下均为验证选中的配置。','','## 验证平均Recall损失预算0.1个百分点，按读取请求最少选参','','|数据集|方法|配置|测试Recall%|测试损失pp|净请求节省|平均延迟μs|延迟降低|受损查询/1000|','|---|---|---|---:|---:|---:|---:|---:|---:|']
for ds,v in results.items():
 b=v['test']['DiskANN']
 for method in ['LR','Mark']:
  r=next(r for r in v['selected'] if r['loss_budget_pp']==.1 and r['objective']=='reads' and r['method']==method);a=v['test'][r['chosen_key']]
  lines.append(f"|{ds}|{method}|{a['key']}|{a['recall_pct']:.3f}|{b['recall_pct']-a['recall_pct']:.3f}|{1-a['reads']/b['reads']:.2%}|{a['latency_us']:.2f}|{1-a['latency_us']/b['latency_us']:.2%}|{a['degraded_queries']}|")
lines+=['','## 验证平均Recall损失预算0.5个百分点，按读取请求最少选参','','|数据集|方法|测试Recall损失pp|净请求节省|平均延迟降低|','|---|---|---:|---:|---:|']
for ds,v in results.items():
 b=v['test']['DiskANN']
 for method in ['LR','Mark']:
  r=next(r for r in v['selected'] if r['loss_budget_pp']==.5 and r['objective']=='reads' and r['method']==method);a=v['test'][r['chosen_key']]
  lines.append(f"|{ds}|{method}|{b['recall_pct']-a['recall_pct']:.3f}|{1-a['reads']/b['reads']:.3%}|{1-a['latency_us']/b['latency_us']:.2%}|")
lines+=['','## 朴素一次低贡献标记规则','','这是预先固定的m=1,r=1配置诊断，不是测试挑选的配置。','','|数据集|标记节点数|测试Recall损失pp|净请求节省|延迟降低|','|---|---:|---:|---:|---:|']
for ds,v in results.items():
 b=v['test']['DiskANN'];a=v['test']['Mark_m1_r1'];lines.append(f"|{ds}|{a['marked_nodes']}|{b['recall_pct']-a['recall_pct']:.3f}|{1-a['reads']/b['reads']:.2%}|{1-a['latency_us']/b['latency_us']:.2%}|")
lines+=['','## 事后诊断：原始测试路径上的关键节点区分能力','','此处仅复用已有测试反事实标签解释固定标记，不参与选参，不代表动态路径的误剪次数。','','|数据集|一次低贡献标记覆盖miss比例|覆盖关键miss比例|','|---|---:|---:|']
diagnostic={}
for ds in results:
 a=np.genfromtxt(ROOT.parent/'io_opportunity'/f'{ds}_L100_candidates.csv',delimiter=',',names=True);a=a[a['cache_miss']!=0];critical=a['singleton_recall_ok']==0
 bits=np.unpackbits(np.frombuffer((ROOT/ds/'Mark_m1_r1.bin').read_bytes()[12:],np.uint8),bitorder='little');skip=bits[a['node'].astype(int)]!=0
 diagnostic[ds]={'miss_mark_fraction':float(skip.mean()),'critical_mark_fraction':float(skip[critical].mean()),'critical_count':int(critical.sum()),'critical_marked':int((skip&critical).sum()),'selection_use':False}
 lines.append(f"|{ds}|{skip.mean():.2%}|{skip[critical].mean():.2%}|")
(ROOT/'baseline_diagnostic.json').write_text(json.dumps(diagnostic,indent=2))
lines+=['','## 解读与边界','','- 低贡献定义是没有邻居进入候选队列、且自己不在训练查询最终top10中。它不使用GT，但也不是跨查询安全标签。','- 全局节点表存在明显查询依赖风险；要求更多重复观察会缩小可标记集合。每数据集只有1000训练查询，结论限定于这一历史覆盖规模，尚未检验更大历史库、查询分区或在线更新。','- 表内Recall预算只约束验证集；测试若超预算，不能宣称达标。平均Recall不下降也不保证逐查询安全。','- Mark在线位图125000字节（约122KiB），离线计数数组8MB；LR模型文件220字节。该内存对比是方法独有存储，不含共享索引、BFS缓存和容器元数据。','- 空标记表也可能因计时波动看起来略快；标记数为0或没有实际读取节省的配置不代表有效剪枝收益。小幅延迟差异应结合三轮范围判断。','- 所有性能包含实际在线查表或模型特征计算/评分；加载不计入查询延迟；读取是submitted SSD requests。完整配置、p95、QPS、各预算选择见REPORT.md和results.json。']
(ROOT/'SUMMARY.md').write_text('\n'.join(lines)+'\n');(ROOT/'verification.json').write_text(json.dumps(checks,indent=2))
complete=json.loads((ROOT/'COMPLETE.json').read_text());complete['independent_verification']=True
for name in ['SUMMARY.md','verification.json','finalize.py','baseline_diagnostic.json','run.sh']:
 complete['sha256'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
(ROOT/'COMPLETE.json').write_text(json.dumps(complete,indent=2))
print((ROOT/'SUMMARY.md').read_text())
