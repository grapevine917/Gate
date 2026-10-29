from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'results.json').read_text());complete=json.loads((ROOT/'COMPLETE.json').read_text());assert complete['complete'] and complete['all_cross_experiment_output_checks_passed']
lock=json.loads((ROOT/'CONFIG_LOCK.json').read_text());checks={}
for ds,ws in r.items():
 for w,v in ws.items():
  assert set(v['metrics'])=={c['key'] for c in lock['configs'][ds]}
  assert (ROOT/'CONFIG_LOCK.json').stat().st_mtime_ns<=(ROOT/ds/w/'command.json').stat().st_mtime_ns
  assert v['verification']['deterministic_three_repetitions'] and v['verification']['new_timing_only']
  base=v['metrics']['DiskANN']
  for key,a in v['metrics'].items():
   assert abs(a['request_saving_vs_diskann']-(1-a['reads']/base['reads']))<1e-12
   assert abs(a['latency_saving_vs_diskann']-(1-a['latency_us']/base['latency_us']))<1e-12
  checks[f'{ds}/{w}']={'same_config_set':True,'config_lock_before_test':True,'same_timing_harness':True,'cross_experiment_output_checks':v['verification']['old_outputs_match']}
for path,digest in lock['artifacts_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
order=['DiskANN','Rank','Rank_Progress','LR','Top3_MLP','All17_MLP','Joint_LR_Top3','History44_LR','Mark_naive','Mark_conservative','LR_Reuse','LR_Guard','Top3_Reuse','Top3_Guard']
labels={'DiskANN':'原始DiskANN','Rank':'Rank规则','Rank_Progress':'Rank+Progress规则','LR':'LR：无复用','Top3_MLP':'Top3 MLP：无复用','All17_MLP':'All17 MLP','Joint_LR_Top3':'LR+Top3联合剪枝','History44_LR':'历史44维LR','Mark_naive':'全局标记：一次观察','Mark_conservative':'全局标记：保守','LR_Reuse':'LR：开启复用','LR_Guard':'LR：带状态检查复用','Top3_Reuse':'Top3 MLP：开启复用','Top3_Guard':'Top3 MLP：带状态检查复用'}
lines=['# 所有基线与复用开关：统一复测摘要','','本轮重新交错测量全部策略，未拼接旧计时；三轮测试中位数。L=100、beam=4、K=10、4线程、BFS cache10000。模型阈值来自之前固定验证参数；复用阈值rho<=0.01、没有自动回退。模型阈值/标签预算并不等价于相同实际Recall，故不能把更低Recall的成本优势宣传为同等质量优势。','','单元格依次为 **Recall@10 % / 请求数每query / 平均延迟ms**。原始DiskANN不运行剪枝模型；LR和Top3无复用仍启用学习式剪枝。']
for w,title in [('natural','自然独立查询：各1000条'),('near','高相似查询：各200条，1%/5%扰动混合')]:
 lines+=['',f'## {title}','','|方法|SIFT：Recall / 请求 / ms|DEEP：Recall / 请求 / ms|','|---|---|---|']
 for key in order:
  cells=[]
  for ds in ['sift','deep']:
   a=r[ds][w]['metrics'].get(key);cells.append('未有对应旧模型/参数' if a is None else f"{a['recall_pct']:.3f} / {a['reads']:.3f} / {a['latency_us']/1000:.3f}")
  lines.append(f"|{labels[key]}|{cells[0]}|{cells[1]}|")
lines+=['','## 高相似查询：相对原始DiskANN的总收益','','|数据集|策略|请求减少|延迟降低|Recall损失pp|','|---|---|---:|---:|---:|']
for ds in ['sift','deep']:
 for key in ['LR','Top3_MLP','Joint_LR_Top3','LR_Reuse','Top3_Reuse']:
  a=r[ds]['near']['metrics'][key];lines.append(f"|{ds}|{labels[key]}|{a['request_saving_vs_diskann']:.2%}|{a['latency_saving_vs_diskann']:.2%}|{a['recall_loss_vs_diskann_pp']:.3f}|")
lines+=['','## 高相似查询：开启复用的额外收益','','此表以对应模型关闭复用为分母，与上表不同。','','|数据集|开启策略|请求额外减少|模型调用减少|延迟额外降低|额外Recall损失pp|','|---|---|---:|---:|---:|---:|']
for ds in ['sift','deep']:
 for key,a in r[ds]['near']['reuse_increment'].items():lines.append(f"|{ds}|{labels[key]}|{a['request_saving']:.2%}|{a['model_call_saving']:.2%}|{a['latency_saving']:.2%}|{a['recall_loss_pp']:.3f}|")
lines+=['','需要区分：原始DiskANN→学习式剪枝的收益，以及学习式剪枝→决定复用的额外收益。自然查询在rho<=0.01下没有实际复用命中，读取与模型调用不变；小幅时间差异不能算复用命中收益。近似查询的父query在历史训练缓存中，这是刻意构造的高命中场景，不代表一般未见查询；所有方法共享这一场景，包括从训练查询构建的全局Mark。','','完整逐配置Recall、模型调用数、p95、QPS、三轮范围、受损query数见REPORT.md与results.json。Rank/Rank+Progress/All17仅SIFT有旧训练产物。History44旧模型使用300条训练query，训练条件不同。所有收益数字是固定参数点的比较，本轮未按相同Recall重调参。']
(ROOT/'SUMMARY.md').write_text('\n'.join(lines)+'\n');(ROOT/'verification.json').write_text(json.dumps(checks,indent=2))
for name in ['SUMMARY.md','summarize.py','verification.json','PROTOCOL.md','run.sh']:complete['sha256'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
complete['summary_verified']=True;(ROOT/'COMPLETE.json').write_text(json.dumps(complete,indent=2));print((ROOT/'SUMMARY.md').read_text())
