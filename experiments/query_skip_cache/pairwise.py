"""Explicit reuse ON/OFF pairs. No auto fallback or test-based policy selection."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
results=json.loads((ROOT/'results.json').read_text());pairs=[]
for ds,models in results.items():
 for model,workloads in models.items():
  for workload,v in workloads.items():
   for key,on in v['test'].items():
    if on['mode'] not in [2,3]:continue
    off=v['test']['fresh'];pairs.append({'dataset':ds,'model':model,'workload':workload,'policy':key,'reuse_off':off,'reuse_on':on,'auto_fallback':False})
(ROOT/'PAIRWISE.json').write_text(json.dumps(pairs,indent=2))
lines=['# 复用开关的直接对照','','两侧使用相同模型、模型阈值、输入query、索引、L=100、beam=4、4线程和BFS cache=10000。无复用=Fresh，每个miss候选重新预测；有复用=执行指定缓存复用策略，未命中仍预测，不使用自动回退。已完成的原实验包含两侧真实运行，以下是原始三轮测试中位数的重新整理，没有新增或重选测试参数。','','## 固定预设相似度阈值rho≤0.01：直接复用','','此配置是原协议预先列出的网格点，用于统一开关对照，不代表验证集最优配置。near为1%/5%相对L2扰动混合的200条测试query，GT独立重算。自然测试有1000条query。']
for workload in ['near','natural']:
 lines+=['',f'### {workload}','','|数据集|模型|复用开关|Recall@10 %|请求/q|模型调用/q|平均延迟ms|P95 ms|复用候选/q|','|---|---|---|---:|---:|---:|---:|---:|---:|']
 for p in pairs:
  if p['workload']!=workload or p['policy']!='reuse_r0.01':continue
  for label,key in [('关闭','reuse_off'),('开启','reuse_on')]:
   a=p[key];lines.append(f"|{p['dataset']}|{p['model']}|{label}|{a['recall_pct']:.3f}|{a['reads']:.3f}|{a['model_calls']:.3f}|{a['latency_us']/1000:.4f}|{a['p95_us']/1000:.4f}|{a['reused']:.3f}|")
lines+=['','## 所有预设阈值的配对变化','','每行均相对同一模型、同一workload的无复用基线；正的延迟降低代表更快。全部阈值及Guard策略在PAIRWISE.json保存两侧绝对值，不用测试选最优。','','|数据集|模型|workload|开启策略|请求减少|模型调用减少|延迟降低|额外Recall损失pp|','|---|---|---|---|---:|---:|---:|---:|']
for p in pairs:
 a=p['reuse_on'];lines.append(f"|{p['dataset']}|{p['model']}|{p['workload']}|{p['policy']}|{a['net_request_saving']:.2%}|{a['model_call_saving']:.2%}|{a['latency_saving']:.2%}|{a['recall_loss_pp']:.3f}|")
lines+=['','自然查询在固定rho≤0.01下实际复用为零，因此模型调用数、读取数、结果不变；小幅时延差异不能作为复用收益。near场景的读取变化来自搜索路径变化，不应全部归因于减少推理。缓存查找与成员查表成本已包含在有复用延迟中。']
(ROOT/'PAIRWISE.md').write_text('\n'.join(lines)+'\n')
complete=json.loads((ROOT/'COMPLETE.json').read_text())
for name in ['pairwise.py','PAIRWISE.md','PAIRWISE.json']:complete['sha256'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
(ROOT/'COMPLETE.json').write_text(json.dumps(complete,indent=2))
print('Explicit ON/OFF pairs:',len(pairs))
