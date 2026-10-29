from pathlib import Path
import json,csv,hashlib,html
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
R=Path(__file__).resolve().parent;LG=R.parents[1]
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
summary=json.loads((R/'selected_results.json').read_text());val=json.loads((R/'validation.json').read_text());test=json.loads((R/'test.json').read_text());offline=json.loads((R/'offline_metrics.json').read_text());verification=json.loads((R/'verification.json').read_text());parts=json.loads((R/'PROTOCOL.json').read_text())['query_partition']
p=np.load(R.parent/'e4_1_redundant_keep/keep_positions.npz');o=np.load(R.parent/'e4_1_redundant_keep/outcomes.npz');pr=np.load(R/'label_predictions.npz');diag={}
selected=json.loads((R/'SELECTION_LOCK.json').read_text())
assert selected['test_used'] is False and selected['model_sha256']==sha(R/'model.bin')
for split in ['fit','validation','test']:
 m=np.isin(o['query_row'],parts[split]);y=o['delta_hits'][m]<0
 probs=pr['risk_probability'][m];label=o['delta_IO'][m];gain=pr['predicted_gain'][m]
 original=p['model_score'][m]
 diag[split]=dict(original_MLP_score_risk_AUC=roc_auc_score(y,original),original_MLP_score_risk_AP=average_precision_score(y,original),gain_label_mean=float(label.mean()),gain_label_std=float(label.std()),gain_label_fraction_equal_1=float((label==1).mean()),gain_pred_std=float(gain.std()))
 bins=[];ece=0
 for lo in np.arange(0,1,.1):
  b=(probs>=lo)&(probs<lo+.1 if lo<.9 else probs<=1)
  if not b.any():continue
  count=int(b.sum());pred=float(probs[b].mean());actual=float(y[b].mean());ece+=count/len(y)*abs(pred-actual)
  bins.append(dict(lo=float(lo),n=count,predicted_risk=pred,actual_harm_rate=actual))
 diag[split]['risk_ECE_10bins']=ece;diag[split]['calibration']=bins
 if split=='test':
  policy=[];configs={c['key']:c for c in json.loads((R/'configs.json').read_text())}
  for s in summary:
   if s['family']!='CF' or s['key'] is None:continue
   c=configs[s['key']];skip=(probs<c['rho'])&(gain>c['gmin'])
   policy.append(dict(target=s['target'],key=s['key'],rho=c['rho'],gain_min=c['gmin'],labeled_candidates=int(m.sum()),predicted_skip=int(skip.sum()),harmful_skipped=int(y[skip].sum()),conditional_harm_rate=float(y[skip].mean()) if skip.any() else None,mean_predicted_gain=float(gain[skip].mean()) if skip.any() else None,mean_actual_single_node_gain=float(label[skip].mean()) if skip.any() else None))
  diag[split]['selected_policy_on_original_Keep_labels_only']=policy
dump(R/'offline_diagnostics.json',diag)
configs={c['key']:c for c in json.loads((R/'configs.json').read_text())}
lines=['# 单节点反事实监督 vs 原始17维MLP强基线','','## 设置与标签','','用户所称E4-B1在当前目录对应E4-1单节点强制Skip数据（93,410个原MLP Keep cache-miss位置）。沿用既有query划分：500条fit、250条validation、250条test，互不重叠。该1000-query池此前已被实验观察，属于探索性池内划分。','','标签：风险为`delta_hits<0`；I/O收益为有符号`baseline_reads-after_reads`，负收益保留。输入严格为相同17维pre-I/O状态，不输入节点ID、原MLP分数、历史或GT。标签只覆盖原固定策略Keep位置，不代表所有候选的完整训练支持。','','新模型为17→128→64共享ReLU骨干、风险和I/O两个输出；风险用sigmoid，收益输出有符号实数。联合损失为未加权BCE加标准化收益MSE。仅fit查询拟合归一化；内部400/100 query划分选epoch，随后全部500条fit重训。原始MLP及DiskANN源码均未修改，在线集成位于独立实验库。','','## 两个在线对照','','1. 原始冻结17维MLP，单阶段全局阈值扫描，共63个阈值。','2. 反事实模型直接替换所有cache-miss处的原MLP预测，Skip条件为`risk<rho && predicted_IO>gmin`，共50个配置，包括关闭I/O门槛的risk-only配置和不剪枝配置。不是原MLP之后再加一个剪枝阶段。','','二者使用相同真实DiskANN图搜索、L100、beam4、K10、4线程、10000节点静态BFS缓存，缓存命中保护。每次输入来自实际改变后的搜索轨迹；不使用离线决策回放。每个Recall下限分别在验证集选最小SSD请求配置，先冻结后测试，不用测试调阈值。','','## 标签上的预测质量（测试集原策略Keep位置）','','|指标|原始MLP分数作为风险排序|反事实模型|','|---|---:|---:|',f"|风险ROC-AUC|{diag['test']['original_MLP_score_risk_AUC']:.4f}|{offline['test']['risk_AUC']:.4f}|",f"|风险AP|{diag['test']['original_MLP_score_risk_AP']:.4f}|{offline['test']['risk_AP']:.4f}|",f"|风险Brier|不将原retain分数当作风险概率|{offline['test']['risk_Brier']:.4f}|",'',f"测试风险正类比例为{offline['test']['risk_prevalence']*100:.2f}%。收益预测MAE={offline['test']['gain_MAE']:.4f}请求，RMSE={offline['test']['gain_RMSE']:.4f}，R²={offline['test']['gain_R2']:.4f}；始终预测fit集收益均值的MAE={offline['test']['gain_constant_fit_mean_MAE']:.4f}。收益头未在MAE上超越该常数对照，不能据风险排序提升宣称I/O收益预测也有效。",'','## 验证选择后真实在线测试','','下表全部为单配置实测，非插值。成本按每query计；延迟为三轮平均query延迟的中位数，范围见selected_results.csv。验证Recall下限相同不等于测试Recall严格匹配；未达到测试下限明确标记，不据此宣布同Recall优胜。','','|验证Recall下限%|策略|配置|验证Recall%|测试Recall%|SSD请求/q|展开/q|延迟µs/q|测试达标|','|---|---|---|---:|---:|---:|---:|---:|---|']
for s in summary:
 if s['key'] is None:lines.append(f"|{s['target']}|{s['family']}|不可行||||||否|");continue
 c=configs[s['key']];setting=f"tau={c['tau']:.6g}" if s['family']=='MLP' else f"rho={c['rho']:g}, IO>{c['gmin']:g}"
 lines.append(f"|{s['target']}|{s['family']}|{setting}|{s['validation_recall']:.3f}|{s['test_recall']:.3f}|{s['reads']:.3f}|{s['expansions']:.3f}|{s['latency_us']:.2f}|{'是' if s['test_meets_target'] else '否'}|")
lines+=['','## 配对查询置信区间','','CF减去强基线；2000次配对query bootstrap，条件于当前训练模型及选定配置，不覆盖训练不确定性。','','|Recall下限%|测试双方达标|Recall差pp [95%CI]|SSD请求差/q [95%CI]|','|---|---|---|---|']
for c in json.loads((R/'paired_comparisons.json').read_text()):
 r=c['recall_pp_ci95'];d=c['reads_ci95'];lines.append(f"|{c['target']}|{'是' if c['both_test_feasible'] else '否'}|{c['recall_pp_delta']:.3f} [{r[0]:.3f}, {r[1]:.3f}]|{c['reads_delta']:.3f} [{d[0]:.3f}, {d[1]:.3f}]|")
lines+=['','## 结论与边界','','反事实监督明显提高了原Keep节点上的风险排序质量；收益回归目前很弱。在线测试没有全面超越强基线：99.44%下限时，CF以更多请求换得更高Recall；99.20%下限时也更保守且成本更高。99.00%下限时双方达标，CF请求从60.856降到59.280（约2.59%），但Recall从99.20%降到99.16%，延迟从4409.90增到4481.66µs，未显示延迟收益。99.60%下限时CF达到99.60%，原MLP仅99.56%，不属于双方均满足该测试约束的对照。即使双方都达标，也不冒充严格相同实测Recall。','','单节点标签针对原MLP轨迹，多个Skip的影响不可相加；新模型在原Skip节点和改变后的状态上存在分布外应用。完整在线搜索是对这种替换的实测检验，不是证明单节点监督能够自动解决联合剪枝。SSD为提交请求，暖OS缓存下的延迟不等于冷盘端到端延迟。','','## 验证与复现','',f"原生模型与PyTorch在{verification['native_vs_pytorch']['states']}个测试标签状态上比较，风险最大误差{verification['native_vs_pytorch']['risk_max_abs_error']:.3g}，收益最大误差{verification['native_vs_pytorch']['gain_max_abs_error']:.3g}。原始MLP在三个阈值的最终ID及请求/展开/Skip/调用数与既有程序完全一致；CF关闭剪枝后也与无剪枝路径完全一致。全部在线Recall独立从保存的ID与GT重算；三轮重复结果ID及计数一致；原始源码、模型和输入标签哈希未变。",'','复现：`bash experiments/counterfactual_supervision/run.sh`。已有完成的在线输出会复用，并检查模型、程序和配置锁。','', '主要文件：PROTOCOL.json、INPUT_LOCK.json、model.bin、model.pt、offline_metrics.json、offline_diagnostics.json、validation_summary.csv、SELECTION_LOCK.json、selected_results.csv、paired_comparisons.json、verification.json。','', '![验证集Recall–SSD曲线与测试选定配置](curves.svg)']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
# Standalone vector plot. Show actual points, no interpolation or test reselection.
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="450"><rect width="1000" height="450" fill="white"/><g font-family="sans-serif" fill="#222"><text x="30" y="28" font-size="17">Counterfactual supervision vs original 17D MLP</text>']
for i,(rows,title) in enumerate([(val,'Validation grid (actual points)'),([a for a in test if a['repeat']==0],'Frozen test configurations (actual points)')]):
 x0=65+i*500;y0=70;w=375;h=300
 def xx(v):return x0+(v-98.8)/1.1*w
 def yy(v):return y0+h-(v-35)/75*h
 svg.append(f'<defs><clipPath id="c{i}"><rect x="{x0}" y="{y0}" width="{w}" height="{h}"/></clipPath></defs><text x="{x0}" y="53" font-size="14">{title}</text>')
 for t in [99,99.2,99.4,99.6,99.8]:svg.append(f'<path d="M{xx(t)} {y0}V{y0+h}" stroke="#ddd"/><text x="{xx(t)}" y="{y0+h+20}" font-size="11" text-anchor="middle">{t:.1f}</text>')
 for t in [40,60,80,100]:svg.append(f'<path d="M{x0} {yy(t)}H{x0+w}" stroke="#ddd"/><text x="{x0-7}" y="{yy(t)+4}" font-size="11" text-anchor="end">{t}</text>')
 svg.append(f'<g clip-path="url(#c{i})">')
 for a in rows:
  color='#2563eb' if a['family']=='MLP' else '#dc2626'
  svg.append(f'<circle cx="{xx(a["recall"])}" cy="{yy(a["reads"])}" r="3.4" fill="{color}" fill-opacity=".7"><title>{html.escape(a["key"])}: Recall {a["recall"]:.3f}%, SSD {a["reads"]:.3f}</title></circle>')
 svg.append(f'</g><text x="{x0+w/2}" y="417" font-size="12" text-anchor="middle">Recall@10 (%)</text><text x="{x0}" y="438" font-size="12">SSD requests / query; blue: original MLP, red: CF</text>')
svg.append('</g></svg>');(R/'curves.svg').write_text(''.join(svg))
locks=json.loads((R/'INPUT_LOCK.json').read_text());assert all(sha(Path(p))==v for p,v in locks.items())
files=[p for p in R.iterdir() if p.suffix in ['.py','.h','.cpp','.sh']]+[R/'native/collect',R/'native/pq_flash_index.cpp',R/'native/online_gate.h',R/'native/risk_model.h',R/'model.bin']
dump(R/'PROVENANCE.json',{str(p.relative_to(LG)):sha(p) for p in files});dump(R/'COMPLETE.json',dict(completed=True,original_files_unchanged=True,online=json.loads((R/'ONLINE_COMPLETE.json').read_text()),report='REPORT.md'))
dump(R/'artifact_manifest.json',{str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in R.rglob('*') if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in str(p)})
print('REPORT COMPLETE')
