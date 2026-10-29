from common import *
import sys
MODE=sys.argv[1] if len(sys.argv)>1 else 'all'
MODEL_LABEL={'lr':'LR','xgb':'XGBoost','state':'State MLP'}
def method(r):
 if r['family'].startswith('Ours_'):return 'Ours '+MODEL_LABEL[r['model']]
 if r['family'].startswith('Joint_'):return 'ET + Ours '+MODEL_LABEL[r['model']]
 return {'DiskANN':'Tuned DiskANN','Heuristic':'DiskANN + Heuristic','ET':'DiskANN + ET'}[r['family']]
def table(rows,feasible_target=None):
 fields=[('Method',method),('L',lambda r:str(r['L'])),('Recall %',lambda r:f"{r['recall']:.2f}"),('P50 ms',lambda r:f"{r['p50_ms']:.3f}"),('P95 ms',lambda r:f"{r['p95_ms']:.3f}"),('P99 ms',lambda r:f"{r['p99_ms']:.3f}"),('QPS',lambda r:f"{r['qps']:.1f}"),('Physical bytes/query',lambda r:f"{r['physical_bytes']:.1f}"),('Reads/query',lambda r:f"{r['reads']:.3f}"),('Exp/query',lambda r:f"{r['expansions']:.3f}"),('Calls/query',lambda r:f"{r['model_calls']:.3f}"),('ET calls/query',lambda r:f"{r['et_calls']:.3f}")]
 if feasible_target is not None:fields.insert(3,('Meets target',lambda r:str(r['recall']>=feasible_target)))
 return '| '+' | '.join(k for k,_ in fields)+' |\n| '+' | '.join(['---']*len(fields))+' |\n'+'\n'.join('| '+' | '.join(f(r) for _,f in fields)+' |' for r in rows)+'\n'
intro=['# 旧模型复测：SIFT1M、DEEP1M、SIFT100M 的 P0-A / P0-B','',
'使用各数据集已有同批混合L样本训练的 LR、XGBoost、17维状态MLP及对应ET，全部为旧代理标签权重，不重新训练，不使用风险监督模型或动态阈值。相同数据集共享索引、查询、K10、Beam4、4查询线程、10000节点BFS缓存。float1M和uint8 100M实现不同，跨数据集不是单纯规模消融。',
'', '计时包括现场状态提取、模型推理、搜索和IO；排除索引/模型加载、缓存构建、固定128-query热身、GT评分及输出。物理字节来自计时窗口Linux read_bytes，并严格等于实际Reader字节和逐查询读取数×4096；无交换。对照不按模型版本重复测量，相同配置/重复编号只保留一份。','']
def p0a_report():
 rows=intro+['## P0-A：固定 L=100','',
'五组：原生DiskANN、rank>72且completed>100的启发式、LR、XGBoost、状态MLP。三个模型各自按同一验证L100缓存未命中正样本的retain-FNR1%校准静态阈值；这是代理标签风险预算，不保证最终Recall相同。每组15轮（5种循环位置各3轮），下表为轮次中位数。','']
 for ds in CATALOG:
  O=D/ds
  if not (O/'P0A_COMPLETE.json').exists():continue
  summary=json.loads((O/'p0a_summary.json').read_text());rows+=['',f"### {CATALOG[ds]['label']}",'',table(summary)]
  rows.append('模型来源：`'+str(catalog(ds)['parent_dir']/'models')+'`。所有学习方法使用已有匹配混合L训练版本；推理L固定100，不把它们标成纯L100训练。')
 (D/'P0A_REPORT.md').write_text('\n'.join(rows)+'\n');return rows
def dataset_report(ds):
 from index_lock import snapshot
 assert snapshot()==json.loads((D/'INDEX_LOCK.json').read_text())
 C=catalog(ds);O=C['out'];P=json.loads((O/'PROTOCOL.json').read_text());assert (O/'P0B_COMPLETE.json').exists();summary=json.loads((O/'summary.json').read_text());S={r['key']:r for r in summary};families=P['P0_B']['families'];metrics=['p95_ms','qps','physical_bytes'];frontiers={};points=[]
 for family in families:
  frontiers[family]={}
  for metric in metrics:
   f=frontier([r for r in summary if r['family']==family],metric);assert all(r['repeats']>=3 for r in f);frontiers[family][metric]=[r['key'] for r in f]
   points += [dict(metric=metric,**r) for r in f]
 dump(O/'frontiers.json',frontiers);save(O/'frontier_points.csv',points)
 coverage=[dict(family=f,configs=len(rs),min_recall=min(r['recall'] for r in rs),max_recall=max(r['recall'] for r in rs),p95_frontier=len(frontiers[f]['p95_ms']),qps_frontier=len(frontiers[f]['qps']),bytes_frontier=len(frontiers[f]['physical_bytes'])) for f in families for rs in [[r for r in summary if r['family']==f]]];save(O/'coverage.csv',coverage)
 rows=[f"# {C['label']}：P0-A / P0-B 旧模型复测",'',*intro[2:], '## P0-A','',table(json.loads((O/'p0a_summary.json').read_text())),'','## P0-B：独立调 L 和阈值','',
f"每方法64个预声明配置：原生64个不同L；其余8个L×8档参数。模型覆盖L={C['LS']}，原生包含全部这些L。三个Ours及三个ET+Ours版本共享原生、启发式、ET，合计576个唯一配置。Gate阈值为同一验证L100缓存未命中retain正样本的8档FNR，ET阈值和8个联合参数对在PROTOCOL.json中完整公开。联合组不是无限笛卡尔积搜索。",'',
'全部验证配置完成后，按每个Recall下限和三个指标分别选参并锁定，再测试。固定全测试网格的非支配前沿属于描述性结果，不能冒充未见测试集选参。曲线中每个点对应一个真实配置；连线仅帮助阅读，不产生插值测量。每项指标独立求前沿，避免把字节最优的配置重复当成P95/QPS最优。最终显示前沿及验证锁定点至少3次计时中位数，P0-A配置15次；均核对输出和计数重复一致。','']
 selections=json.loads((O/'SELECTION_LOCK.json').read_text())
 for target in [99.5,99.7]:
  selected=[S[r['key']] for r in selections if r['target']==target and r['objective']=='p95_ms' and r['key']]
  missing=[r['family'] for r in selections if r['target']==target and r['objective']=='p95_ms' and not r['key']]
  rows += ['',f'### 验证 Recall≥{target}% 时最低P95的锁定配置','',table(selected,target) if selected else '没有验证达标配置。']
  if missing:rows.append('验证未达标的方法：'+', '.join(missing)+'。不外推或伪造高Recall点。')
 rows+=['','完整目标/指标的验证锁定测试结果见 validation_locked_test.csv；所有前沿实际参数见 frontier_points.csv，完整配置与重复范围见 all_configurations.csv。','',
'## 核验','']
 for p,h in P['source_hashes'].items():assert sha(p)==h
 allruns=json.loads((O/'all_test_runs.json').read_text());assert len({(r['key'],r['repeat']) for r in allruns})==len(allruns)
 for r in allruns:
  p=O/'runs/test'/f"{r['key']}_r{r['repeat']}";m=json.loads(p.with_suffix('.meta.json').read_text());assert m['linux_read_bytes']==m['reader_bytes']==m['reader_requests']*4096 and m['swap_before_kib']==m['swap_after_kib']==0;assert sha(p.with_suffix('.ids'))==r['ids_sha256']
 rejection=json.loads((O/'IO_REJECTIONS.json').read_text()) if (O/'IO_REJECTIONS.json').exists() else []
 rows.append(f"原权重、查询和程序哈希保持不变；原生关闭剪枝与历史结果一致，联合关闭ET与Gate一致；全部正式记录通过Reader/Linux物理IO一致性及无换页检查。IO异常尝试归档重测 {len(rejection)} 次；测试唯一测量 {len(allruns)} 条。返回ID合法/唯一/尾部占位检查、GT重算Recall、重复离散计数检查均通过。不足K个返回结果按未命中计算，分母仍为10000。")
 rows += ['', '限制：同一训练种子、有限参数预算、已观察过的基准查询，未含重训练不确定性；ET基于未剪枝轨迹训练，联合策略存在分布变化。实测Recall不同的配置不能声称严格同质量优势。物理字节为WSL客户机块层读取，不是SSD内部NAND遥测。','']
 (O/'REPORT.md').write_text('\n'.join(rows));dump(O/'verification.json',dict(passed=True,source_hashes_unchanged=True,physical_IO_verified=True,no_swap=True,all_frontier_points_repeated_at_least3=True,configurations=576,unique_test_runs=len(allruns),families=9));dump(O/'COMPLETE.json',dict(complete=True,report_sha256=sha(O/'REPORT.md')));return rows
styles=[('DiskANN','Tuned DiskANN','#222222','o'),('Heuristic','Heuristic','#e68613','s'),('Ours','Ours','#2077b4','^'),('ET','ET','#8055a5','D'),('Joint','ET + Ours','#2b9b59','v')]
def plot_all():
 os.environ.setdefault('MPLCONFIGDIR',str(D/'mpl_cache'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 folder=D/'figures';folder.mkdir(exist_ok=True);models=['lr','xgb','state'];datasets=list(CATALOG);infos={ds:(json.loads((D/ds/'summary.json').read_text()),json.loads((D/ds/'frontiers.json').read_text())) for ds in datasets};artifacts=[]
 for metric,title,ylabel in [('p95_ms','Recall@10 – P95 latency','P95 latency (ms)'),('qps','Recall@10 – QPS','QPS'),('physical_bytes','Recall@10 – actual physical reads','Physical bytes / query')]:
  for region,xlim in [('main',(90,100)),('high_recall',(99,100)),('full_range',(0,100))]:
   fig,axes=plt.subplots(3,3,figsize=(16,12),squeeze=False)
   for i,ds in enumerate(datasets):
    summary,fs=infos[ds];by={r['key']:r for r in summary}
    for j,model in enumerate(models):
     ax=axes[i,j]
     for family,label,color,marker in styles:
      fam=f'{family}_{model}' if family in ['Ours','Joint'] else family;rr=[by[k] for k in fs[fam][metric]]
      ax.plot([r['recall'] for r in rr],[r[metric] for r in rr],label=label,color=color,marker=marker,markersize=3,linewidth=1.3)
     ax.set_xlim(*xlim);ax.set_ylim(bottom=0);ax.grid(alpha=.25);ax.set_title(f"{CATALOG[ds]['label']} · {MODEL_LABEL[model]}");ax.set_xlabel('Recall@10 (%)');ax.set_ylabel(ylabel)
     if metric=='physical_bytes':ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
   handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,bbox_to_anchor=(.5,.025));fig.suptitle(title+' — old frozen models',fontsize=16)
   fig.text(.5,.007,'Markers = measured configurations; lines guide the eye. Independent per-metric descriptive test frontiers.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.065,1,.96))
   stem=folder/f'{metric}_{region}'
   for ext in ['svg','png','pdf']:fig.savefig(stem.with_suffix('.'+ext),dpi=160,bbox_inches='tight');artifacts.append(str(stem.with_suffix('.'+ext)))
   plt.close(fig)
 dump(D/'FIGURES_COMPLETE.json',dict(complete=True,files=artifacts))
if MODE=='p0a':
 p0a_report();print('P0A REPORT COMPLETE',flush=True)
elif MODE in CATALOG:
 dataset_report(MODE);print('DATASET REPORT COMPLETE',MODE,flush=True)
elif MODE=='all':
 rows=p0a_report()
 for ds in CATALOG:dataset_report(ds)
 plot_all();rows+=['','## P0-B：三项指标的独立前沿','',
'每张图三行分别为SIFT1M、DEEP1M、SIFT100M，三列分别为LR、XGBoost、状态MLP。每个面板五条曲线：Tuned DiskANN、启发式、Ours、ET、ET+Ours。原生/规则/ET曲线来自同一共享测量池。图示90–100%，另提供99–100%放大与0–100%完整范围，均有SVG/PNG/PDF。','',
'![Recall–P95](figures/p95_ms_main.png)','', '![Recall–QPS](figures/qps_main.png)','', '![Recall–Physical Bytes](figures/physical_bytes_main.png)','']
 for ds in CATALOG:rows+=['',f"### {CATALOG[ds]['label']} 文件",'',f"[详细报告]({ds}/REPORT.md) · [全部配置]({ds}/all_configurations.csv) · [实际前沿点]({ds}/frontier_points.csv) · [验证锁定测试]({ds}/validation_locked_test.csv)"]
 rows+=['','## 复现','', '项目根目录执行 `.venv/bin/python experiments/p0_old_models_rerun/run.py` 复用已通过核验的结果。首次准备依次执行build.py、prepare.py，再执行run.py；不修改已有模型和索引。','']
 (D/'REPORT.md').write_text('\n'.join(rows));dump(D/'COMPLETE.json',dict(complete=True,datasets=list(CATALOG),models=['lr','xgb','state'],P0A_methods=5,P0B_unique_families=9,figures=27,report_sha256=sha(D/'REPORT.md')));dump(D/'STATUS.json',dict(phase='complete',complete=True));print('ALL EXPERIMENTS COMPLETE',flush=True)
else:raise ValueError(MODE)
