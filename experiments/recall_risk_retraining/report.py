from common import *
assert (D/'ONLINE_COMPLETE.json').exists() and (D/'OFFLINE_COMPLETE.json').exists()
P=json.loads((D/'PROTOCOL.json').read_text());S=json.loads((D/'SELECTION_LOCK.json').read_text());OP=json.loads((D/'ONLINE_PROTOCOL.json').read_text());metrics=json.loads((D/'offline_metrics.json').read_text())
names={('old','MLP'):'Old proxy MLP',('old','LR'):'Old proxy LR',('proxy','MLP'):'Matched proxy MLP',('proxy','LR'):'Matched proxy LR',('risk','MLP'):'Recall-risk MLP',('risk','LR'):'Recall-risk LR'}
def name(r):return names.get((r['variant'],r['family']),r['family'])
def aggregate(rows,keys):
    out=[]
    for key in keys:
        rr=[r for r in rows if r['key']==key];r=rr[0].copy();r.pop('repeat');r['repeats']=len(rr)
        for k in ['total_hits','physical_bytes','reads','expansions','skips','model_calls','io_rounds','ids_sha256']:assert len({a[k] for a in rr})==1,(key,k)
        for k in ['p95_ms','p50_ms','p99_ms','qps','mean_ms','io_ms']:r[k]=float(np.median([a[k] for a in rr]))
        r['p95_min_ms']=min(a['p95_ms'] for a in rr);r['p95_max_ms']=max(a['p95_ms'] for a in rr);r['test_meets_validation_floor']=r['total_hits']>=S['validation_floor_hits'];out.append(r)
    return out
def table(rows):
    fs=[('Method',name),('τ',lambda r:f"{r['tau']:.7g}"),('Recall %',lambda r:f"{r['recall']:.2f}"),('Meets floor',lambda r:str(r['test_meets_validation_floor'])),('P50 ms',lambda r:f"{r['p50_ms']:.4f}"),('P95 ms',lambda r:f"{r['p95_ms']:.4f}"),('P99 ms',lambda r:f"{r['p99_ms']:.4f}"),('Mean ms',lambda r:f"{r['mean_ms']:.4f}"),('QPS',lambda r:f"{r['qps']:.1f}"),('KiB/query',lambda r:f"{r['physical_bytes']/1024:.2f}"),('Reads/query',lambda r:f"{r['reads']:.3f}"),('Exp/query',lambda r:f"{r['expansions']:.3f}"),('Skips/query',lambda r:f"{r['skips']:.3f}"),('Calls/query',lambda r:f"{r['model_calls']:.3f}"),('IO ms',lambda r:f"{r['io_ms']:.4f}"),('IO rounds',lambda r:f"{r['io_rounds']:.3f}")]
    return '| '+' | '.join(k for k,_ in fs)+' |\n| '+' | '.join(['---']*len(fs))+' |\n'+'\n'.join('| '+' | '.join(f(r) for _,f in fs)+' |' for r in rows)+'\n'
primary=json.loads((D/'primary_test_runs.json').read_text());defaults=json.loads((D/'default_test_runs.json').read_text());allruns=json.loads((D/'all_test_runs.json').read_text())
summary=aggregate(primary,[c['key'] for c in S['configs']]);default_summary=aggregate(defaults,S['default_keys']);assert all(r['repeats']==16 for r in summary+default_summary)
save(D/'selected_results.csv',summary);dump(D/'selected_results.json',summary);save(D/'default_results.csv',default_summary)
rows=['# SIFT1M：最终 Recall 风险监督重训','',
'仅SIFT1M，完整100万条128维float索引。部署 L=100、K=10、beam=4、4查询线程、缓存10000节点。训练1200条查询，验证/测试各1000，三者互不重叠。旧模型及数据保持不变，所有新模型、标签和原始测量独立保存。本轮不使用动态阈值。',
'','## 宽 L 与风险标签的作用','',
'每条查询先做L=300和1000搜索，用真实GT确认其到达的近邻，再按首次发现且入队的边回溯父路径。路径标记仅用于离线难例分层，绝不输入模型。L=100的未剪枝、旧MLP、旧LR三种轨迹覆盖原本Keep和Skip的所有未缓存候选。',
'', '对被抽中的候选，从查询起点重跑同一前缀，严格核对候选序号、节点和字节完全相同的17维状态，单次强制Keep/Skip，之后继续原冻结策略。复用原决策分支，并额外运行空干预；标签risk=1当且仅当Keep的最终GT命中数大于Skip。另保存净损失幅度、近邻身份丢失/新增及有符号IO收益，训练本轮二分类风险。',
'','按teacher路径、晚期非路径、其余节点三层抽样，使用population/sample权重还原候选总体。宽L不是用更宽队列的节点价值直接替代L100价值，而是帮助发现值得检查的路径；风险由实际L100分支结果给出。精确GT已存在，宽L不增加未知的真实近邻。','']
teacherrows=[]
for split in ['train','validation','test']:
    a=labels(split);m=json.loads((D/f'LABELS_{split.upper()}_COMPLETE.json').read_text());q=np.concatenate([np.genfromtxt(p,delimiter=',',names=True) for p in sorted((D/'labels'/split).glob('chunk_*.csv'))])
    teacher=dict(split=split,queries=len(q),samples=len(a),sampled_risk_fraction=m['risk_fraction'],population_weighted_risk_fraction=m['weighted_risk_fraction'],teacher300_recall=float(q['teacher300_hits'].mean()*10),teacher1000_recall=float(q['teacher1000_hits'].mean()*10),baseline100_recall=float(q['off_hits'].mean()*10),MLP100_recall=float(q['mlp_hits'].mean()*10),LR100_recall=float(q['lr_hits'].mean()*10),shams=int(q['shams'].sum()))
    teacherrows.append(teacher);rows.append(f"- {split}：{len(q)}查询、{len(a)}样本；加权有害比例 {m['weighted_risk_fraction']*100:.2f}%；L100/teacher300/teacher1000 Recall={teacher['baseline100_recall']:.2f}%/{teacher['teacher300_recall']:.2f}%/{teacher['teacher1000_recall']:.2f}%；空干预 {teacher['shams']} 次。")
save(D/'teacher_and_label_summary.csv',teacherrows)
rows+=['','## 重训与匹配对照','',
'新风险LR/MLP与匹配代理标签LR/MLP使用完全相同的52756个状态和抽样权重，只更换监督目标。代理标签来自强制Keep分支的admitted>0或最终Top10。输入仍是17维pre-IO状态，不包含GT、节点ID、原模型分数、宽L标记或未来信息。MLP结构17-128-64-1，LR17维；加权BCE/逻辑回归，训练查询内部1000/200划分选轮次，再拟合全部1200条。归一化只来自训练查询，常量列权重固定为0。',
'', '旧代理模型使用历史纯L100训练样本（76800），因此主要标签消融应看新风险模型对匹配代理模型；旧模型作为实际现状基线。单种子。本轮没有额外训练不使用宽L采样的风险模型，不能单独量化宽L采样的净贡献。','',
'## 留出干预样本上的风险预测','',
'所有指标按逆抽样权重计算。旧/匹配代理模型的输出仅作风险排序，不能作为最终Recall风险概率解读。风险模型Brier仅检查本次采集分布下的概率误差。阈值在验证干预样本上按有害节点FNR≤1%标定，再报告测试FNR和安全节点剪枝覆盖率；这与在线搜索的最终Recall约束不同。','',
'| Model | Risk ROC-AUC | Risk AP | Risk Brier | Test harmful FNR | Test safe skip coverage |','| --- | --- | --- | --- | --- | --- |']
for r in metrics:
    if r['split']=='test':rows.append(f"| {names[(r['variant'],r['family'])]} | {r['risk_ROC_AUC']:.4f} | {r['risk_AP']:.4f} | {r['risk_Brier'] if r['risk_Brier'] is not None else '—'} | {r['test_risk_FNR']*100:.2f}% | {r['test_safe_skip_fraction']*100:.2f}% |")
rows+=['','## 验证锁定后的在线测试','',
f"各学习模型扫描同一预声明阈值网格，旧模型额外保留原默认阈值。统一验证Recall下限={S['validation_floor_recall']:.2f}%（旧静态默认两组的较低验证Recall），在达标配置中选择最少物理读取，随后更高Recall、较低阈值打破平局。选择在测试前锁定，测试不调参。DiskANN及固定启发式作为共享参考，启发式不保证满足此质量下限。",'',
'八组按八种循环位置各重复两次，共16轮/组；旧默认配置也在本轮复测16次。计时取各轮指标中位数，物理IO/输出计数在重复间一致。各行实际测试Recall不同，不能称为严格同Recall比较；未达测试下限如实标记。','',table(summary),'','### 原静态默认阈值参考','',table(default_summary)]
effects=[]
for family in ['MLP','LR']:
    risk=next(r for r in summary if r['family']==family and r['variant']=='risk')
    for variant in ['old','proxy']:
        baseline=next(r for r in summary if r['family']==family and r['variant']==variant)
        e=dict(family=family,baseline=variant,risk_key=risk['key'],baseline_key=baseline['key'],both_test_meet_floor=bool(risk['test_meets_validation_floor'] and baseline['test_meets_validation_floor']),recall_delta_pp=risk['recall']-baseline['recall'],p95_change_percent=(risk['p95_ms']/baseline['p95_ms']-1)*100,qps_change_percent=(risk['qps']/baseline['qps']-1)*100,physical_bytes_change_percent=(risk['physical_bytes']/baseline['physical_bytes']-1)*100)
        effects.append(e);rows.append(f"风险{family}相对{variant}：测试双方达标={e['both_test_meet_floor']}；Recall {e['recall_delta_pp']:+.2f} pp，P95 {e['p95_change_percent']:+.2f}%，QPS {e['qps_change_percent']:+.2f}%，物理字节 {e['physical_bytes_change_percent']:+.2f}%。")
save(D/'effects.csv',effects)
# Independent artifacts and physical IO audit, not just trusting summary flags.
with (W/'data/test.gt').open('rb') as f:n,k=np.fromfile(f,'<u4',2);gt=np.fromfile(f,'<u4').reshape(int(n),int(k))[:,:10]
assert len({(r['key'],r['repeat']) for r in allruns})==len(allruns);checked=set()
for r in allruns:
    p=D/'runs/test'/f"{r['key']}_r{r['repeat']}";m=json.loads(p.with_suffix('.meta.json').read_text());assert m['linux_read_bytes']==m['reader_bytes']==m['reader_requests']*4096;assert m['swap_before_kib']==m['swap_after_kib']==0;assert sha(p.with_suffix('.ids'))==r['ids_sha256']
    if r['key'] not in checked:
        ids=np.fromfile(p.with_suffix('.ids'),'<u8').reshape(1000,10);assert sum(len(set(a)&set(b)) for a,b in zip(ids,gt))==r['total_hits'];checked.add(r['key'])
for manifest in [D/'INPUT_LOCK.json',*[D/f'LABELS_{s}_COMPLETE.json' for s in ['TRAIN','VALIDATION','TEST']]]:
    mm=json.loads(manifest.read_text());hh=mm if manifest.name=='INPUT_LOCK.json' else mm['hashes'];assert all(sha(p)==h for p,h in hh.items())
assert all(sha(p)==h for p,h in OP['sources'].items())
rejected=json.loads((D/'IO_REJECTIONS.json').read_text()) if (D/'IO_REJECTIONS.json').exists() else []
rows+=['','## 核验与边界','',
f"历史原生/MLP/LR前缀及输出匹配；全部9600个query-policy空干预通过，所有选中分支都只执行一次干预且状态匹配。各次在线reader字节=Linux read_bytes=读取次数×4096，无交换；ID重算Recall；重复间计数一致；原始模型、查询及标签哈希不变。在线IO异常归档重试 {len(rejected)} 次，未混入统计。新模型原生导出最大误差低于4e-7。",'',
'监督是相对于三种冻结采集策略的单节点最终命中风险，多个同时剪枝的损失不可相加，也不能保证覆盖新模型产生的所有状态。分数较高表示本次分布下单节点跳过更可能有害，并非全局Recall保证。未训练额外收益头，安全但增加后续读取的节点仍可能被剪；线上完整搜索用于检验这种风险。',
'', '这次标签同时利用完整GT与宽L路径采样，但GT此前已可用，不应把改善都归因于L增大。匹配代理对照隔离了标签改变；宽L采样本身还缺独立消融。离线风险排序提高不等于在线吞吐必然提高。单种子、有限阈值网格、已观察过的留出查询，不能声明统计显著性或普遍最优。','',
'## 文件与复现','', '新风险权重：`models/risk/state.bin`、`models/risk/lr.bin`；Python权重及归一化也保留。匹配代理对照在`models/proxy/`。标签`labels/`保留状态、干预条件、两分支结果、教师分层和权重；原模型未覆盖。','',
'项目根目录运行 `.venv/bin/python experiments/recall_risk_retraining/run.py` 可复用已有通过检查的采集和测试结果。首次构建采集器运行 build.py；协议、输入锁、选参锁和所有逐查询CSV均在本目录。','']
(D/'REPORT.md').write_text('\n'.join(rows));verification=dict(passed=True,only_SIFT1M=True,source_hashes_unchanged=True,all_shams_passed=9600,prefix_state_byte_identical=True,physical_IO_verified=True,no_swap=True,repeated_IDs_and_counts_equal=True,unique_test_runs=len(allruns),validation_configs=len(json.loads((D/'online_validation.json').read_text())),models=4,test_query_overlap_with_train=False,thresholds_locked_before_test=True)
dump(D/'verification.json',verification);dump(D/'COMPLETE.json',dict(complete=True,verification=verification,report_sha256=sha(D/'REPORT.md'),effects=effects));dump(D/'STATUS.json',dict(phase='complete',complete=True));print('\n'.join(rows));print('ALL COMPLETE',flush=True)
