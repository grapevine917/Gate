"""Measure replacement policies with real evolving histories, frozen offline thresholds."""
from pathlib import Path
import os,json,csv,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parent;LG=ROOT.parents[1];OLD=ROOT.parent/'io_predictability';results=json.loads((ROOT/'results.json').read_text())
def read(p,dt):
 with open(p,'rb') as f:n,d=np.fromfile(f,np.uint32,2);return np.fromfile(f,dt).reshape(int(n),int(d))
def run(ds,split,name,tau,model):
 dst=ROOT/ds/'online'/split/name;dst.parent.mkdir(parents=True,exist_ok=True)
 args=[str(LG/'DiskANN/build/apps/search_disk_index'),'--data_type','float','--dist_fn','l2','--index_path_prefix',str(LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index'),'--query_file',str(ROOT/ds/split/'query.fbin'),'--gt_file',str(ROOT/ds/split/'gt.bin'),'-K','10','-L','100','--result_path',str(dst),'-W','4','-T','4','--num_nodes_to_cache','10000']
 env={k:v for k,v in os.environ.items() if not k.startswith('DISKANN_')};env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DISKANN_QUERY_STATS=str(dst)+'.csv')
 if model:env.update(DISKANN_M3_MODEL=str(model),DISKANN_M3_TAU=str(tau))
 Path(str(dst)+'.command.json').write_text(json.dumps({'args':args,'env':{k:v for k,v in env.items() if k.startswith('DISKANN_')}},indent=2))
 with open(str(dst)+'.log','w') as f:subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 ids=read(str(dst)+'_100_idx_uint32.bin',np.uint32);gt=read(ROOT/ds/split/'gt.bin',np.uint32)[:,:10];hits=np.array([len(set(a)&set(b)) for a,b in zip(ids,gt)])
 rows=list(csv.DictReader(open(str(dst)+'.csv')));assert len(rows)==300
 return {'recall':float(hits.mean()/10),'reads':sum(int(r['ssd_reads']) for r in rows),'hits':hits,'mean_latency_us':float(np.mean([float(r['latency_us']) for r in rows]))}
summary={}
for ds in ['sift','deep']:
 summary[ds]={}
 for split in ['validation','test']:
  base=run(ds,split,'baseline',0,None);entries={}
  tau=json.loads((OLD/ds/'state_lr/thresholds.json').read_text())['unsafe_fpr_0.005']['retain_threshold']
  settings=[('original_17_lr',tau,OLD/ds/'state_lr/model.bin')]
  for name in ['retrained_17_lr','full_44_lr']:
   for key in ['coverage_0.25','critical_budget_0.005']:
    # Training uses strict safe_score > threshold; native uses strict retain < 1-threshold.
    settings.append((name+'__'+key,1-results[ds]['models'][name]['thresholds'][key],ROOT/ds/(name+'.bin')))
  for name,t,model in settings:
   r=run(ds,split,name,t,model);r['recall_loss_pp']=100*(base['recall']-r['recall']);r['net_request_saving']=1-r['reads']/base['reads'];r['degraded_queries']=int((r['hits']<base['hits']).sum());r.pop('hits');entries[name]=r;print(ds,split,name,r,flush=True)
  base.pop('hits');summary[ds][split]={'baseline':base,'policies':entries}
  (ROOT/'online_results.json').write_text(json.dumps(summary,indent=2))
lines=['','## 增强策略替换上线的动态路径测试','','独立运行完整查询，历史随策略实际决定更新；阈值来自上文条件标签的验证集，测试不重选。单次运行延迟仅作诊断，不能宣称稳定加速。下表只展示测试，完整验证及测试见 online_results.json。','','|数据集|策略|测试Recall@10|相对无剪枝损失pp|净请求节省|受损查询/300|','|---|---|---:|---:|---:|---:|']
for ds,r in summary.items():
 for n,v in r['test']['policies'].items():lines.append(f"|{ds}|{n}|{v['recall']:.3%}|{v['recall_loss_pp']:.3f}|{v['net_request_saving']:.2%}|{v['degraded_queries']}|")
with open(ROOT/'REPORT.md','a') as f:f.write('\n'.join(lines)+'\n')

p=ROOT/'REPORT.md'
text=p.read_text().replace('本实验是冻结策略分布上的条件可预测性检验；不是增强策略在线部署的净请求/Recall实测。','上面的分类指标衡量冻结策略分布上的条件可预测性；下面另列增强策略真实在线部署的净请求/Recall实测。')
conclusion='结论：本轮没有证据表明加入历史能跨数据集稳定改善关键候选保护。在相同验证安全剪枝覆盖率下，完整44维LR相比重训17维LR在SIFT更差（6 vs 2），DEEP略好（1 vs 3，但误剪率差的95%区间包含0）；非线性模型也无一致优势。原LR已误剪子组在两数据集各仅2个关键候选，不能据此证明稳定救回能力。增强LR直接替换原策略后，同一25%验证覆盖阈值的测试净请求节省仅SIFT 0.09%、DEEP 1.07%，原LR为24.05%、23.10%；使用更宽松的验证0.5%关键误剪预算时，增强LR的在线Recall损失达到2.63/2.67个百分点。历史特征存在明显策略分布迁移问题，当前结果不能作为联合剪枝安全性的依据。'
text=text.replace('# 剪枝历史可预测性实验\n','# 剪枝历史可预测性实验\n\n'+conclusion+'\n',1)
p.write_text(text)
