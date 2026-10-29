#!/usr/bin/env python3
"""Frozen L100 gates across search L; actual live searches, interleaved timing."""
import hashlib
import json
import numpy as np
from run_online_m3 import E,BIN,execute,mat
from calibrate_et_pruning import env_for,model_hashes
from analyze_lid_buckets import table,dump
import plot_online_curves as fig
OUT=E/'varying_l_comparison'
LS=[50,100,150,200]
COLORS={'DiskANN':(.18,.25,.32),'Learned ET':(.49,.24,.67),'Top3 pruning':(.08,.56,.36)}
COSTS=[('ssd_reads_per_query','SSD read requests / query'),('expansions_per_query','Expansions / query'),('latency_us','Latency (ms)')]

def render(path,rows,overview=False):
    W,H=1810,660 if overview else 1700
    s=fig.image_surface(0,W,H) if path.suffix=='.png' else (fig.svg_surface if path.suffix=='.svg' else fig.pdf_surface)(str(path).encode(),W,H)
    c=fig.create(s);fig.rgb(c,1,1,1);fig.paint(c)
    fig.text(c,35,37,'SIFT1M: varying L, DiskANN vs Learned ET vs Top3 pruning',25,True)
    fig.text(c,35,65,'Same 1,000 test queries | K=10, beam=4, threads=4, cache=10,000 | 3 interleaved repeats, median latency',14)
    for i,(method,col) in enumerate(COLORS.items()):
        fig.segment(c,45+i*450,100,75+i*450,100,col,3);fig.text(c,85+i*450,105,method,15)
    if overview:
        fig.text(c,35,132,'Fixed L100 operating points: ET unsafe-query target 0.5%; pruning retain-FNR target 0.5%',14)
    for ri,l in enumerate([None] if overview else LS):
        group=rows if overview else [r for r in rows if r['L']==l]
        lo=np.floor((min(r['recall_pct'] for r in group)-.1)*2)/2
        for ci,(key,title) in enumerate(COSTS):
            left,top,pw,ph=80+ci*600,190+ri*370,470,250
            vals=[r[key]/(1000 if key=='latency_us' else 1) for r in group]
            xmax=max(vals)*1.12
            xp=lambda v:left+v/xmax*pw
            yp=lambda v:top+ph-(v-lo)/(100-lo)*ph
            fig.text(c,left,top-25,('L sweep' if overview else f'L={l}')+' | '+title,17,True)
            for v in np.linspace(lo,100,5):
                fig.segment(c,left,yp(v),left+pw,yp(v),(.88,.9,.92));fig.text(c,left-55,yp(v)+4,f'{v:.2f}',12)
            for v in np.linspace(0,xmax,5):
                fig.segment(c,xp(v),top,xp(v),top+ph,(.93,.94,.95));fig.text(c,xp(v)-12,top+ph+24,f'{v:.2f}' if key=='latency_us' else f'{v:.0f}',12)
            for method,col in COLORS.items():
                points=sorted([r for r in group if r['method']==method],key=lambda r:r['L'] if overview else r[key])
                xy=lambda r:(xp(r[key]/(1000 if key=='latency_us' else 1)),yp(r['recall_pct']))
                for a,z in zip(points,points[1:]):fig.segment(c,*xy(a),*xy(z),col,2)
                for r in points:
                    x,y=xy(r);fig.rgb(c,*col);fig.arc(c,x,y,4,0,2*np.pi);fig.fill(c)
                    if overview:fig.text(c,x+6,y-7,str(r['L']),11,color=col)
                    if key=='latency_us':
                        fig.segment(c,xp(r['latency_min_us']/1000),y,xp(r['latency_max_us']/1000),y,col,1)
            fig.text(c,left,top-7,'Recall@10 (%)',12)
            fig.text(c,left+90,top+ph+51,title,14)
    fig.text(c,35,H-65,'Models and thresholds frozen from L=100. Nominal risk labels are not recalibrated guarantees at other L.',14)
    fig.text(c,35,H-38,'SSD reads are index requests; latency includes native model inference/state construction. Bars: repeat min/max.',14)
    if path.suffix=='.png':assert fig.png(s,str(path).encode())==0
    fig.finish(s);fig.destroy(c);fig.surface_destroy(s)

def main():
    OUT.mkdir(exist_ok=True);(OUT/'COMPLETE.json').unlink(missing_ok=True)
    configs=[]
    et=json.loads((E/'learned_termination/All17/thresholds.json').read_text())
    pr=json.loads((E/'feature_ablation/Top3/thresholds.json').read_text())
    for l in LS:
        configs.append(dict(L=l,method='DiskANN',key=f'L{l}_baseline',op='baseline',threshold=None,env={}))
        for method,params in [('Learned ET',et),('Top3 pruning',pr)]:
            for op,v in params.items():
                tau=v['threshold'];configs.append(dict(L=l,method=method,key=f'L{l}_'+('et_' if method=='Learned ET' else 'prune_')+op,op=op,threshold=tau,env=env_for(None,tau) if method=='Learned ET' else env_for(tau,None)))
    dump(OUT/'configs.json',[{k:v for k,v in r.items() if k!='env'} for r in configs])
    q=E/'online_m3/test_query.fbin';gt=E/'online_m3/test_gt.bin';truth=mat(gt,'<u4')[:,:10]
    execute('warmup',q,gt,{},search_l=200,output_dir=OUT)
    rng=np.random.default_rng(42);runs=[];refs={}
    for repeat in range(3):
        for i in rng.permutation(len(configs)):
            cfg=configs[i];label=f'r{repeat}_{cfg["key"]}'
            row=execute(label,q,gt,cfg['env'],search_l=cfg['L'],output_dir=OUT)
            row.update({k:v for k,v in cfg.items() if k!='env'});row['repeat']=repeat
            ids=mat(OUT/f'{label}_{cfg["L"]}_idx_uint32.bin','<u4')
            stats=np.genfromtxt(OUT/f'{label}.csv',delimiter=',',names=True)
            assert len(stats)==1000
            np.testing.assert_array_equal(stats['query_row'],np.arange(1000))
            recall=(ids[:,:10,None]==truth[:,None,:]).any(axis=2).mean()*100
            assert abs(recall-row['recall_pct'])<1e-8
            if cfg['method']=='Top3 pruning':
                np.testing.assert_array_equal(stats['expansions'],stats['gate_evaluated']-stats['gate_skipped']);assert not stats['et_evaluated'].any()
            elif cfg['method']=='Learned ET':assert not stats['gate_evaluated'].any() and np.all(stats['et_stopped']<=1)
            else:assert not stats['gate_evaluated'].any() and not stats['et_evaluated'].any()
            if cfg['key'] in refs:
                ref=refs[cfg['key']];np.testing.assert_array_equal(ids,ref[0])
                for col in ('ssd_reads','expansions'):np.testing.assert_array_equal(stats[col],ref[1][col])
            else:refs[cfg['key']]=(ids.copy(),stats.copy())
            runs.append(row);dump(OUT/'runs.json',runs)
            print(f'{len(runs)}/168 {label}: recall={recall:.3f}, reads={row["ssd_reads_per_query"]:.3f}, latency={row["latency_us"]/1000:.3f}ms',flush=True)
    rows=[]
    for cfg in configs:
        trials=[r for r in runs if r['key']==cfg['key']];assert len(trials)==3
        r={k:v for k,v in cfg.items() if k!='env'}
        for k in ('recall_pct','ssd_reads_per_query','expansions_per_query','latency_us','qps','latency_p50_us','latency_p95_us','gate_evaluations_per_query','skips_per_query'):
            r[k]=float(np.median([t[k] for t in trials]))
        r['latency_min_us']=min(t['latency_us'] for t in trials);r['latency_max_us']=max(t['latency_us'] for t in trials);rows.append(r)
    dump(OUT/'summary.json',rows);table(OUT/'summary.csv',rows)
    overview=[r for r in rows if r['op'] in ('baseline','unsafe_query_0.5pct','fnr_0.5pct')]
    table(OUT/'fixed_threshold_summary.csv',overview)
    for ext in ('png','svg','pdf'):
        render(OUT/f'recall_cost_by_l.{ext}',rows);render(OUT/f'fixed_threshold_l_sweep.{ext}',overview,True)
    dump(OUT/'provenance.json',dict(L=LS,queries=1000,K=10,beam=4,threads=4,cache=10000,repeats=3,seed=42,models=model_hashes(),binary_sha256=hashlib.sha256(BIN.read_bytes()).hexdigest(),models_and_thresholds='frozen from L100; no retraining or per-L recalibration',repeat_ids_and_counts_identical=True,recall_independently_verified=True))
    lines=['# 不同 L 的三方法比较','','全部实际 C++ 搜索，同一批 1000 test query，K=10、beam=4、threads=4、cache=10000。56 个配置随机交错重复三次，共 168 次；延迟取 median，误差线为 min/max。包含真实模型推理与状态构造。','','DiskANN 每个 L 一个点；Learned ET 使用 All17 的 8 档阈值，Top3 pruning 使用 5 档阈值。模型和阈值固定于原 L=100，其他 L 属于泛化实验；名义 unsafe-query/FNR 不代表其他 L 的实际风险保证。','','![每个L阈值曲线](recall_cost_by_l.png)','','![固定阈值跨L曲线](fixed_threshold_l_sweep.png)','','第二张图固定 ET unsafe-query target=0.5%、pruning retain-FNR target=0.5%，点标注 L；两种风险定义不同，不声称风险相等，只展示固定配置跨 L 的行为。','','| L | Method | Recall@10 % | SSD Reads/query | Expansions/query | Latency ms |','|---|---|---:|---:|---:|---:|']
    for r in overview:lines.append(f'| {r["L"]} | {r["method"]} | {r["recall_pct"]:.3f} | {r["ssd_reads_per_query"]:.3f} | {r["expansions_per_query"]:.3f} | {r["latency_us"]/1000:.3f} |')
    lines+=['','逐query 独立重算 Recall，三次重复的 IDs、读取与展开计数一致；pruning gate 计数恒等式通过。原始数据与原实验保留。SSD reads 为索引请求数，Latency 为当前主机预热状态。','','复现：`.venv/bin/python experiments/sift1m/run_l_comparison.py`']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(OUT/'COMPLETE.json',dict(complete=True,configs=56,runs=168,plots=True))

if __name__=='__main__':main()
