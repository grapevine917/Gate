#!/usr/bin/env python3
"""Per-L Top3 pruning / All17 ET retraining, native calibration, fresh live controls."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
import argparse,copy,json,time,hashlib
from pathlib import Path
import joblib,numpy as np
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from run_online_m3 import E,ROOT,execute,mat,save_mat
import build_training_data as td
from train_model1 import predict
from train_state_baselines import ranking,threshold_from_validation
from analyze_lid_buckets import dump,table as write_table
from calibrate_et_pruning import env_for
import plot_online_curves as fig
def table(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    write_table(path,[{k:row.get(k) for k in fields} for row in rows])

OUT=E/'retrained_high_l'
RISKS_ET=[0,.001,.0025,.005,.01,.02,.05,.1]
RISKS_P=[.001,.0025,.005,.01,.02]
SPLITS={s:np.load(E/f'training_data_l100/{s}/query_ids.npy') for s in ('train','validation','test')}

def run(l,label,q,gt,env,folder):
    return execute(label,q,gt,env,search_l=l,output_dir=folder)

def result(folder,label,l,dtype='<u4'):
    suffix='idx_uint32' if dtype=='<u4' else 'dists_float'
    return mat(folder/f'{label}_{l}_{suffix}.bin',dtype)

def prepare(l):
    folder=OUT/f'L{l}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'DATA_COMPLETE.json').exists():return
    query_path=ROOT/'data/sift/sift_query.fbin';gt=ROOT/'data/sift/gt100.bin'
    boundary=folder/'baseline_boundaries.f32'
    trace=E/'v2/trace_150.trace' if l==150 else folder/f'trace_{l}.trace'
    if not (folder/f'collect_baseline_{l}_idx_uint32.bin').exists():
        env={'DISKANN_ET_TRACE':str(boundary)}
        if l==200:env.update(DISKANN_TRACE_VERSION='2',DISKANN_TRACE_PREFIX=str(folder/'trace'))
        print('COLLECT',l,flush=True);run(l,'collect_baseline',query_path,gt,env,folder)
    baseline=result(folder,'collect_baseline',l)
    if l==150:
        np.testing.assert_array_equal(baseline,mat(E/'v2/results_150_idx_uint32.bin','<u4'))
        np.testing.assert_array_equal(result(folder,'collect_baseline',l,'<f4'),mat(E/'v2/results_150_dists_float.bin','<f4'))
    testref=E/'varying_l_comparison'/f'r0_L{l}_baseline_{l}_idx_uint32.bin'
    np.testing.assert_array_equal(baseline[SPLITS['test']],mat(testref,'<u4'))
    native=np.fromfile(boundary,dtype='<f4').reshape(-1,21)
    native=native[np.lexsort((native[:,10],native[:,0]))]
    assert len(np.unique(native[:,0]))==10000 and np.isfinite(native).all()
    byq={};qs=native[:,0].astype('int64');starts=np.flatnonzero(np.r_[True,qs[1:]!=qs[:-1]])
    for start,end in zip(starts,np.r_[starts[1:],len(qs)]):byq[int(qs[start])]=native[start:end]
    split_for={int(q):s for s,ids in SPLITS.items() for q in ids}
    buffers={s:dict(raw=[],y=[],qids=[],etraw=[],ety=[],etqids=[]) for s in SPLITS}
    seen=set();td.L=l;mask=folder/f'oracle_{l}.mask';counts={}
    with mask.open('w') as mf:
        for q,search_l,visited,ex,ed in td.blocks(trace):
            assert search_l==l and q not in seen;seen.add(q)
            pre,_,v=td.replay(q,ex,ed);assert v==visited and np.isfinite(pre).all()
            in_final=np.isin(ex['node'],baseline[q]);assert in_final.sum()==10
            safe=(ex['admitted']==0)&~in_final;y=(~safe).astype('uint8')
            for node in ex['node'][safe]:mf.write(f'{q} {int(node)}\n')
            b=byq[q];beam=ed[::64]['beam_step'];lastbeam=int(beam[in_final].max())
            expected=(b[:,10]>lastbeam).astype('uint8');np.testing.assert_array_equal(b[:,19],expected)
            node_to_row={int(node):i for i,node in enumerate(ex['node'])}
            ix=np.array([node_to_row[int(node)] for node in b[:,1]])
            np.testing.assert_array_equal(b[:,2:19],pre[ix])
            s=split_for[q];buf=buffers[s]
            buf['raw'].append(pre);buf['y'].append(y);buf['qids'].append(np.full(len(y),q,dtype='uint32'))
            buf['etraw'].append(b[:,2:19]);buf['ety'].append(expected);buf['etqids'].append(np.full(len(b),q,dtype='uint32'))
            if len(seen)%1000==0:print('REPLAY',l,len(seen),flush=True)
    assert seen==set(range(10000))
    for s,buf in buffers.items():
        dest=folder/'data'/s;dest.mkdir(parents=True,exist_ok=True)
        for name,parts in buf.items():
            array=np.concatenate(parts);qname='etqids' if name.startswith('et') else 'qids'
            order=np.argsort(np.concatenate(buf[qname]),kind='stable');array=array[order]
            np.save(dest/f'{name}.npy',array)
        np.save(dest/'query_ids.npy',SPLITS[s])
        counts[s]=dict(queries=len(SPLITS[s]),pruning_samples=sum(map(len,buf['y'])),et_samples=sum(map(len,buf['ety'])))
    del buffers
    print('ORACLE VERIFY',l,flush=True)
    run(l,'full_skip_oracle',query_path,gt,{'DISKANN_SKIP_PREFIX':str(folder/'oracle'),'DISKANN_SKIP_ENTIRE_NODE':'1'},folder)
    for dtype in ('<u4','<f4'):np.testing.assert_array_equal(result(folder,'full_skip_oracle',l,dtype),result(folder,'collect_baseline',l,dtype))
    dump(folder/'data_manifest.json',dict(L=l,splits=counts,columns=td.PRE,pruning_label='retain=1 unless zero admissions AND node outside final top10; full-mask ID/distance parity verified',et_label='safe=1 iff all final baseline top10 already scored; independently verified using last final beam',full_queue_replay=True,original_query_split=True))
    dump(folder/'DATA_COMPLETE.json',dict(complete=True,L=l,queries=10000,oracle_parity=True,state_replay_and_et_labels_verified=True))

def train(l,kind):
    folder=OUT/f'L{l}';dest=folder/kind;dest.mkdir(exist_ok=True)
    if (dest/'TRAINING_COMPLETE.json').exists():return
    et=kind=='ET';indices=list(range(17)) if et else [3,7,6]
    prefix='et' if et else ''
    raw={s:np.load(folder/f'data/{s}/{prefix}raw.npy') for s in SPLITS}
    ys={s:np.load(folder/f'data/{s}/{prefix}y.npy') for s in SPLITS}
    x={s:np.array(v[:,indices],dtype='float32',order='C') for s,v in raw.items()};del raw
    mean=x['train'].mean(axis=0,dtype='float64').astype('float32')
    scale=np.sqrt(np.mean((x['train'].astype('float64')-mean)**2,axis=0)).astype('float32');scale[scale==0]=1
    for a in x.values():a-=mean;a/=scale
    model=MLPClassifier(hidden_layer_sizes=(128,64),activation='relu',solver='adam',batch_size=2048,learning_rate_init=.001,alpha=.0001,shuffle=True,random_state=42,early_stopping=False)
    best_loss=float('inf');stale=0;history=[];start=time.perf_counter()
    with threadpool_limits(limits=4):
        for epoch in range(1,61):
            model.partial_fit(x['train'],ys['train'],classes=np.array([0,1]))
            val=ranking(ys['validation'],predict(model,x['validation']))
            history.append(dict(epoch=epoch,train_loss=float(model.loss_),validation=val))
            if val['log_loss']<best_loss-.0001:
                best_loss=val['log_loss'];best_epoch=epoch;best=copy.deepcopy(model);stale=0
            else:stale+=1
            dump(dest/'training_history.json',history)
            print('TRAIN',l,kind,epoch,'best',best_epoch,'stale',stale,'val_loss',val['log_loss'],flush=True)
            if stale>=8:break
        model=best;probs={s:predict(model,v) for s,v in x.items()}
    bundle=dict(model=model,mean=mean,scale=scale,feature_indices=indices,feature_names=[td.PRE[i] for i in indices])
    joblib.dump(bundle,dest/'model.joblib')
    with (dest/'model.bin').open('wb') as f:
        f.write(b'ABGATE01');np.array([len(indices)],dtype='<u4').tofile(f);np.array(indices,dtype='<u4').tofile(f)
        for a in (mean,scale):a.astype('<f4').tofile(f)
        for w,b in zip(model.coefs_,model.intercepts_):np.ascontiguousarray(w,dtype='<f4').tofile(f);b.astype('<f4').tofile(f)
    for s,p in probs.items():np.save(dest/f'{s}_p.npy',p)
    dump(dest/'metrics.json',dict(L=l,kind=kind,features=bundle['feature_names'],architecture=[len(indices),128,64,1],fit=dict(best_epoch=best_epoch,epochs=len(history),seconds=time.perf_counter()-start,max_epochs=60,patience=8,min_delta=.0001),ranking={s:ranking(ys[s],p) for s,p in probs.items()},seed=42,normalization='train only',test_used_for_selection=False))
    dump(dest/'TRAINING_COMPLETE.json',dict(complete=True))

def native_env(folder,kind,tau):
    return {'DISKANN_ET_MODEL':str(folder/kind/'model.bin'),'DISKANN_ET_TAU':str(tau)} if kind=='ET' else {'DISKANN_M3_MODEL':str(folder/kind/'model.bin'),'DISKANN_M3_TAU':str(tau)}

def calibrate(l,validation_query=None,validation_gt=None):
    folder=OUT/f'L{l}';dest=folder/'calibration';dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists():return
    ids=SPLITS['validation'];q=Path(validation_query) if validation_query else E/'learned_termination/validation_query.fbin';gt=Path(validation_gt) if validation_gt else E/'learned_termination/validation_gt.bin'
    verification={}
    for kind in ('ET','pruning'):
        trace=dest/f'{kind}_native.f32';env=native_env(folder,kind,1.1 if kind=='ET' else 0)
        env['DISKANN_ET_TRACE' if kind=='ET' else 'DISKANN_M3_AUDIT']=str(trace)
        run(l,'native_'+kind,q,gt,env,dest)
        for dtype in ('<u4','<f4'):np.testing.assert_array_equal(result(dest,'native_'+kind,l,dtype),result(folder,'collect_baseline',l,dtype)[ids])
        a=np.fromfile(trace,dtype='<f4').reshape(-1,21 if kind=='ET' else 19)
        raw=np.load(folder/f'data/validation/{"et" if kind=="ET" else ""}raw.npy');y=np.load(folder/f'data/validation/{"et" if kind=="ET" else ""}y.npy')
        qids=np.load(folder/f'data/validation/{"et" if kind=="ET" else ""}qids.npy')
        python=np.load(folder/kind/'validation_p.npy')
        if kind=='ET':
            original=ids[a[:,0].astype('int64')];order=np.lexsort((a[:,10],original));a=a[order];original=original[order]
            np.testing.assert_array_equal(original,qids);np.testing.assert_array_equal(a[:,2:19],raw);np.testing.assert_array_equal(a[:,19],y)
            p=a[:,-1];np.testing.assert_allclose(p,python,atol=2e-6,rtol=2e-5)
            maxerror=float(abs(p-python).max())
        else:
            # Existing native pruning audit records only local query 0; column 0 is node ID.
            order=np.lexsort((a[:,10],a[:,9]));a=a[order]
            first=raw[qids==ids[0]];ix=np.lexsort((first[:,9],first[:,8]));first=first[ix]
            np.testing.assert_array_equal(a[:,1:18],first)
            expected=python[qids==ids[0]][ix]
            np.testing.assert_allclose(a[:,-1],expected,atol=2e-6,rtol=2e-5)
            maxerror=float(abs(a[:,-1]-expected).max());p=python
        thresholds={}
        if kind=='ET':
            local=np.searchsorted(ids,original);maximum=np.full(len(ids),-1,dtype='float32');np.maximum.at(maximum,local[y==0],p[y==0]);rank=np.sort(maximum)[::-1]
            for risk in RISKS_ET:
                budget=int(np.floor(risk*len(ids)));tau=max(0,float(np.nextafter(rank[budget],np.float32(np.inf))))
                assert int((maximum>=tau).sum())<=budget
                thresholds[f'unsafe_query_{risk*100:g}pct']=dict(threshold=tau,target_unsafe_stop_query_rate=risk,validation_unsafe_stops=int((maximum>=tau).sum()),selection_split='validation_native')
        else:
            for risk in RISKS_P:thresholds[f'fnr_{risk*100:g}pct']=threshold_from_validation(y,p,risk)
        dump(folder/kind/'thresholds.json',thresholds)
        verification[kind]=dict(native_state_parity=True,disabled_gate_baseline_parity=True,cpp_python_max_probability_error=maxerror,calibration='native float32 all validation boundaries' if kind=='ET' else 'Python validation scores; native first-query audit',validation_only=True)
        print('CALIBRATED',l,kind,flush=True)
    dump(dest/'verification.json',verification);dump(dest/'COMPLETE.json',dict(complete=True))

def evaluate():
    dest=OUT/'online';dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists():return
    configs=[]
    for l in (150,200):
        folder=OUT/f'L{l}'
        configs.append(dict(L=l,method='DiskANN',version='baseline',op='baseline',key=f'L{l}_baseline',env={}))
        for kind,method,oldpath in [('ET','Learned ET',E/'learned_termination/All17'),('pruning','Top3 pruning',E/'feature_ablation/Top3')]:
            for version,path in [('frozen_L100',oldpath),('retrained',folder/kind)]:
                for op,c in json.loads((path/'thresholds.json').read_text()).items():
                    env=native_env(folder,kind,c['threshold']) if version=='retrained' else env_for(None,c['threshold']) if kind=='ET' else env_for(c['threshold'],None)
                    configs.append(dict(L=l,method=method,version=version,op=op,threshold=c['threshold'],key=f'L{l}_{kind}_{version}_{op}',env=env))
    dump(dest/'configs.json',[{k:v for k,v in c.items() if k!='env'} for c in configs])
    q=E/'online_m3/test_query.fbin';gt=E/'online_m3/test_gt.bin';truth=mat(gt,'<u4')[:,:10]
    run(200,'warmup',q,gt,{},dest);rng=np.random.default_rng(42);runs=[];refs={}
    existing=json.loads((dest/'runs.json').read_text()) if (dest/'runs.json').exists() else []
    existing_by_label={r['label']:r for r in existing}
    for repeat in range(3):
        for ix in rng.permutation(len(configs)):
            c=configs[ix];label=f'r{repeat}_{c["key"]}'
            if label in existing_by_label:row=existing_by_label[label]
            else:
                row=run(c['L'],label,q,gt,c['env'],dest);row.update({k:v for k,v in c.items() if k!='env'});row['repeat']=repeat
            ids=result(dest,label,c['L']);dist=result(dest,label,c['L'],'<f4')
            stats=np.genfromtxt(dest/f'{label}.csv',delimiter=',',names=True);assert len(stats)==1000
            np.testing.assert_array_equal(stats['query_row'],np.arange(1000))
            recall=(ids[:,:10,None]==truth[:,None,:]).any(axis=2).mean()*100;assert abs(recall-row['recall_pct'])<1e-8
            if c['method']=='Top3 pruning':
                np.testing.assert_array_equal(stats['expansions'],stats['gate_evaluated']-stats['gate_skipped']);assert not stats['et_evaluated'].any()
            elif c['method']=='Learned ET':assert not stats['gate_evaluated'].any() and np.all(stats['et_stopped']<=1)
            else:assert not stats['et_evaluated'].any() and not stats['gate_evaluated'].any()
            if c['key'] in refs:
                old=refs[c['key']];np.testing.assert_array_equal(ids,old[0]);np.testing.assert_array_equal(dist,old[1])
                for col in ('ssd_reads','expansions'):np.testing.assert_array_equal(stats[col],old[2][col])
            else:refs[c['key']]=(ids,dist,stats)
            if c['version']=='baseline':np.testing.assert_array_equal(ids,result(OUT/f'L{c["L"]}','collect_baseline',c['L'])[SPLITS['test']])
            if c['version']=='frozen_L100':
                oldkey=f'L{c["L"]}_'+('et_' if c['method']=='Learned ET' else 'prune_')+c['op']
                np.testing.assert_array_equal(ids,result(E/'varying_l_comparison','r0_'+oldkey,c['L']))
            runs.append(row);dump(dest/'runs.json',runs)
            print('ONLINE',len(runs),'/',len(configs)*3,label,'Recall',recall,'Reads',row['ssd_reads_per_query'],flush=True)
    summary=[]
    for c in configs:
        trials=[r for r in runs if r['key']==c['key']];assert len(trials)==3
        r={k:v for k,v in c.items() if k!='env'}
        for k in ('recall_pct','ssd_reads_per_query','expansions_per_query','latency_us','qps','latency_p50_us','latency_p95_us','gate_evaluations_per_query','skips_per_query'):
            r[k]=float(np.median([t[k] for t in trials]))
        r['latency_min_us']=min(t['latency_us'] for t in trials);r['latency_max_us']=max(t['latency_us'] for t in trials);summary.append(r)
    dump(dest/'summary.json',summary);table(dest/'summary.csv',summary)
    dump(dest/'COMPLETE.json',dict(complete=True,configs=len(configs),runs=len(runs),independent_recalls=True,repeat_ids_distances_and_counts_identical=True,frozen_controls_match_prior=True))

def render(path,rows,minimum_recall=None):
    if minimum_recall is not None:rows=[r for r in rows if r["recall_pct"]>=minimum_recall]
    W,H=1840,1070
    s=fig.image_surface(0,W,H) if path.suffix=='.png' else (fig.svg_surface if path.suffix=='.svg' else fig.pdf_surface)(str(path).encode(),W,H)
    c=fig.create(s);fig.rgb(c,1,1,1);fig.paint(c)
    fig.text(c,35,37,'Per-L retraining: ET All17 vs Top3 pruning',26,True)
    if minimum_recall is not None:fig.text(c,1450,37,f'Recall >= {minimum_recall}%',18,True)
    fig.text(c,35,65,'Same 8,000/1,000/1,000 query split | train-only normalization | per-L validation calibration | 3 live trials',14)
    colors={('DiskANN','baseline'):(.18,.25,.32),('Learned ET','frozen_L100'):(.7,.6,.8),('Learned ET','retrained'):(.49,.24,.67),('Top3 pruning','frozen_L100'):(.55,.75,.65),('Top3 pruning','retrained'):(.08,.56,.36)}
    for i,((m,v),col) in enumerate(colors.items()):
        x=45+(i%3)*580;y=100+(i//3)*30;fig.segment(c,x,y,x+30,y,col,3);fig.text(c,x+40,y+5,m+' '+v,14)
    costs=[('ssd_reads_per_query','SSD read requests / query'),('expansions_per_query','Expansions / query'),('latency_us','Latency (ms)')]
    for ri,l in enumerate((150,200)):
        group=[r for r in rows if r['L']==l];lo=np.floor((min(r['recall_pct'] for r in group)-.05)*4)/4
        if minimum_recall is not None:lo=minimum_recall-.03
        for ci,(key,title) in enumerate(costs):
            left,top,pw,ph=85+ci*610,215+ri*370,470,260
            xmax=max(r[key]/(1000 if key=='latency_us' else 1) for r in group)*1.1
            xp=lambda v:left+v/xmax*pw;yp=lambda v:top+ph-(v-lo)/(100-lo)*ph
            fig.text(c,left,top-30,f'L={l} | '+title,17,True);fig.text(c,left,top-9,'Recall@10 (%)',12)
            for v in np.linspace(lo,100,5):
                fig.segment(c,left,yp(v),left+pw,yp(v),(.88,.9,.92));fig.text(c,left-57,yp(v)+4,f'{v:.2f}',12)
            for v in np.linspace(0,xmax,5):
                fig.segment(c,xp(v),top,xp(v),top+ph,(.93,.94,.95));fig.text(c,xp(v)-10,top+ph+24,f'{v:.2f}' if key=='latency_us' else f'{v:.0f}',12)
            for (m,version),col in colors.items():
                pts=sorted([r for r in group if r['method']==m and r['version']==version],key=lambda r:r[key])
                xy=lambda r:(xp(r[key]/(1000 if key=='latency_us' else 1)),yp(r['recall_pct']))
                for a,b in zip(pts,pts[1:]):fig.segment(c,*xy(a),*xy(b),col,1.3 if version=='frozen_L100' else 2.5)
                for r in pts:
                    x,y=xy(r);fig.rgb(c,*col);fig.arc(c,x,y,3 if version=='frozen_L100' else 4,0,2*np.pi);fig.fill(c)
                    if key=='latency_us':fig.segment(c,xp(r['latency_min_us']/1000),y,xp(r['latency_max_us']/1000),y,col)
            fig.text(c,left+70,top+ph+51,title,14)
    fig.text(c,35,1000,'Fresh randomized timing of frozen and retrained models together. SSD reads are index requests; bars: trial min/max.',14)
    fig.text(c,35,1030,'Thresholds selected only on validation; curves connect measured points, without interpolation or test tuning.',14)
    if path.suffix=='.png':assert fig.png(s,str(path).encode())==0
    fig.finish(s);fig.destroy(c);fig.surface_destroy(s)

def report():
    rows=json.loads((OUT/'online/summary.json').read_text())
    for ext in ('png','svg','pdf'):
        render(OUT/f'recall_cost_retrained.{ext}',rows)
        render(OUT/f'recall_cost_retrained_high_recall.{ext}',rows,99.7)
    comparisons=[]
    for l in (150,200):
        for version in ('frozen_L100','retrained'):
            et=[r for r in rows if r['L']==l and r['version']==version and r['method']=='Learned ET']
            pr=[r for r in rows if r['L']==l and r['version']==version and r['method']=='Top3 pruning']
            for cost in ('ssd_reads_per_query','expansions_per_query','latency_us'):
                comparisons.append(dict(L=l,version=version,cost=cost,pruning_points_dominated_by_et=sum(any(e['recall_pct']>=p['recall_pct'] and e[cost]<=p[cost] and (e['recall_pct']>p['recall_pct'] or e[cost]<p[cost]) for e in et) for p in pr),total_pruning_points=len(pr)))
    dump(OUT/'observed_comparisons.json',comparisons)
    fixed=[r for r in rows if r['op'] in ('baseline','unsafe_query_0.5pct','fnr_0.5pct')]
    table(OUT/'fixed_0.5pct_summary.csv',fixed)
    lines=['# L=150/200 重新训练与校准','','沿用原 8000/1000/1000 query 划分，各 L 分别训练 Top3 pruning 与 All17 ET。结构均为 d→128→64→1，seed=42、batch=2048、Adam lr=.001、alpha=.0001；train-only 标准化，validation logloss 早停（最多60 epochs、patience=8、min_delta=.0001）。','','Pruning 标签为基线零 admission 且节点不属于最终 top10 的安全跳过；各 L 全量 oracle mask 实际搜索核验 ID 和距离完全一致。训练状态逐 query 重放原队列验证。ET 使用真实 beam boundary，safe 标签表示全部最终基线 top10 已评分，并用最后 final-top10 beam 独立核验。','','ET 的 8 档 unsafe-query 目标、pruning 的 5 档 retain-FNR 目标分别按对应 L 的 validation 概率重新校准（ET 用全量 native float32，pruning 沿用 Python 概率校准并审计首条 validation query 的原生概率）；C++/Python 概率、原生状态和禁用 gate 时的 baseline parity 均核验。两类风险定义不同，不等价于 Recall 损失保证。','','原 L100 模型与阈值保留，并与新模型交错重新计时；同一1000 test query，L=150/200、K=10、beam=4、threads=4、cache=10000。54配置×3次=162次实搜，median 延迟含推理与状态构造。三次 ID、距离、读取与展开数一致，Recall 独立重算；冻结 controls 复现此前结果。','','![重训比较](recall_cost_retrained.png)','','![高召回细节](recall_cost_retrained_high_recall.png)','','## 固定名义0.5%风险档','','这张表用于查看对应档位变化，ET 与 pruning 风险含义不同，不能当作等 Recall 比较。','','| L | Method | Version | Recall % | Reads | Expansions | Latency ms |','|---|---|---|---:|---:|---:|---:|']
    for r in fixed:lines.append(f'| {r["L"]} | {r["method"]} | {r["version"]} | {r["recall_pct"]:.3f} | {r["ssd_reads_per_query"]:.3f} | {r["expansions_per_query"]:.3f} | {r["latency_us"]/1000:.3f} |')
    lines+=['','## 实测支配关系','','| L | Version | Cost | Pruning被ET支配点数 |','|---|---|---|---:|']
    for r in comparisons:lines.append(f'| {r["L"]} | {r["version"]} | {r["cost"]} | {r["pruning_points_dominated_by_et"]}/5 |')
    lines+=['','这里只统计实测点，不插值、不用test选择部署阈值。小幅延迟差异应结合重复范围解释。新实验同时改变训练与校准，不能单独归因于重新训练；未添加仅重新校准的中间消融。原测试集已用于探索，因此结果仍属探索性验证。','','复现：`.venv/bin/python experiments/sift1m/retrain_high_l.py`']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    exact=[]
    for l in (150,200):
        for version in ('frozen_L100','retrained'):
            et=[r for r in rows if r['L']==l and r['version']==version and r['method']=='Learned ET']
            pr=[r for r in rows if r['L']==l and r['version']==version and r['method']=='Top3 pruning']
            for recall in sorted(set(round(r['recall_pct'],6) for r in et)&set(round(r['recall_pct'],6) for r in pr)):
                a=min((r for r in et if abs(r['recall_pct']-recall)<1e-8),key=lambda r:r['ssd_reads_per_query'])
                b=min((r for r in pr if abs(r['recall_pct']-recall)<1e-8),key=lambda r:r['ssd_reads_per_query'])
                exact.append(dict(L=l,version=version,recall_pct=recall,et_key=a['key'],pruning_key=b['key'],et_reads=a['ssd_reads_per_query'],pruning_reads=b['ssd_reads_per_query'],pruning_extra_reads_pct=100*(b['ssd_reads_per_query']/a['ssd_reads_per_query']-1),et_expansions=a['expansions_per_query'],pruning_expansions=b['expansions_per_query'],et_latency_ms=a['latency_us']/1000,pruning_latency_ms=b['latency_us']/1000))
    if exact:table(OUT/'exact_recall_comparisons.csv',exact)
    lines+=['','## 相同 Recall 的实测比较','','只比较完全相同的整体 Recall；同 Recall 多个配置取该方法最低 Reads 的点，属于离线描述，不用于选择部署阈值。','','| L | Version | Recall % | ET Reads | Pruning Reads | Pruning额外Reads % |','|---|---|---:|---:|---:|---:|']
    for r in exact:lines.append(f'| {r["L"]} | {r["version"]} | {r["recall_pct"]:.2f} | {r["et_reads"]:.3f} | {r["pruning_reads"]:.3f} | {r["pruning_extra_reads_pct"]:.2f} |')
    lines+=['','这一实验比较“重训+对应L重新校准”与“L100冻结模型+冻结阈值”的完整流程，不能拆出训练和校准各自的因果贡献。', '重新校准后，名义风险档的成本/Recall位置明显变化；应按相同Recall和实测曲线比较，而非按同风险百分比直接比较。']
    lines+=['','## 结果解释','','重训并重新校准后，L=200 的5个 pruning 实测点全部被 ET 的实测点在 Recall–Reads、Recall–Expansions 和 Recall–Latency 上支配；冻结模型时，Reads/Expansions 上没有 pruning 点被 ET 支配。上一轮“L增大时pruning相对竞争力增强”的判断仅对冻结L100模型与阈值成立，本补实验不支持将其推广到按L适配的模型。','','L=200 相同Recall=99.89%、99.91%、99.92%时，pruning分别比ET多读取约11.49%、18.06%、5.70%。L=150在相同Recall=99.81%时，pruning多读取约7.96%；但pruning最高达到99.83%，超过本次ET最高99.81%，因此L=150仍有更高召回的独有区间，不能称ET全面支配。','','重新校准使pruning的原5档FNR点整体更保守、更高Recall，尤其L=200；这既是召回恢复，也是成本增加。不能把变贵直接解释为模型变差，或把风险档位置变化归因于训练本身。后续若要拆因果，需要“冻结模型+对应L仅重新校准”的中间对照。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(OUT/'provenance.json',dict(L=[150,200],queries=dict(train=8000,validation=1000,test=1000),original_query_split_sha256={k:hashlib.sha256((E/f'training_data_l100/{k}/query_ids.npy').read_bytes()).hexdigest() for k in SPLITS},model_sha256={f'L{l}_{kind}':hashlib.sha256((OUT/f'L{l}/{kind}/model.bin').read_bytes()).hexdigest() for l in (150,200) for kind in ('ET','pruning')},seed=42,K=10,beam=4,threads=4,node_cache=10000,train_only_normalization=True,thresholds_selected_on_validation_only=True,test_queries_used_previously_for_exploration=True))
    dump(OUT/'COMPLETE.json',dict(complete=True,L=[150,200],models=4,configs=54,runs=162,plots=True))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','prepare','train','calibrate','evaluate','report'],default='all');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.stage in ('all','prepare'):
        for l in (150,200):prepare(l)
    if args.stage in ('all','train'):
        for l in (150,200):
            for kind in ('pruning','ET'):train(l,kind)
    if args.stage in ('all','calibrate'):
        for l in (150,200):calibrate(l)
    if args.stage in ('all','evaluate'):evaluate()
    if args.stage in ('all','report'):report()

if __name__=='__main__':main()
