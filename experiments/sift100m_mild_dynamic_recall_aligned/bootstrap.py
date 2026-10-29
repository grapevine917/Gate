"""Prepare isolated L100 artifacts; never overwrite the original SIFT100M study."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,struct,hashlib,shutil,subprocess
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
D=Path(__file__).resolve().parent;R=D.parent/'sift100m_p0';S=D.parent/'sift1m_mild_dynamic_recall_aligned';G=D.parent/'sift100m_progress_gate'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
M=D/'models_train100';M.mkdir(exist_ok=True);(D/'native').mkdir(exist_ok=True)
if not (D/'TRAIN_COMPLETE.json').exists():
    a=np.fromfile(R/'data/l100.prune.bin','<f4').reshape(-1,20);a=a[a[:,13]==0];x=a[:,3:].copy();y=a[:,2].copy()
    assert x.shape[1]==17 and set(x[:,15])=={100} and np.isfinite(x).all()
    mu=x.mean(0,dtype=np.float64).astype('f4');sd=x.std(0,dtype=np.float64).astype('f4');sd[sd<1e-8]=1
    lr=LogisticRegression(C=1,max_iter=1500,tol=1e-5,random_state=20261030).fit((x-mu)/sd,y)
    with (M/'lr.bin').open('wb') as f:
        f.write(b'LRGATE01');f.write(struct.pack('<I',0))
        for v in [mu,sd,-lr.coef_[0],-lr.intercept_]:np.asarray(v,'<f4').tofile(f)
    joblib.dump(dict(model=lr,mean=mu,scale=sd),M/'lr.joblib')
    shutil.copy2(R/'models/l100.bin',M/'state.bin');shutil.copy2(R/'models/l100.pt',M/'state.pt')
    dump(D/'TRAIN_COMPLETE.json',dict(complete=True,training_L=100,samples=len(x),train_queries=1200,MLP='reuse original pure L100 model',LR='fresh pure L100 fit on original L100 cache-miss train rows, same 17 features',training_source_sha256=sha(R/'data/l100.prune.bin'),MLP_sha256=sha(M/'state.bin'),LR_sha256=sha(M/'lr.bin')))
    print('PURE L100 LR TRAINED; L100 MLP REUSED',flush=True)
if not (D/'thresholds_L100.json').exists():
    out=D/'validation_lr.scores'
    subprocess.run([str(R/'native/score'),str(M/'lr.bin'),str(R/'data/vectors.cvec'),str(R/'prediction/validation.bin'),str(out),str(D/'validation_lr.bench.json')],check=True)
    a=np.fromfile(R/'data/validation.prune.bin','<f4').reshape(-1,20);miss=a[:,13]==0;s=np.fromfile(out,'<f4');assert len(s)==int(miss.sum()) and np.isfinite(s).all()
    full=np.ones(len(a),'f4');full[miss]=s;positive=np.sort(full[a[:,2]==1]);points=[]
    for risk in [0,.001,.0025,.005,.01,.02,.05,.1]:
        allowed=int(np.floor(risk*len(positive)));tau=0. if risk==0 else float(positive[allowed]);false=int(np.sum((a[:,2]==1)&(full<tau)));assert false<=allowed
        points.append(dict(risk=risk,tau=tau,false_skips=false,retain_count=len(positive)))
    # Native-vs-Python export check on all held-out validation miss candidates.
    v=joblib.load(M/'lr.joblib');expected=v['model'].predict_proba((a[miss,3:]-v['mean'])/v['scale'])[:,1];error=float(np.max(np.abs(s-expected)));assert error<2e-5
    dump(D/'LR_NATIVE_VERIFICATION.json',dict(max_error=error,samples=len(s)))
    old=json.loads((R/'threshold_calibration.json').read_text());dump(D/'thresholds_L100.json',{'100':dict(state=old['l100'],lr=points)})
# One resident native process can execute mixed method batches. No method shares
# query search state; cache is read-only and each config gets the same warmup.
src=(G/'search.cpp').read_text()
start=src.index(' std::ifstream probe(argv[6])');end=src.index(' std::ifstream configs(argv[6])',start)
src=src[:start]+''' std::unique_ptr<diskann::OnlineGate> mlp=std::make_unique<diskann::OnlineGate>(std::string(argv[4])+"/state.bin","",0);
 std::unique_ptr<diskann::OnlineGate> lr=std::make_unique<diskann::OnlineGate>(std::string(argv[4])+"/lr.bin","",0);
'''+src[end:]
src=src.replace('float tau,et_tau;','float tau,et_tau,late_tau;').replace('>>rank>>expansions){','>>rank>>expansions>>late_tau){')
src=src.replace('if(kind!=first_kind)throw std::runtime_error("one family per process");auto*gate=owned.get();if(gate)gate->threshold=tau;if(et)et->threshold=et_tau;', 'auto*gate=(kind==2 || kind==4)?mlp.get():(kind==3 || kind==5)?lr.get():nullptr;if(gate)gate->threshold=tau;')
src=src.replace('progress{rank,expansions,et_tau}','progress{rank,expansions,late_tau}').replace('(kind==2 && rank>=0 && expansions>=0)','((kind==4 || kind==5) && rank>=0 && expansions>=0)').replace('online_termination=et.get()','online_termination=nullptr')
assert 'first_kind' not in src and 'et.get()' not in src
(D/'search.cpp').write_text(src)
commands=json.loads((R/'build.commands.json').read_text());cmd=next(c for c in commands if str(R/'search.cpp') in c)
cmd=[x.replace(str(R/'native'),str(G/'native')).replace(str(R/'search.cpp'),str(D/'search.cpp')) for x in cmd]
cmd[cmd.index('-o')+1]=str(D/'native/search');dump(D/'build_command.json',cmd)
with (D/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
for name in ['common.py','measurement.py','prepare_validate.py','evaluate_test.py','report.py']:shutil.copy2(S/name,D/name)
s=(D/'common.py').read_text().replace("A=D.parent/'sift1m_p0a_L100_train_compare';W=D.parent/'sift1m_wide_p0'","A=D;W=D.parent/'sift100m_p0'")
(D/'common.py').write_text(s)
s=(D/'measurement.py').read_text().replace('valid=ids<1000000','valid=ids<100000000')
s=s.replace("D.parent/'sift1m/index',W/'data'/f'{split}.fbin'","W/'index/ann',W/'data'/f'{split}.u8bin'")
s=s.replace("ROOT/'data/sift/sift_base.fbin'","W/'data/vectors.cvec'")
# Ordered tests are a single mixed-method resident process, preserving exact
# cyclic interleaving while avoiding repeated 100M index loads.
s=s.replace('if ordered:batches=[[c] for c in todo]','if ordered:batches=[todo] if todo else []')
(D/'measurement.py').write_text(s)
s=(D/'prepare_validate.py').read_text().replace("dataset='SIFT1M',base_vectors=1000000","dataset='SIFT100M',base_vectors=100000000")
s=s.replace('retrained=False','retrained=dict(MLP=False,LR=True)')
s=s.replace("W/'data/validation.fbin'","W/'data/validation.u8bin'").replace("W/'data/test.fbin'","W/'data/test.u8bin'")
s=s.replace("for key,old in [('DiskANN_L100','DiskANN'),('Heuristic_r72_e100','Heuristic')]:assert V[key]['ids_sha256']==sha(A/'runs/validation'/f'{old}_r0.ids')", "for key,old in [('DiskANN_L100','DiskANN_L100_p0'),('Heuristic_r72_e100','Heuristic_L100_p3')]:assert V[key]['ids_sha256']==sha(W/'runs/validation'/f'{old}_r0.ids')")
# SIFT100M L100 has ~97% Recall, so use native validation baseline as the
# reference point rather than importing SIFT1M's unreachable 99.60% target.
s=s.replace("aligned=choose_alignment(validation,V,9960)","nominal=int(round((V['MLP_static_b1']['total_hits']+V['LR_static_b1']['total_hits'])/2));dump(D/'ALIGNMENT_TARGET_LOCK.json',dict(nominal_hits=nominal,source='mean static MLP/LR validation hits, before test'));aligned=choose_alignment(validation,V,nominal)")
s=s.replace('nearest99.60%','nearest the validation static-reference Recall')
s=s.replace("controls='Fresh contemporaneous shared control pool; identical config/repeat measured once across both studies'", "controls='Fresh shared controls; resident mixed-method process with per-config128-query warmup; identical config/repeat measured once across studies'")
(D/'prepare_validate.py').write_text(s)
s=(D/'evaluate_test.py').read_text().replace('choose_alignment(initial,V,9960)',"choose_alignment(initial,V,json.loads((D/'ALIGNMENT_TARGET_LOCK.json').read_text())['nominal_hits'])")
s=s.replace('models_not_retrained=True','MLP_reused_LR_retrained_pure100=True')
(D/'evaluate_test.py').write_text(s)
s=(D/'report.py').read_text().replace('SIFT1M','SIFT100M').replace('sift1m_mild_dynamic_recall_aligned','sift100m_mild_dynamic_recall_aligned')
s=s.replace('模型复用已保存的纯 L=100 MLP/LR，没有重新训练。','MLP 复用已有 SIFT100M 纯 L=100 权重；LR 使用同一批 SIFT100M L=100 训练样本补训，未使用旧的混合 L LR。')
s=s.replace('最接近 99.60% 的共同整数命中数','最接近验证集静态 MLP/LR 平均 Recall 的共同整数命中数')
(D/'report.py').write_text(s)
dump(D/'BOOTSTRAP_COMPLETE.json',dict(complete=True,training=json.loads((D/'TRAIN_COMPLETE.json').read_text()),binary_sha256=sha(D/'native/search'),parent_1M_protocol=sha(S/'PROTOCOL.json'),native_core=sha(G/'native/libdiskann.a')))
print('BOOTSTRAP COMPLETE',flush=True)
