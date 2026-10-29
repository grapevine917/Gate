"""Generate scale-specific P0 training/evaluation from the completed DEEP protocol."""
from pathlib import Path
R=Path(__file__).resolve().parent;D=R.parent/'deep_p0'
train=(D/'train.py').read_text()
train=train.replace("p=np.fromfile(R/'data/mixed.prune.bin','<f4').reshape(-1,20);p=p[p[:,13]==0];", "p=np.fromfile(R/'data/mixed.prune.bin','<f4').reshape(-1,20);nodes=np.fromfile(R/'data/mixed.prune_nodes.u32','<u4');assert len(nodes)==len(p);mask=p[:,13]==0;nodes=nodes[mask];p=p[mask];")
train=train.replace("b'D3GATE01' if a.shape[1]==401", "b'S3GATE01' if a.shape[1]==529")
train=train.replace("LG/'data/deep1M_base.fbin',dtype='<f4'", "LG/'data/sift100m/base.100M.u8bin',dtype='u1'")
train=train.replace('shape=(1000000,96)','shape=(100000000,128)').replace('base[p[:,1].astype(int)]','base[nodes],dtype=\'f4\'')
train=train.replace('(len(x),401)','(len(x),529)').replace('joint[:,:96]=q;joint[:,96:192]=v;joint[:,192:288]=np.abs(q-v);joint[:,288:384]=q*v;joint[:,384:]=x','joint[:,:128]=q;joint[:,128:256]=v;joint[:,256:384]=np.abs(q-v);joint[:,384:512]=q*v;joint[:,512:]=x')
old="f.write(struct.pack('<I',len(ix)));np.concatenate([q[ix],p[ix,1:2],x[ix],expected],1).astype('<f4').tofile(f)"
new="f.write(struct.pack('<I',len(ix)));check=np.empty((len(ix),150),'<f4');check[:,:128]=q[ix];check.view('<u4')[:,128]=nodes[ix];check[:,129:146]=x[ix];check[:,146:]=expected;check.tofile(f)"
assert old in train;train=train.replace(old,new)
(R/'train.py').write_text(train)
ev=(D/'evaluate.py').read_text().replace('DEEP1M','SIFT100M').replace('DEEP','SIFT100M').replace("D=R.parent/'deep1m'","D=R")
ev=ev.replace("str(ROOT/'data/deep1M_base.fbin')","str(R/'data/vectors.cvec')").replace("str(D/'index')","str(R/'index/ann')")
ev=ev.replace("str(R/'data'/f'{split}.fbin')","str(R/'data'/f'{split}.u8bin')")
old="raw=np.empty((len(miss),115),'<f4');raw[:,:96]=q[miss[:,0].astype(int)];raw[:,96]=miss[:,1];raw[:,97:114]=miss[:,3:];raw[:,114]=miss[:,2];raw.tofile(scores_input)"
new="nodes=np.fromfile(R/'data'/f'{split}.prune_nodes.u32','<u4');assert len(nodes)==len(a);raw=np.empty((len(miss),147),'<f4');raw[:,:128]=q[miss[:,0].astype(int)];raw.view('<u4')[:,128]=nodes[mask];raw[:,129:146]=miss[:,3:];raw[:,146]=miss[:,2];raw.tofile(scores_input)"
assert old in ev;ev=ev.replace(old,new).replace('range(10,301)','range(10,max(LS)+1)').replace('ids<1000000','ids<100000000')
ev=ev.replace("Full SIFT100M validation L100 baseline candidates; cache hits protected; same retain-FNR budgets for all Gates; differs from SIFT P0-C training-quantile grid. Never calibrate using test.","Full SIFT100M validation L100 baseline candidates; cache hits protected; matched retain-FNR budgets. Never calibrate using test.")
(R/'evaluate.py').write_text(ev)
an=(D/'analyze.py').read_text().replace('DEEP1M','SIFT100M').replace('DEEP','SIFT100M').replace('401D','529D').replace('401–128','529–128').replace('[q96,v96','[q128,v128').replace('1M×96','100M×128').replace('96维','128维').replace('完整向量表384MB','无损压缩uint8向量表（约9.5GB，含解码推理成本）').replace('experiments/deep_p0/run.sh','experiments/sift100m_p0/run.sh')
an=an.replace("该SIFT100M查询集在历史实验中被观察，属于探索性实验。","本次采用预先冻结的互斥查询划分，有限配置网格属于探索性比较。")
an=an.replace('SIFT100M查询曾在历史实验中观察，属于探索性实验。','本次查询划分预先冻结。').replace('SIFT100M基准查询曾在历史实验中观察；本次属于探索性实验。','本次为预先冻结查询划分上的有限网格实验。')
an=an.replace('q–v模型后期明显过拟合，按内部独立查询选择早期最佳点，而非最终160轮权重。','q–v模型按内部独立查询的最小损失选择轮次，再使用全部训练查询重拟合。')
an=an.replace('没有复用SIFT模型权重','没有复用SIFT1M或DEEP模型权重')
an=an.replace('286550',"{train['prune_samples']}")
# Dynamic sample count appears inside ordinary string literals in the old report.
an=an.replace("'四种Gate使用相同{train", "f'四种Gate使用相同{train").replace("'四种Gate共享{train", "f'四种Gate共享{train")
# Single quotes inside those f-strings would clash with the outer literal.
an=an.replace("{train['prune_samples']}",'{train["prune_samples"]}')
# Include the 100M implementation/resource difference in every final report.
an=an.replace("(root/'REPORT.md').write_text", "text+=['','100M实现：原始uint8 L2索引；所有节点ID以uint32/uint64精确存储。q–v额外驻留完整无损压缩向量表（实际字节见COMPRESSION_COMPLETE.json），每次评分解压32向量块，解码开销计入评分与搜索；这与1M完整float表的内存和CPU成本不同。']\n (root/'REPORT.md').write_text")
an=an.replace("(R/'REPORT.md').write_text", "root_text+=['','100M资源差异：采用uint8原始向量索引；q–v使用额外驻内存的无损压缩原始向量表，解码计入推理和搜索。PQ34本体约3.4GB；向量表实际大小见COMPRESSION_COMPLETE.json。节点ID始终以uint32/uint64保存，未经过float32转换。']\n(R/'REPORT.md').write_text")
(R/'analyze.py').write_text(an)
print('PIPELINE GENERATED')
