# SIFT100M P0-A：Ours使用mixed-L LR

固定原图R64/Lbuild100/PQ34、K10、L100、Beam4、10,000节点BFS缓存；同一1000条测试查询、4查询线程/MKL单线程。三方法各9次，三种循环顺序各重复3次，计时包括状态提取、推理、I/O和搜索；加载、128条热身、GT与输出不计。

|方法|Recall@10|P95 ms|QPS|直接读请求/Query|实测KiB/Query|展开/Query|
|---|---:|---:|---:|---:|---:|---:|
|Vanilla|96.95%|8.633|501.4|110.744|442.976|116.394|
|Heuristic|96.08%|7.554|574.9|96.151|384.604|101.801|
|Ours|96.68%|8.523|543.9|101.949|407.796|107.599|

相对本轮Vanilla：

|方法|Recall损失 pp|读取量降低|P95降低|QPS提升|
|---|---:|---:|---:|---:|
|Heuristic|0.87|13.18%|12.50%|14.66%|
|Ours|0.27|7.94%|1.27%|8.46%|

## 重复范围与判读

Ours阈值仅用验证集L100候选轨迹按retain-FNR1%标定，缓存命中固定保留；该代理风险不保证最终Recall损失。Heuristic固定rank>72且completed_expansions>100。

- Vanilla：P95范围8.510–9.487ms，QPS范围472.2–510.3。
- Heuristic：P95范围7.410–7.743ms，QPS范围562.2–580.5。
- Ours：P95范围8.410–8.898ms，QPS范围524.3–553.1。

逐查询配对bootstrap5000次见paired_query_bootstrap.csv，仅反映冻结策略下的查询抽样，不含训练/选参不确定性，不构成计时显著性检验。
所有重复ID与离散计数一致；Recall从ID与GT复算；实际reader字节与Linux read_bytes一致。I/O为WSL2客户机块层实测，不是SSD内部NAND计数。
固定L的成本交换不等于同Recall最优配置优势；后者看P0-B。两个Gate版本独立计时，且训练来源不同，不是纯模型实现消融。
本次采用预先冻结的互斥查询划分，有限配置网格属于探索性比较。原始结果见all_runs.json、runs/和summary.csv。

100M实现：原始uint8 L2索引；所有节点ID以uint32/uint64精确存储。q–v额外驻留完整无损压缩向量表（实际字节见COMPRESSION_COMPLETE.json），每次评分解压32向量块，解码开销计入评分与搜索；这与1M完整float表的内存和CPU成本不同。
