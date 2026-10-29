# DEEP1M P0-A：Ours使用mixed-L LR

固定原图R64/Lbuild100/PQ34、K10、L100、Beam4、10,000节点BFS缓存；同一1000条测试查询、4查询线程/MKL单线程。三方法各9次，三种循环顺序各重复3次，计时包括状态提取、推理、I/O和搜索；加载、128条热身、GT与输出不计。

|方法|Recall@10|P95 ms|QPS|直接读请求/Query|实测KiB/Query|展开/Query|
|---|---:|---:|---:|---:|---:|---:|
|Vanilla|99.47%|7.580|563.4|102.199|408.796|112.190|
|Heuristic|99.38%|6.694|641.0|91.231|364.924|101.222|
|Ours|99.37%|7.320|665.3|84.474|337.896|94.467|

相对本轮Vanilla：

|方法|Recall损失 pp|读取量降低|P95降低|QPS提升|
|---|---:|---:|---:|---:|
|Heuristic|0.09|10.73%|11.68%|13.78%|
|Ours|0.10|17.34%|3.43%|18.10%|

## 重复范围与判读

Ours阈值仅用验证集L100候选轨迹按retain-FNR1%标定，缓存命中固定保留；该代理风险不保证最终Recall损失。Heuristic固定rank>72且completed_expansions>100。

- Vanilla：P95范围7.444–7.882ms，QPS范围556.7–573.3。
- Heuristic：P95范围6.521–6.867ms，QPS范围630.5–651.2。
- Ours：P95范围7.109–7.408ms，QPS范围648.8–690.1。

逐查询配对bootstrap5000次见paired_query_bootstrap.csv，仅反映冻结策略下的查询抽样，不含训练/选参不确定性，不构成计时显著性检验。
所有重复ID与离散计数一致；Recall从ID与GT复算；实际reader字节与Linux read_bytes一致。I/O为WSL2客户机块层实测，不是SSD内部NAND计数。
固定L的成本交换不等于同Recall最优配置优势；后者看P0-B。两个Gate版本独立计时，且训练来源不同，不是纯模型实现消融。
该DEEP查询集在历史实验中被观察，属于探索性实验。原始结果见all_runs.json、runs/和summary.csv。
