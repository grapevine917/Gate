# DEEP1M P0-A：Ours使用L100 MLP

固定原图R64/Lbuild100/PQ34、K10、L100、Beam4、10,000节点BFS缓存；同一1000条测试查询、4查询线程/MKL单线程。三方法各9次，三种循环顺序各重复3次，计时包括状态提取、推理、I/O和搜索；加载、128条热身、GT与输出不计。

|方法|Recall@10|P95 ms|QPS|直接读请求/Query|实测KiB/Query|展开/Query|
|---|---:|---:|---:|---:|---:|---:|
|Vanilla|99.47%|7.579|567.5|102.199|408.796|112.190|
|Heuristic|99.38%|6.625|639.6|91.231|364.924|101.222|
|Ours|99.34%|7.409|684.3|80.953|323.812|90.946|

相对本轮Vanilla：

|方法|Recall损失 pp|读取量降低|P95降低|QPS提升|
|---|---:|---:|---:|---:|
|Heuristic|0.09|10.73%|12.59%|12.69%|
|Ours|0.13|20.79%|2.24%|20.57%|

## 重复范围与判读

Ours阈值仅用验证集L100候选轨迹按retain-FNR1%标定，缓存命中固定保留；该代理风险不保证最终Recall损失。Heuristic固定rank>72且completed_expansions>100。

- Vanilla：P95范围7.376–7.707ms，QPS范围554.4–586.3。
- Heuristic：P95范围6.533–6.696ms，QPS范围633.3–650.4。
- Ours：P95范围7.325–7.653ms，QPS范围658.7–696.3。

逐查询配对bootstrap5000次见paired_query_bootstrap.csv，仅反映冻结策略下的查询抽样，不含训练/选参不确定性，不构成计时显著性检验。
所有重复ID与离散计数一致；Recall从ID与GT复算；实际reader字节与Linux read_bytes一致。I/O为WSL2客户机块层实测，不是SSD内部NAND计数。
固定L的成本交换不等于同Recall最优配置优势；后者看P0-B。两个Gate版本独立计时，且训练来源不同，不是纯模型实现消融。
该DEEP查询集在历史实验中被观察，属于探索性实验。原始结果见all_runs.json、runs/和summary.csv。
