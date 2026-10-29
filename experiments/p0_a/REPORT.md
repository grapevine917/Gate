# P0-A：固定搜索参数的节点展开剪枝实验

固定 SIFT1M 原图索引（R64、Lbuild100、PQ34B），K=10，L=100，Beam Width=4，BFS静态缓存10,000节点；同一组1,000条测试查询，4查询线程，每查询MKL单线程。

三方法使用同一搜索实现；Vanilla 不做特征提取或模型推理。Ours 为原始冻结17维MLP（17–128–64–1），不是新训练的 mixed-L/CF/q–v 模型。

本次预先定义默认 Gate 为已有验证集 retain-FNR1% 档位，τ=0.055717818439006805，p(retain)<τ 时 Skip。该档位此前未被指定为通用默认值；retain-FNR不等于最终Recall损失。Heuristic 使用已有同档位 Rank+Progress：rank>72 且 completed_expansions>100。所有方法保护缓存命中节点；Skip发生在磁盘读取和精确评分之前，并消耗本次Beam槽位。没有本次阈值扫描或测试集选参。

每个方法9次完整运行，共27次。顺序为三种循环排列各重复三次，每个位置出现3次；每次配置先热身128条查询。P95为每次1,000条查询P95的跨次中位数；QPS为1000/四线程搜索墙钟时间的跨次中位数。搜索窗口包括特征提取、推理、I/O和邻居处理，排除加载、缓存构建、热身、GT及输出。

| 方法 | Recall@10 | P95 ms | QPS | 直接读请求/Query | 块层读取 KiB/Query | 展开/Query |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 99.67% | 7.551 | 564.6 | 102.405 | 409.620 | 111.841 |
| Heuristic | 99.52% | 6.729 | 627.6 | 91.745 | 366.980 | 101.180 |
| Ours | 99.59% | 7.537 | 652.1 | 83.212 | 332.848 | 92.648 |

相对 Vanilla（Recall为百分点损失，成本为降低比例）：

| 方法 | Recall损失 pp | 读请求减少 | P95降低 | QPS提升 |
|---|---:|---:|---:|---:|
| Heuristic | 0.15 | 10.41% | 10.88% | 11.16% |
| Ours | 0.08 | 18.74% | 0.19% | 15.49% |

Ours在该固定工作点以0.08个百分点Recall损失换取18.74%的客户机块层读取流量减少和15.49%的QPS提升；P95仅下降0.19%，重复范围重叠，应解释为基本持平，不能声称尾延迟明显改善。Heuristic尾延迟下降更明显，但Recall损失为0.15个百分点。

## 物理 I/O 口径

在 LinuxAlignedFileReader 边界累计实际直接读请求数量及各请求长度，并在正式搜索前后取 /proc/self/io 的 read_bytes 差值；O_DIRECT绕过客户机文件页缓存。本索引每个请求4096字节。所有27次运行中，reader请求数与每Query统计之和一致，reader读取字节与Linux块层read_bytes一致：True。不是以“展开数×估算字节数”代替实测。

当前环境为WSL2虚拟块设备，以上是客户机块层真实读取计数；无法观测宿主机缓存命中、NVMe设备指令数或SSD内部NAND操作。重复读同一扇区也计入读取流量，不是唯一扇区数。

## 稳定性与验证

- Vanilla：9次P95范围 7.388–8.051 ms；QPS范围 538.4–580.3。
- Heuristic：9次P95范围 6.682–7.210 ms；QPS范围 600.5–634.7。
- Ours：9次P95范围 7.452–7.841 ms；QPS范围 623.0–671.8。
- Heuristic：按Query配对bootstrap 5,000次，Recall损失95%区间 0.060–0.260 pp；每Query读请求节省95%区间 10.513–10.806。这是冻结策略下的查询抽样区间，不是重复调参置信区间。
- Ours：按Query配对bootstrap 5,000次，Recall损失95%区间 0.030–0.150 pp；每Query读请求节省95%区间 17.955–20.409。这是冻结策略下的查询抽样区间，不是重复调参置信区间。

全部重复运行的结果ID和离散计数完全一致；Recall从返回ID与GT重新计算。Vanilla的完整结果ID、Recall和I/O与既有L100基准一致；原始Gate设τ=0后完整ID、hits、reads和expansions均与Vanilla逐Query相同。输入模型、索引、查询和源代码哈希均核验未变。

## 解释边界

该实验只回答固定参数引入剪枝后的Recall与系统收益交换；不证明相同Recall下优于调小L的Vanilla，后者应由联合参数扫描主实验回答。当前Query集曾用于探索性评估，未宣称全新确认性测试。

重跑：在项目根目录执行 `.venv/bin/python experiments/p0_a/prepare.py`、`.venv/bin/python experiments/p0_a/run.py`、`.venv/bin/python experiments/p0_a/verify.py`、`.venv/bin/python experiments/p0_a/report.py`。已有完整运行会复用；重新计时需先将runs目录及运行日志归档。

原始结果：summary.csv、summary.json、all_runs.json、runs/*.csv、runs/*.meta.json、runs/*.ids；协议和输入锁见PROTOCOL.json及INPUT_LOCK.json。
