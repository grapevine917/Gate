# E3：离线复用上界与策略对比

## 结论与主工作点

主实验Hybrid（R4）测试Recall@10=99.620%，相对无复用MLP变化-0.020个百分点，ModelSaving=0.558%，ReuseRate=0.557%。测试是否满足最多0.2pp附加损失：是。

冻结原始路径上，任意同节点历史Skip的可用率上限为1.786%；免费模型决策Oracle允许安全复用当前同为Skip的比例为0.459%。质量Oracle B在已评估的31种完整轨迹菜单内、最多损失20个Top10命中的约束下，ModelSaving=1.509%，附加Recall变化-0.200pp。

这里的上界有明确范围：静态上界限制在原始轨迹；Oracle A限制为历史Skip与当前Skip一致；Oracle B是已执行轨迹菜单内的精确最优，不是任意逐节点组合的全局最优。不能把这些数字冒充部署Query Cache的实际收益。

## 固定协议

SIFT1M，既有DiskANN图，L=100、K=10、beam=4、4线程、BFS静态缓存10000；现有state-only MLP阈值0.024352464824914932，历史1000条训练Query、验证1000条、测试1000条，历史与验证/测试不重叠。Query与状态界限只来自训练记录，标准化沿用E2训练统计。测试集此前已在E1/E2观察，不是首次使用的新测试集。

主Hybrid使用最近3条历史Query。所有31个候选的定义及Oracle菜单在测试前固定；只用验证真实图搜索选择可部署策略参数，以最多0.2pp附加Recall损失为约束最大化ModelSaving，不按测试结果调参。R1/R2固定规则即便超出预算也完整报告。

选中Hybrid配置：{"key": "R4_q2_s3", "kind": 4, "qcut": 370.59762491429177, "scut": 2.8942509681998327, "votes": 2, "oracle": 0}。

策略定义：
- R0：无复用，始终评分所有cache miss候选。
- R1：任一历史MLP Skip即对节点形成全局标记；复用这些Skip。
- R2：仅最近一条历史Query的节点Skip记录。
- R3：所有同节点历史Keep/Skip记录中的状态最近邻；最近标签为Skip且通过状态阈值才复用。
- R4：最近3条Query中先通过Query距离限制，同节点记录再选状态最近邻，需标签Skip且通过状态距离限制。
- R5：最近3条中通过Query限制的历史记录，至少2或3个可用Query，并且所有可用同节点记录均为Skip；缺失不投票。

R1不等于此前按admitted=0且不在最终Top10构造的全局一次标记。后者作为Legacy Global Mark额外重新测量，保持旧位图原定义。

## 静态日志回放

从R0原始轨迹出发，所有方法在同一批模型调用位置判断可复用性。该部分ReuseRate=ModelSaving，不会表达真实路径变化，也不从模型分歧推断最终Recall。

|策略|静态ModelSaving %|静态FalseSkipRate %|错误Skip/复用 %|
|---|---:|---:|---:|
|R0|0.000|0.000|N/A|
|R1|1.786|1.455|74.303|
|R2|0.761|0.590|70.732|
|R3|0.757|0.439|52.903|
|R4|0.559|0.303|49.476|
|R5|0.027|0.025|82.143|
|OracleA|0.459|0.000|0.000|

FalseSkipRate定义为“历史复用Skip且shadow fresh模型Keep”的数量除以该轨迹上所有fresh Keep；同时报告错误Skip占复用数，避免分母歧义。它是模型参照分歧率，并非真实Recall有害率。

## 真正的图搜索重放

先执行具有真实路径变化的规划搜索：在每个当前状态计算fresh分数作诊断，再应用复用策略，记录每次cache miss的Node ID、17维状态、fresh分数、是否复用。规划阶段的模型节省是逻辑计数，物理上仍运行shadow模型，不能据此宣称已经减少实际推理。

再从头执行原生图搜索，按离线决策记录在读取前真正绕过复用位置的MLP。逐位置校验Node ID和完整状态；非复用位置fresh分数也必须匹配。最终ID、命中数、读取、展开、复用数与规划完全一致，且真实模型调用计数等于eligible-reused。

|策略|Recall@10 %|ΔRecall pp|ModelSaving %|ReuseRate %|FalseSkipRate %|SSD请求/q|展开/q|均值µs|p95µs|预算通过|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|R0|99.640|0.000|0.000|0.000|0.000|93.410|102.847|6723.06|8214.55|是|
|R1|98.170|-1.470|1.781|1.783|1.473|92.228|101.665|6716.90|8167.56|否|
|R2|98.860|-0.780|0.761|0.760|0.603|92.958|102.395|6732.97|8328.00|否|
|R3|99.620|-0.020|0.754|0.753|0.434|93.039|102.476|6692.34|8115.24|是|
|R4|99.620|-0.020|0.558|0.557|0.302|93.161|102.598|7109.32|9798.63|是|
|R5|99.590|-0.050|0.027|0.027|0.025|93.395|102.832|6970.99|9183.11|是|
|OracleA|99.640|0.000|0.459|0.459|0.000|93.410|102.847|6641.98|8081.79|是|
|OracleB|99.440|-0.200|1.509|1.510|1.187|92.436|101.873|6666.89|8076.35|是|

![复用与质量](figures/reuse_quality.png)

均值和p95分别取三轮对应统计的中位数，原始三轮范围保留于summary.csv/results.json。配置随机顺序串行运行、每个进程预热，节点缓存固定。CPU未绑核，微小时间差可能属于主机/I/O波动，不据此宣称稳定加速。读取是提交请求，不是物理SSD读盘次数。

ModelSaving分母固定为R0实际调用数；ReuseRate分母是各自变化后路径上的eligible数。因此两者不一定相等，模型调用变化中也可能包含路径变化，不能把二者混用。实际绕过调用数量另由reused字段保存。

Hybrid的ModelSaving查询聚类bootstrap95%区间=[0.4562, 0.6713]%，ΔRecall区间=[-0.05, 0.0]pp；历史库固定，区间不含历史采样不确定性。

计时排除了Query近邻搜索、状态匹配和缓存维护，因为决策已离线生成；仍包括O(1)决策记录访问、前缀一致性检查、正常搜索状态提取与I/O。它是用于排除系统开销的搜索重放延迟，不是线上Query Cache延迟。所有策略包括R0采用同一回放框架。

## Oracle的范围与质量证据

Oracle A免费知道当前模型是否Skip。它只在历史节点有Skip且当前也Skip时复用，因此所有结果ID与R0逐字节一致，附加Recall损失为0。计时回放通过预存Oracle结果跳过推理；在线算法不能获得这项免费知识。

Oracle B对每条Query在冻结的完整轨迹菜单中选一种，目标为最大总调用节省，约束全测试集附加损失不超过0.2pp（1000查询×10结果对应20个命中）。使用精确多重选择背包，允许无复用回退，也允许不同Query的命中改善抵消损失；这是全局均值预算，不保证每条Query无损。之后真正重放选中的逐Query轨迹，核验质量约束和物理调用数。

Oracle B证书：1545次调用节省，实际少命中20个，预算20个。选择菜单和逐Query来源见OracleB_certificate.json、OracleB_choices.csv。它使用测试GT，明确不可部署；未搜索菜单之外所有节点组合，因此不宣称无条件最优。

## 旧全局一次标记 baseline

|方法|Recall %|SSD请求/q|均值µs|p95µs|
|---|---:|---:|---:|---:|
|Legacy Global Mark|90.550|92.304|6711.62|7431.52|

该方法未标记节点直接按原DiskANN执行，不运行MLP；因此不能将其“模型调用为0”当作相对于R0的复用ModelSaving。其计时包含位图查找，也不与离线决策记录的成本口径混淆。

## 五类成本与实现取舍

以下来自额外软件区域profile运行，互斥分解为distance/model/memory/SSD/bookkeeping，独立于上面的无计时器延迟重复。memory仅为显式PQ数据聚集与向量复制；隐式内存等待归属正在执行的区域。SSD是同步reader提交/等待时间，不是设备物理时间。

|方法|distance µs|model µs|memory µs|SSD µs|bookkeeping µs|
|---|---:|---:|---:|---:|---:|
|R0|138.61|145.83|174.06|5826.25|405.96|
|R4|138.16|145.18|187.31|5982.01|422.24|
|OracleA|138.09|143.55|196.94|5855.70|417.02|
|OracleB|135.69|142.45|186.70|5828.14|411.91|

![调用与读取分开](figures/calls_reads.png)

以本轮R0模型区域145.83µs/query作量级参考，即使免费Oracle A复用，其纯评分区域可减少约0.669µs/query，约占原五项成本0.01001%。这只是固定路径区域成本乘复用比例的估算，不是独立测得的端到端加速，也不包含真实Query-NN和状态查找的额外支出。

是否继续实现在线Query Cache，应同时看历史标记可用覆盖、质量预算内可绕过的调用数量、模型本身占比及尚未计入的匹配/维护成本。若主Hybrid与Oracle的可省评分空间都很小，即便图回放某轮略快，也不能证明在线缓存值得实现。

## 日志、验证与复现

- PROTOCOL.json、configs.json：测试前固定的算法、网格与Oracle菜单；SELECTION_LOCK.json为验证选中参数。
- validation_static.csv、test_static.csv：静态回放；validation_results.csv、test_all_results.csv：完整实际图搜索。
- plans/*.tape：规划轨迹；timing/*.csv、*.ids：真实跳过模型后的回放统计与结果。
- summary.csv、results.json、upper_bound_analysis.json、uncertainty.json：最终汇总与边界。
- experiment.log、build.log、plans/*.log、timing/*.log及command.json：完整日志与命令。
- verification.json、PROVENANCE.json、artifact_manifest.json、COMPLETE.json：核验、源输入哈希和结束证明。
- figures/：PNG/SVG/PDF。

```bash
cd /home/lxyue/projects/learned-gate
.venv/bin/python experiments/e3_reuse/prepare.py
.venv/bin/python experiments/e3_reuse/build_native.py
.venv/bin/python experiments/e3_reuse/run.py
```

既有E1/E2文件、DiskANN源文件和原二进制不修改。改变模型/策略/数据时使用新输出目录，不混用旧完成记录。

## 扩展：同时复用Keep与Skip，避免低估预测缓存价值

上面的R0–R5主表研究历史Skip标记。若目标是避免重复预测，历史Keep也可以复用：它省掉推理，仍然读取和展开节点。为避免仅凭Skip覆盖不足就否定整个决策缓存，另加独立的Keep/Skip缓存实验，冻结训练距离界限，用相同验证0.2pp约束选择状态阈值。核心旧结果不改名、不覆盖。

D3=同节点状态NN；D4=最近3条Query＋同节点状态NN；D5=多Query的Keep或Skip一致共识。Oracle A±免费知道当前决策，并在任一同节点历史中存在相同Keep/Skip时复用；相比Skip-only Oracle，其范围更广，仍不可部署。Oracle B±在已回放的原31种轨迹与验证选中的双决策轨迹菜单内作质量背包优化。

|策略|Recall %|ΔRecall pp|ModelSaving %|ReuseRate %|FalseSkipRate %|SSD请求/q|均值µs|p95µs|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|D0|99.640|0.000|0.000|0.000|0.000|93.410|6828.70|8623.04|
|D3|99.620|-0.020|8.627|8.626|0.434|93.421|7333.29|10355.79|
|D4|99.620|-0.020|4.451|4.450|0.302|93.374|6931.09|8780.56|
|D5|99.590|-0.050|0.486|0.486|0.025|93.444|6913.35|8312.36|
|OracleA_both|99.640|0.000|14.874|14.874|0.000|93.410|6935.58|9491.63|
|OracleB_both|99.560|-0.080|14.992|14.992|0.177|93.446|6920.72|8681.78|

![Keep与Skip决策复用](figures/both_decisions.png)

双决策Hybrid的实际模型调用减少4.451%，相对R0请求数变化-0.036/q；状态NN的调用减少8.627%，请求变化0.011/q。Keep复用说明省预测并不等同额外I/O减少。

双决策FalseSkipRate只计复用Skip但fresh为Keep；复用Keep但fresh为Skip另存false_keep。每种策略的Keep/Skip复用数量和两类分歧见both_results.json中的diagnostics。规划使用shadow评分；正式回放确实不执行这些复用位置的模型推理，并验证最终路径及物理调用计数。

Oracle A±的模型调用减少上界为14.874%，原始结果逐ID无损。Oracle B±菜单内最大调用节省为14.992%，实际少命中8个（预算20个），不是任意决策组合的全局最优。

按本轮R0纯模型评分区域作参考，免费Oracle A±能去掉的评分约21.69µs/query，占原五项时间约0.324%。这是纯评分区域的估算，不是端到端硬上界；真实查找、匹配、状态提取与维护仍需计入。

## 是否继续实现在线Query Cache

本轮建议暂缓完整在线Query Cache实现。在当前1000条自然历史查询与较轻量MLP下，Skip-only Hybrid仅减少约0.6%的模型调用；即使允许Keep/Skip复用，Hybrid减少的评分时间仍只有数微秒/query。免费模型决策Oracle的纯评分收益也只是总搜索时间的约千分之几，实际在线开销尚未计入。当前证据支持“存在预测复用”，不足以支持“值得投入完整在线Query Cache”。该建议限于当前工作负载、历史库规模、模型成本和预算，不能外推为任何更贵模型或高重复请求流都没有价值。

双决策实验文件：BOTH_PROTOCOL.json、BOTH_SELECTION_LOCK.json、both_summary.csv、both_results.json、both_plans/、both_timing/。初始计时与额外编译有重叠，已全部保留在timing_pilot/并排除；最终timing/在额外实验全部完成后串行重测，详见TIMING_RESTART.json与experiment.log。

扩展复现：先执行`.venv/bin/python experiments/e3_reuse/build_both.py`，再执行`.venv/bin/python experiments/e3_reuse/run_both.py`；然后重跑主run.py汇总。两组正式计时不得同时运行。

## 历史记录规模

历史状态库的序列化大小为15,558,996字节，另有1000条128维float32 Query向量512000字节；模型文件42,644字节。运行时节点索引和容器开销未单独测量，不能把序列化大小当作总RSS；离线决策记录仅为评估产物，不代表可部署的缓存设计。详见history_memory.json。

固定R0路径上，有任意同节点历史Keep/Skip记录的位置占比为17.013%，是该路径、该历史库、要求同节点时的原始可用性上限。它不能外推为任意改变路径后的通用上界。
