# RAM 导航预算与 I/O–CPU 权衡

复用 ram_topology 的固定节点/逐源出度/有效边预算 Original、Geometry、验证选定 History 图，底层原始 Vamana 图不修改。SIFT1M、DEEP1M；训练历史8,000，验证与独立测试各1,000。没有 GloVe 结果、不训练模型、不在线改图。

主扫描仅改变 RAM 最大展开预算：0/4/8/16/32/64/96/128/256/512，候选队列容量64。0保留同一缓存与导航内存，跳过导航，仅用原始medoid。预算不是实际展开数；同时报告实际展开、距离计算与达到预算上限的查询数量。历史图补测队列容量8/16/32/128/256，展开预算16/64/256；队列容量与展开预算是两个不同参数。

固定四入口交接（预算0无新增入口），入口为最近四个实际展开节点，并保留原始medoid候选；底层k=10/L=100/beam=4，无剪枝、早停或SSD图修改。所有配置四线程，MKL/BLAS1线程，OMP_PROC_BIND=close/OMP_PLACES=cores。性能试验串行运行、配置随机交错，不并行竞争测量。

队列统一预分配512容量工作区，运行时用reserve设置有效候选容量，不释放预分配存储。四线程工作区、缓存向量/原始邻接、导航CSR/IDs/元数据完整计入原有RAM字节上限。所有配置实际缓存+导航字节数一致，不通过更多内存混淆预算比较。

导航CPU指标使用CLOCK_THREAD_CPUTIME_ID，单查询实际线程CPU时间，与原有导航wall time同时记录；整体端到端包含导航和完整底层搜索，QPS按四线程批次墙钟测量。Submitted SSD Read Requests是O_DIRECT Reader提交请求，不是设备物理I/O。

独立验证集先扫描全部配置；每种拓扑要求总体Recall@10不低于本图budget64/queue64与no-navigation两者的较高值。单次测量端到端最快的两个合格配置、最低请求合格配置及参考64/64组成短名单，再交错追加两轮验证。最终从短名单中选择三轮中位端到端时延最低的合格配置，冻结到frozen_selection.json后才运行测试。不是穷举所有重复测量配置的全局最优证明。最低请求配置用于展示请求与CPU的取舍，不自动视为最快。

独立测试比较每图选定配置、各图64/64、公共no-navigation和历史图最低请求配置，五轮交错取中位数；请求数和Recall必须五轮一致。测试阶段另测历史图queue64完整预算曲线，未进入主对照的曲线点仅一轮，不能用单轮时间重选最佳配置。报告逐查询Recall下降/提高和请求差的5,000次配对bootstrap；总体Recall保持不是逐查询安全。测试若不满足质量目标，不能声称质量保持的改进；不能用测试反馈修改已经冻结的配置。

检查64/64的逐查询Recall/请求/导航距离计算与旧实验完全一致；实际RAM预算一致且不超上限；实际展开与队列耗尽情况记录。CPU计时调用产生极小但共有的测量开销。

复现：`cmake --build learned-gate/DiskANN/build --target search_ram_navigation -j4`；`OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 learned-gate/.venv/bin/python learned-gate/experiments/ram_budget/run.py`；先运行同目录current_recall.py，再运行report.py生成报告与预算曲线CSV。

补充条件质量对照（current_recall.py）：若验证集当前64配置的Recall低于no-navigation，较强主约束可能只允许禁用导航或很小预算。为区分“保持当前导航质量”和“恢复原始medoid质量”，历史图另以其64/64验证Recall作为唯一质量阈值，按相同三轮验证短名单规则选择配置，冻结后五轮交错重测选定配置与64/64；报告测试是否满足当前质量及更强no-navigation质量，绝不混称两者。两种验证阈值相同时复用主结果，不再重复。选择函数仅访问验证结果，测试反馈不参与选择。
