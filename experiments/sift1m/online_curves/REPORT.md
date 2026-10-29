# Recall / cost comparison

![Curves](recall_cost_curves.png)

三个方法都使用相同 1000 个 held-out query；三轮所有配置交错随机运行，取中位数。上排展示全部点，下排放大 Recall>=99%。
DiskANN varying L：L=20,40,50,60,70,80,90,100,120,150。
Early Termination：固定 L=100；按现有 io_limit 接口，在 beam 边界检查 I/O 预算 B=20,40,60,70,80,85,90,95,100,105,110,120,140；可能最多超出3个读取，横轴使用实际请求数。
Learned Expansion Pruning：M3，L=100；validation FNR targets=0.1%,0.25%,0.5%,1%,2%；使用上一实验的固定 tau。512MB v 表常驻RAM，推理成本计入延迟。
Latency 是 mean query latency，横线显示三轮 min/max；主机为预热状态。没有从其他 query 集合拼接曲线。
曲线仅连接实测点，没有平滑拟合。所有84次运行的结果ID重复一致，Recall独立重算通过，I/O预算与gate计数核对通过。
Early Termination 在这里具体指 I/O-budget cutoff，不代表所有提前终止算法。
绘图用本地 Cairo，导出 PNG/SVG/PDF；CSV 和逐次测量保留。重现：先 run_online_curves.py，再 plot_online_curves.py。
已核验 L100 与既有基线、M3 各阈值的 Recall/Reads/Expansion 一致；io140 与 unrestricted L100 一致。
