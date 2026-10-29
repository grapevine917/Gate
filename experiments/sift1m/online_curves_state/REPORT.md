# Recall / cost comparison

![Curves](recall_cost_curves.png)

所有方法都使用相同 1000 个 held-out query；三轮所有配置交错随机运行，取中位数。上排展示全部点，下排放大 Recall>=99%。
DiskANN varying L：L=20,40,50,60,70,80,90,100,120,150。
Early Termination：固定 L=100；按现有 io_limit 接口，在 beam 边界检查 I/O 预算 B=20,40,60,70,80,85,90,95,100,105,110,120,140；可能最多超出3个读取，横轴使用实际请求数。
Learned Expansion Pruning：M3，L=100；validation FNR targets=0.1%,0.25%,0.5%,1%,2%；使用上一实验的固定 tau。512MB v 表常驻RAM，推理成本计入延迟。
Latency 是 mean query latency，横线显示三轮 min/max；主机为预热状态。没有从其他 query 集合拼接曲线。
曲线仅连接实测点，没有平滑拟合。所有99次运行的结果ID重复一致，Recall独立重算通过，I/O预算与gate计数核对通过。
Early Termination 在这里具体指 I/O-budget cutoff，不代表所有提前终止算法。
绘图用本地 Cairo，导出 PNG/SVG/PDF；CSV 和逐次测量保留。重现：先 run_online_curves.py，再 plot_online_curves.py。
已核验 L100 与既有基线、M3 各阈值的 Recall/Reads/Expansion 一致；io140 与 unrestricted L100 一致。

State-only MLP：17→128→64→1，仅使用 pre_io_state，不加载完整 v 表。五档 FNR 目标与 M3 相同，各自只用 validation 校准阈值。
本次所有四种方法重新交错测量，未将不同实验时段的 latency 拼到同一图中。

| 模型 | FNR target | tau | Recall@10 % | Expansion/query | SSD Reads/query | Mean latency ms | QPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| M3 | 0.1% | 0.0155834733 | 99.640 | 106.350 | 96.918 | 7.527 | 525.40 |
| M3 | 0.25% | 0.0259240922 | 99.640 | 102.699 | 93.272 | 6.999 | 564.78 |
| M3 | 0.5% | 0.0362158976 | 99.640 | 99.225 | 89.808 | 6.760 | 584.44 |
| M3 | 1% | 0.0524246283 | 99.630 | 94.141 | 84.732 | 6.664 | 592.62 |
| M3 | 2% | 0.0787507668 | 99.500 | 86.636 | 77.244 | 5.968 | 661.50 |
| State-only | 0.1% | 0.0157926977 | 99.640 | 105.803 | 96.369 | 6.659 | 593.54 |
| State-only | 0.25% | 0.0243524648 | 99.640 | 102.842 | 93.410 | 6.544 | 603.68 |
| State-only | 0.5% | 0.0368009172 | 99.640 | 98.754 | 89.327 | 6.129 | 644.22 |
| State-only | 1% | 0.0557178184 | 99.590 | 92.623 | 83.209 | 5.845 | 674.99 |
| State-only | 2% | 0.0822792202 | 99.480 | 84.749 | 75.358 | 5.254 | 750.57 |

重现：prepare_online_state_mlp.py；run_online_curves.py --with-state；plot_online_curves.py --with-state。
