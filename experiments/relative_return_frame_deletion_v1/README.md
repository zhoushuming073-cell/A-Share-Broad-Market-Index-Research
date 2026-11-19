# 沪深300相对上证综指的关键交易日依赖性

[独立PDF](research_report.pdf) · [与多维对冲报告合订的PDF](combined_research_report.pdf) · [可编辑研究报告](research_report.md) · [衔接章节](research_report_addendum.md)

使用价格指数的日对数收益。对组内两指数同时屏蔽选中日期，再从1开始用保留的日收益重新复合。原始数据只有2.07%的终点相对财富优势，事后删除两个最有利日即可抹去。该现象描述历史敏感性。均匀、分层和连续块试验均条件于这份固定行情，推断时间序列均值另用HAC和有放回区块Bootstrap。

## 复现

在仓库根目录使用 Python 3.13。依赖包和日历数据版本见 requirements.txt。出图使用 Microsoft YaHei 中文字体；PDF生成复用仓库现有的 Windows 桌面 Word、Pandoc 及python-docx排版。整合版读取上一个实验的已发表PDF。

```shell
python -m pip install -r experiments/relative_return_frame_deletion_v1/requirements.txt
python experiments/relative_return_frame_deletion_v1/run.py
```

每批1,000次，由两个主种子、机制编号、删除日数和批起点产生子种子。内核按Floyd无放回抽样并逐保留日递推两条对数净值，保存汇总而非全部模拟的日路径。批完成后原子写入 .cache，重启跳过已完成批。缓存以输入、配置及采样代码内容寻址；Git中提供确定性复算所需代码及数据、终点Q抽样值和各机制各预算第一例的确切删帧日期。二进制缓存可由本命令重建。完整复算不依赖网络行情接口。

## 设计与规模

| 设计 | 限额与实际预算 |
|---|---|
| 均匀无放回 | 0.5%、1%、2%、5%、10%、20%、30%，每档100,000次；另10个实际k每档3,000次 |
| 年度／日历月份分层 | 每种机制17个实际k，每档3,000次。配额通过累积随机取整，使每层误差小于1日且总k一致 |
| 连续交易日区块 | 5、10、20、60日四种块长；每种17个k，每档3,000次。块可邻接，不重叠；含不足一块的余数 |
| 第二种子复核 | 核心七档均匀抽样各5,000次 |
| 漂移均值的时间序列Bootstrap | 循环块长5、20、60，各20,000次，独立于上述删帧计数 |

合计1,071,000次有效删帧模拟。1%、2%等配额四舍五入至整数；双尾按每端分别取整。保留日相关性用日对数收益计算。路径指标仅在保留日求均方根或极值；画图在实际日历的删去日保持净值。无合成年化业绩。

## 数据与结果

| 问题 | 可核对的文件 |
|---|---|
| 原始事实与滚动相对收益 | [原始统计](results/original_metrics.csv)、[逐日数据](results/daily_relative_returns.csv)、[年度贡献](results/annual_contributions.csv)、[状态分组](results/market_states.csv) |
| 关键日期、定向删除与路径 | [正贡献排序](results/top_positive_days.csv)、[负贡献排序](results/top_negative_days.csv)、[确定性结果](results/deterministic_deletion.csv)、[合成路径及被删日标志](results/synthetic_paths.csv.gz) |
| 随机历史的分布 | [均匀](results/random_deletion_summary.csv)、[分层](results/stratified_deletion_summary.csv)、[区块](results/block_deletion_summary.csv)、[全部主设计的终点Q样本](results/terminal_Q.npz)、[各组首抽日期](results/first_draw_dates.csv) |
| 路径拟合与同预算经验位置 | [拟合指标](results/fit_metrics.csv)、[定向对照](results/equal_budget_comparison.csv)，below及equal为经验频率 |
| 稳健位置与推断 | [稳健估计](results/robust_locations.csv)、[HAC、年份、联合等均值及Bootstrap](results/statistical_tests.csv) |
| 独立核验 | [解析对照](results/analytic_checks.csv)、[独立种子](results/replica_summary.csv)、[收敛轨迹](results/convergence.csv)、[核验记录](results/verification.json) |
| 对冲账户的可比口径 | [同区间0.5多沪深300、0.5空上证及其成本](results/hedge_bridge.csv) |

模拟分布CSV中，每个指标包括均值、中位数、标准差、均值蒙特卡洛标准误、1/5/25/75/95/99分位数；p_flip_lo及p_flip_hi为Wilson 95%区间。D、Q、收益及波动用小数；pos、neg为日对数差的可加和贡献。财富比值Q不代表真实可交易多空收益。
