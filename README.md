# A股宽基指数的内部收益抵消与投资价值研究

**[下载完整 Word 研究报告](research_report.docx)** · [报告 Markdown 源稿](reports/integrated/research_report.md) · [宽基投资价值初稿](reports/preliminary_report.md) · [实验完成情况](reports/experiment_status.md) · [复现说明](reports/reproduction.md)

当一条宽基指数收盘只涨跌一点时，它的成分股究竟经历了多大的涨跌？本研究用前一交易日的成分权重衡量每只股票的收益贡献，再考察这些贡献在行业内部和行业之间抵消了多少。投资价值由长期收益、风险和组合表现共同检验。

研究对象为国证1000、中证A100、上证100和沪深300。上证综指为主要基准，中证全指为辅助基准。价格指数共同覆盖2015-01-05至2026-11-17，共2,884个收盘点。目前按历史权重核验的内部抵消覆盖一个交易日。

后续指数行情研究采用固定九指数池，在上述六条基础上加入中证500（000905）、创业板指（399006）和科创50（000688）。九条共同样本从科创50正式发布实时行情的2020-07-23收盘开始。[研究池名单与数据说明](data/README.md) · [九指数共同收盘价](outputs/tables/index_closes_nine.csv)

九指数实验覆盖36组配对及10万条联合区块模拟路径。[九指数实验报告](experiments/nine_index_path_dependence_v1/research_report.md) · [原始与同步删日对照图](experiments/nine_index_path_dependence_v1/figures/nine_paths.png)

七指数V2实验考察21组配对的关键日重合，以及整体分化、并集、重复集合和投票法的删日效果。Word第13章收录完整实验，表52列出逐组结果，图51至图53展示各配对独立删日前后的曲线。[七指数V2报告](experiments/seven_index_convergence_v2/research_report.md) · [计算代码与结果](experiments/seven_index_convergence_v2/)

Word第14章检验两条累计曲线在已有高低差时是否平行。表55逐组比较前后半段，表56给出60、120和252日窗口的删日效果。[起点重设示例](experiments/seven_index_convergence_v2/figures/10_rebased_example.png) · [21组后半段对照图](experiments/seven_index_convergence_v2/figures/11_late_pair_shapes.png) · [逐组距离与改善率](experiments/seven_index_convergence_v2/results/local_shape_pairs.csv)

七指数V3复核原始价格的局部路径。完整60、120、252日重置窗口在后半段分别缩小6.23%、5.50%、10.17%，同步区块检验未检出显著的前后差异。Word第15章给出21组热力图、中央日口径复现及早期偏移集中度。[V3报告](experiments/seven_index_convergence_v3/research_report.md) · [复现代码](experiments/seven_index_convergence_v3/run.py) · [区块检验结果](experiments/seven_index_convergence_v3/results/bootstrap_tests.csv)

## 指数涨幅背后的个股贡献

2026-11-02，中证A100成分股带来的正贡献为131.96个基点，负贡献为−65.03个基点。两者合计使指数上涨约66.93个基点，按贡献绝对值计算，当日有 **66.02%** 的涨跌相互抵消。国证1000同日可匹配的996只股票中，行业内部抵消占总抵消的 **94.19%**。这两组数分别回答抵消了多少、抵消发生在哪里。它们的历史稳定性仍待连续成分权重和行业数据验证。[证券级贡献](outputs/tables/e4_snapshot_contributions.csv) · [行业分解](outputs/tables/e5_decomposition.csv)

## 主线实验与阅读顺序

| 研究问题 | 实验与现有证据 | 查看结果 |
|---|---|---|
| 指数内部有多少涨跌相互抵消？主要发生在哪里？ | E4、E5。四条宽基有单日权重归因，国证1000另有当期行业分解。 | [抵消率与复制误差](outputs/tables/e4_snapshot_metrics.csv) · [行业贡献](outputs/tables/e5_industry_contributions.csv) |
| 四条宽基承担哪些市场与风格风险？ | E1—E3。比较两个市场基准、行情状态及规模和价值等因子。 | [市场暴露](outputs/tables/e1_regression.csv) · [状态日历](outputs/tables/e2_states.csv) · [因子暴露](outputs/tables/e3_coefficients.csv) |
| 震荡中的抵消是否源于行业轮动？能否帮助预测风险？ | E6、E7。已有计算入口，连续行业贡献与日度抵消面板待补。 | [所需历史字段与覆盖范围](reports/data_gaps.md) |
| 同时持有多条宽基能否改善风险收益？ | E8。约束最小方差组合在样本外较月度等权降低波动，夏普改善仍有较大不确定性。 | [组合绩效](outputs/tables/e8_performance.csv) · [区块抽样区间](outputs/tables/e8_bootstrap_differences.csv) |

## 下一步研究

优先取得四条宽基在历史各期实际生效的成分、权重，以及同期个股收益和行业归属。先在普通交易日、调样日和公司行为日逐日对照官方指数收益，记录缺失权重与复制误差，再比较不同行情下行业内外抵消的占比。沪深300已有23期名单的等权分化观察，可用于交叉检查。[数据盘点](reports/data_inventory.md) · [成分股覆盖说明](data/constituent_coverage.md) · [研究执行协议](研究执行协议_v1.md)

## 数据与复现

原始行情、历史成分股文件和 SQLite 库位于 [data](data/)。输入快照、实验表、图和日志均已归档。从已有权重快照复算 E4 和 E5，可在仓库根目录运行以下命令。建议使用 Python 3.13。

```shell
python -m pip install -r requirements.txt
python scripts/run_e4_offsetting.py
python scripts/run_e5_decomposition.py
```

其余主线实验的运行顺序、输入字段与预热期见[完整复现说明](reports/reproduction.md)。[数据来源](data/README.md) · [运行日志](outputs/logs/) · [文件校验清单](outputs/logs/publication_manifest.csv)

## 专题附录

指数间多空配对实验单独归档，包含参数搜索、逐日净值、成本和工具回放。[配对交易附录与复现入口](reports/pair_experiments.md)

六指数探索性实验比较市场极端日与配对分化日对终点排名的影响。[脚本](scripts/pair_extreme_days.py) · [统计表](outputs/tables/pair_extreme_days_summary.csv) · [剔除日期](outputs/tables/pair_extreme_days_exclusions.csv) · [对照图](outputs/figures/pair_extreme_days.png)
