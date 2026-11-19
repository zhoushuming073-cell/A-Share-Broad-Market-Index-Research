# 上证综指／沪深300多维多空策略搜索

[正式PDF](research_report.pdf) · [可编辑源稿](research_report.md) · [逐日净值](results/daily_nav.csv) · [经过核验的逐日账户](results/daily_positions.csv)

搜索八种逻辑，按交易费用、借券和融资后的日终权益比较固定参数库与半年滚动规则。账户区分两条腿的带符号实际仓位、收益、成交额、自然日资金费用及共同／相对损益。所有统计以本仓库既有价格指数快照为基础。

## 复算

在仓库根目录、Python 3.13 环境执行下列命令。分析与绘图使用本目录依赖；最后的出版步骤复用仓库 v2 中文排版模块，在 Windows 上调用桌面 Word 输出带目录PDF。

```shell
python -m pip install -r experiments/multidimensional_longshort_search_v1/requirements.txt
python experiments/multidimensional_longshort_search_v1/run.py
```

可单独运行 verify.py 审计已导出的台账，或运行 report.py 从结果重新生成图表与报告。SQLite 于每一完整外窗写入检查点，恢复时重做尚未完成的窗口，再重建所有榜单。输入、配置和核心源码的SHA-256指纹校验检查点；指纹变化会重新计算。在相同版本环境运行。SQLite 中的 Python 序列化对象仅供本仓库可信实验恢复使用。

## 实验协议

- 数据见上级 four_pair_tests_v2/data/input_closes.csv，只用000001、000300。日收益由收盘价相邻比计算，与XSHG交易日历对齐。无风险利率为零。5bp成交费、2%借券、3%多头融资、空头卖出款全额占用，最大2.5倍总敞口。遇到保证金超限强制重设目标；净资产耗尽抛出异常。本次全部完成的历史账户均保持偿付能力。
- config.yaml 保存种子、预算、费用及评分权重；strategies.spec 是条件搜索空间入口。Sobol按策略族在各自的有效参数维度生成，加扰种子逐族派生。重复参数用规范化JSON的SHA-256前16位标识；同一窗缓存训练和验证评价，保留重复提议记录。费用单位为每元初始资金的累计金额；指标cost、turnover分别为每日期初资产比例之和折年。年度与月份按自然历，包括样本边缘不完整期间。
- 第一窗有独立的2015–16训练、2017验证段。之后按自然半年滚动，近三年含两个训练年份及最近约一年验证，最后一个收盘留作执行。NSGA-II三个目标为验证CAGR、负最大回撤与Sharpe。前沿邻域及时间窗口均以当期已知信息选定。每个算法阶段可继承先前阶段高分点。day t的信号使用至t−2收盘的数据；执行于t−1收盘。每个窗口结尾平仓，下一期将上期资产作为新本金。没有远期重叠标签。
- 首窗形成的1,245个去重参数全部进入固定库，独立于后续行情。此后滚动TPE及NSGA-II临时产生的参数也参加当窗竞争，但不会追溯加入首窗固定库。累计23,942条搜索及冻结库比较记录，独特参数与评价行数应分开理解。全历史榜单是在后续历史已知后选择；真正逐窗可执行的是wf系列。
- 理想榜无合格者时保留带列名的空表，另外列出理想分最高的10个活跃候选；绘图字段ideal_curve表示此时的近似者。每个窗口若无过门槛的理想解，则使用当期近似者交易，walk_forward中的ideal_gate记录判定。现金比例由信号决定，最低活跃率门槛约束低交易频率偏好。

## 结果文件

| 内容 | 文件 |
|---|---|
| 每个提议、参数、种子、算法、折次、训练及验证结果 | [all_trials.csv](results/all_trials.csv) |
| 每个固定参数的全期成本、风险及分期结果 | [candidate_metrics.csv](results/candidate_metrics.csv) |
| 各内层验证的非支配解、完整固定库事后前沿 | [inner_fronts.csv](results/inner_fronts.csv)、[pareto_front.csv](results/pareto_front.csv) |
| 三套榜单及不合格近似者 | [收益](results/top_return.csv)、[理想](results/top_ideal_curve.csv)、[综合](results/top_overall.csv)、[近似](results/ideal_nearest.csv) |
| 窗口截止、冻结参数及其下一窗表现 | [walk_forward.csv](results/walk_forward.csv)、[相同窗口重新开仓的固定参数](results/matched_fixed_folds.csv) |
| 择时、相对收益及成本 | [attribution.csv](results/attribution.csv)、[periods.csv](results/periods.csv) |
| 算法收敛、参数敏感性、成本压力 | [convergence.csv](results/convergence.csv)、[robustness.csv](results/robustness.csv)、[parameter_effects.csv](results/parameter_effects.csv) |
| 审计、十一组图及矢量源 | [verification.json](results/verification.json)、[figures](figures/) |

所有候选与基准的全历史CAGR差均在候选表中。榜单中的main_cagr和append_cagr是用已冻结的首窗参数在指定时期重新从空仓评价的结果；冠军身份仍是事后选的。periods是从相应实际账户中截取的阶段表现，切窗时是否平仓会影响比较。三维超体积按固定劣参考点计算，算法产生的参数彼此相关，收敛轨迹不能用于推断未来收益显著性。
