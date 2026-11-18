# A股宽基指数投资价值研究

本仓库公开项目的行情快照、数据库、历史成分股资料、实验代码、逐日回放、成本情景、图表、研究报告和运行日志。当前数据截至 **2026-11-17**。研究使用的多数指数序列是价格指数；配对交易收益是模拟结果。

## 从哪里开始

- [完整配对交易研究报告](reports/pair_trading_full_research.md)
- [交易工具与成本报告](reports/pair_trading_implementation.md)
- [上证指数／沪深300复现报告](reports/sse_csi300_replication.pdf)
- [冠军配对完整历史曲线](outputs/figures/pair_champion_full_history.png)与[逐日回放](outputs/tables/pair_champion_full_history.csv)
- [数据字典与来源](data/README.md)、[数据缺口](reports/data_gaps.md)、[研究执行协议](研究执行协议_v1.md)

配对实验涵盖库中六条宽基指数、15个无序配对、30个多空方向和1,560种策略配置。2024-07-02至2026-11-17的测试期收益冠军是做多上证指数、做空上证100，使用衰减系数0.94的EWMA对冲比例，每月更新。它是**测试期事后冠军**。按照验证期规则提前锁定的策略是做多沪深300、做空上证100。请将两者的表现和[逐年重新选对结果](outputs/tables/walk_forward_results.csv)一起阅读。

## 本地复算

建议使用 Python 3.13。安装[依赖版本](requirements.txt)后，在仓库根目录运行

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/run_pair_search.py
python scripts/plot_pair_champion_full_history.py
python scripts/replicate_sse_csi300.py
```

`run_pair_search.py`读取已提交的 `data/a_share_indices.sqlite`，重算全部配对、成本、年度滚动及报告。该脚本在生成交易工具报告时优先读取 `outputs/inputs/pair_instruments/` 的已存行情。图表与结果写入 `outputs/`，报告写入 `reports/`。复现PDF需要 Playwright 和可用的 Edge 浏览器。核心配对回放仅依赖库内收盘数据，无须再次下载。

原始行情CSV位于 `data/raw_index/`，历史成分股抓取材料位于 `data/raw_constituents/`。归档 `outputs/pair_paths/A_B.npz` 保留所有策略的逐日路径，其中 `ids` 给出配置顺序，`date` 为收益日期，`w` 与 `v` 是两腿权重，`ret` 为归一到1倍总敞口的日毛收益，`raw_ret` 为原始敞口口径，`traded` 为每1元初始资本的日成交额。复利净值为 `(1+ret)` 连乘。

主要导出表包括 [全组合结果](outputs/tables/all_pair_results.csv)、[三份排行榜](outputs/tables/top_return_pairs.csv)、[成本敏感性](outputs/tables/cost_sensitivity.csv)、[逐年候选记录](outputs/tables/walk_forward_trials.csv)、[工具回放](outputs/tables/pair_instrument_returns.csv)。每个文件的 SHA-256 见 [归档清单](outputs/logs/publication_manifest.csv)，公开前复算记录见 [发布日志](outputs/logs/publication_replay.md)。

## 数据与解释边界

指数行情由 AKShare 调用东方财富接口抓取，采集时间、批次和原始快照校验值记录在数据库 `ingest_batch`。ETF与IF行情的缓存也已提交。基于指数价格的理论净值、成本情景净值以及ETF/期货工具回放有不同口径，详情见各报告。上证100空头的历史工具与券源尚未得到可核实的完整数据。
