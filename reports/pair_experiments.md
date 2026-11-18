# 专题附录 · 宽基指数间的配对交易

本专题考察做多一条宽基、做空另一条宽基能否提取相对收益。主研究考察成分股贡献、行业内外抵消与指数配置价值。[返回宽基投资价值主报告](preliminary_report.md)

| 实验 | 报告与数据 |
|---|---|
| 早期配对与指定组合复现 | [初次实验](pair_hedge_report.html) · [上证综指／沪深300复现报告](sse_csi300_replication.pdf) |
| 六条宽基全组合搜索 | [完整报告](pair_trading_full_research.md) · [全部1,560种配置](../outputs/tables/all_pair_results.csv) · [年度滚动选对](../outputs/tables/walk_forward_results.csv) |
| 收益、夏普与稳定性排行 | [收益榜](../outputs/tables/top_return_pairs.csv) · [夏普榜](../outputs/tables/top_sharpe_pairs.csv) · [综合榜](../outputs/tables/top_robust_pairs.csv) |
| 冠军净值与数据核验 | [完整历史曲线](../outputs/figures/pair_champion_full_history.png) · [逐日净值](../outputs/tables/pair_champion_full_history.csv) · [两次实验的口径](pair_experiment_explanation.md) · [2024年9月净值跃升审计](pair_jump_audit.md) |
| 工具与交易成本 | [可交易性报告](pair_trading_implementation.md) · [成本情景](../outputs/tables/cost_sensitivity.csv) · [工具回放](../outputs/tables/pair_instrument_returns.csv) |

测试期收益冠军由测试结果回看确定。验证期提前锁定策略及逐年滚动选择的收益，均可在完整报告中对照。

## 复现

在仓库根目录安装 [Python 依赖](../requirements.txt) 后运行

```shell
python -m unittest discover -s tests
python scripts/run_pair_search.py
python scripts/plot_pair_champion_full_history.py
python scripts/replicate_sse_csi300.py
```

归档文件 [outputs/pair_paths](../outputs/pair_paths/) 按多头及空头指数代码命名。每个 NPZ 的 ids 给出配置顺序，date 为收益日期，w 和 v 为签名权重，ret 为一倍总敞口日毛收益，raw_ret 为原始敞口日毛收益，traded 为每元初始资本的日成交额。复现 PDF 需要 Playwright 和 Edge 浏览器。[原始发布回放记录](../outputs/logs/publication_replay.md)
