# 公开发布前的实验回放

运行环境为 Windows、Python 3.13.7。项目数据库以 SQLite 只读方式检查，发布前未重新抓取指数行情。

## 执行

```text
python -m unittest discover -s tests
........
Ran 8 tests in 2.299s
OK
```

随后对归档数据库和所有日度配对路径进行独立一致性核验。用每天两腿权重与指数收益重算日收益，再与全组合表的测试期复利结果核对。

```text
SQLite quick_check OK
6 indices; 30 directed pairs; 52 configurations each; 1,560 daily paths verified
Full-history champion terminal gross NAV 1.117193796
Full-history champion terminal cost-scenario NAV 1.000329881
```

核心重跑入口为 `python scripts/run_pair_search.py`，完整历史冠军曲线由 `python scripts/plot_pair_champion_full_history.py` 生成。归档的全部CSV、图表、PDF、HTML、数据库、行情快照与逐日 `npz` 可用下级 `publication_manifest.csv` 的 SHA-256 校验。外部ETF与股指期货缓存位于 `outputs/inputs/pair_instruments/`，重跑时优先读取归档缓存。

报告中测试冠军是依据测试期回看选出的策略。报告也单列验证期提前锁定的策略、年度滚动选择与多重检验结果。公开仓库不代表这些结果具有事先可交易性。
