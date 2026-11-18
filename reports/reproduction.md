# 实验复现说明

在项目根目录运行。已下载的输入保存在 outputs/inputs，输入文件与校验值列在 data_lineage.csv。复现现有结果时可直接从审计命令开始。官方接口和公共供应商的下载可能随时间变化。

    python -m pip install numpy pandas scipy statsmodels==0.15.0 matplotlib openpyxl xlrd requests
    python scripts/audit_research_data.py
    python scripts/run_e1_beta.py
    python scripts/run_e2_regimes.py
    python scripts/run_e3_factors.py
    python scripts/run_e4_offsetting.py
    python scripts/run_e5_decomposition.py
    python scripts/run_e6_rotation.py
    python scripts/run_e7_prediction.py
    python scripts/run_e8_portfolio.py
    python -m unittest discover -s tests -v

补采输入可单独运行 python scripts/fetch_research_inputs.py --hs300-stages。六条指数的两日官方点位可用 python scripts/fetch_official_snapshot.py 获取，需额外安装 AKShare。所有分析读取 data/a_share_indices.sqlite 的只读连接。市场收益由收盘点位重算，源表、原始 Excel 和原始 CSV 保留。

E6 以 --panel 接收 CSV，字段为 trade_date、index_code、industry、industry_contribution、industry_start_weight、stock_abs_contribution。权重和贡献用小数。某日不在指数内的行业用已核实的零权重与零贡献体现，缺失行业数据保留缺失并更正后再运行。

E7 以 --panel 接收 CSV，字段为 trade_date、index_code、market_return_20、index_vol_20、log_volume、cssd、csad、co、gross、between_share。每行是该日收盘时已经可用的特征。日度目标用数据库收益构造，训练标签须已到期。缺乏真实面板时，这两个入口输出 BLOCKED。

全样本范围从2015-01-06到2026-11-17。E2、E8因滚动预热使用更短的样本，E3止于因子可用日期。HAC主规格为5日滞后，E7按预测期限计算。Markov优化和配对区块 Bootstrap 使用种子20261118。软件版本与库哈希见 [data_quality.json](../outputs/logs/data_quality.json)。
