# 连续偏移的条件归因

先阅读 [冻结协议](protocol.md) 和 [结果报告](research_report.md)。基线c99d3fe，协议提交7771bf3。实验在独立目录读取历史归档。全期两指数价格及市场代理沿用V1。CH因子文件保存于仓库outputs/inputs。未下载新数据。

在本目录运行以下命令，生成全部统计、图表和报告。使用 Python、NumPy、pandas、SciPy、statsmodels、Numba、openpyxl、PyYAML、matplotlib、tabulate。首次运行编译数值内核，图表优先使用微软雅黑。

    python run_all.py
    python -m unittest discover -s tests -v

## 输出索引

| 文件或前缀 | 读法 |
|---|---|
| daily | panel区分完整日历与CH共同日历。逐日保存前收盘参数所得条件贡献、拟合差与误差；收益以对数小数计。beta为基础市场斜率，up_extra为上涨额外贡献。 |
| periods、period_attribution、case_attribution、case_increments | 窗口或同号阶段的实际起止日期。归因表各项均以累计对数基点计，error为模型残差。 |
| neutralized_windows、counterfactual_errors、case_refits | 对每个60日窗口比较本地与全期名单；removed是窗口内实际中和日数。主规格先估计后中和。重新估计仅用于案例K=2敏感性。 |
| yearly、regimes、local_loss、changes | 年度损失改善为日损失差的和。leave只排除评价日；local_loss按60、120、252日逐窗加总。changes比较相邻年份的市场斜率。 |
| holding_windows、holding_metrics、holding_tests | holding_windows的d、fitted、error是窗口累计对数小数；另列基点绝对误差和平方基点误差。miss率分母为所有实际大幅窗口。stride=1为逐日滚动，否则为固定交易日网格。 |
| null_tests、null_draws、primary、secondary_checks | 高尾经验p值用加一修正。npz保存主统计量在各区块与四种基准下的模拟样本。六项预设主检验联合Holm校正。局部截距等附加诊断单列。 |

models.py 复用V1数据核验和滤波器。windows.py 处理日期掩码，evaluate.py 保留缺口和评价时间轴，nulls.py 实现每次重选的搜索。reporting.py 与 narrative.py 从结果表生成九幅图及中文稿。
