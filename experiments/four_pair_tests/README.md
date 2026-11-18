# A股宽基指数多空对冲策略研究

[Word 研究报告](research_report.docx) · [可编辑 Markdown 源稿](research_report.md) · [中英文题名及摘要](research_report.md#摘要)

实验仅覆盖报告指定的四个指数。价格表首个收盘日为2015-01-05，末日为2026-11-17。每单位本金承担一倍总名义敞口，逐日按目标权重调仓。模拟对两腿实际成交金额收取单边5个基点，并对空头按年率2%计费。不同起点各自独立开平仓，选向测试额外保存了期初空仓账户。

| 研究问题 | 回查入口 |
|---|---|
| 固定持有252日时，每个起点赚亏多少？ | [逐笔明细](results/e1_all_entries.csv) · [逐日净值数组](results/e1_holding_paths.npz) · [收益分布](figures/e1_distribution.png) |
| 按已知历史选方向，与事后选方向差多少？ | [冻结方向记录](results/e2_direction_selection.csv) · [测试期空仓起步的六条净值](results/e2_fresh_test_paths.csv) · [事后选向逐对比较](results/e2_hindsight_comparison.csv) · [配对检验](results/e2_paired_tests.csv) |
| 对冲方法、更新频率、同指与跨指影响什么？ | [阶段统计](results/e2_e4_statistics.csv) · [逐日账户及收益归因](results/e2_e4_daily.csv.gz) · [图与SVG源文件](figures/) |

## 从仓库复现

在仓库根目录执行。分析环境使用 Python 3.13.7 及根目录记录的依赖版本。图中中文在原环境使用 Windows 的 Microsoft YaHei 字体。若只想验证已公布的账户结果，安装 NumPy 和 pandas 后直接运行第二条 Python 脚本命令即可。

```shell
python -m pip install -r requirements.txt
python experiments/four_pair_tests/run.py
python experiments/four_pair_tests/verify.py
```

run.py 从仓库 SQLite 库读取价格并写出本目录的快照、结果和图表。verify.py 用导出的收盘价重建账户，核对净值、交易费用、借券费用，以及选择日之前可获得的方向排名。核验结果见 [verification.json](results/verification.json)。[独立目录复算日志](replay_log.json)记录了输入库摘要，以及逐项比较的文件与数组。读取CSV时应按字符串读入指数代码，以保留前导零。

## 更新报告

排版使用 Pandoc 和 python-docx。以下版本用于本次交付。第三行从统计表重建初始 Markdown，手工修订源稿后可跳过该行。最后一行用 Windows 桌面 Word 更新目录、页码与图表编号。其他平台可不加 `--word-fields` 生成含原生公式的文档，再在 Word 中更新域。

```shell
python -m pip install pypandoc_binary==1.17 python-docx==1.2.0
python -m pip install pywin32==311  # Windows 桌面 Word 自动化
python experiments/four_pair_tests/build_manuscript.py
python experiments/four_pair_tests/build_word.py --word-fields
```
