# 双指数相对收益速度实验

## 冻结的执行方案

复用 four_pair_tests_v2 的收盘、seven_index_convergence_v2 的长样本中证全指、relative_return_frame_deletion_v1 的每日差额和正负排名，与 SQLite 行情对账。CH3/CH4 工作簿核对市场超额简单收益，以 log(1+mktrf+rf_dly) 重建市场总简单收益的对数形式，同时保留超额因子原列作敏感性。研究对象始终只有沪深300与上证综指。

先核对输入，再以沪深300减上证综指的日对数收益差为速度。固定日历置零用于路径，回归跳过中和日，同号游程遇零或缺失即终止。局部转折用前后斜率变号识别；需要下一交易日确认。按方向、K和窗口输出所有候选。HAC区间为逐项95%区间。随机游走斜率以高斯似然估计过程方差，允许零方差边界。全期估计用于历史描述，条件贡献使用更新前斜率；样本外在扩展训练集上每63日重新估计超参数，每日更新状态。固定模型每日扩展。已实现的市场收益只用于同期归因。

同步循环移动区块在零平均差基准下重抽联合收益，逐样本重排关键日和搜索各窗口极值。等量随机日期在未改动历史上抽取。多重检验按基准与统计量对不同方向、K、窗口联合校正。预设60日主窗、20日主区块，5及60日作敏感性。统计单位为交易日；区块内保留原顺序。全样本残差检验须说明拟合截距的机械约束，另报告顺序验证期误差。模型增量须能在顺序验证中降低损失；区间内同号积累只构成描述证据。

## 运行

从本目录执行 python run_all.py，再执行 python -m unittest discover -s tests。依赖 numpy、pandas、numba、scipy、statsmodels、matplotlib、openpyxl、pyyaml、tabulate。输入路径相对于仓库，所有新文件留在本实验目录。results 内模型收益按日对数小数计，带 bp 的字段已转换为基点。figures 内保存PNG，绘图可由入口重建。

## 结果索引

- inputs、source_audit 为输入及归档校验。daily 含全部39种组合的日期掩码、滚动速度和窗口和、OLS及滤波状态。peaks、turns 记录全局极值与局部转折。
- runs 逐段记录日差、动态残差及平滑速度。run_summary 中交易日长度与区间数量分列。对于平滑速度，contribution_bp 为平滑序列之和，窗口重叠，不能解读为原始财富。
- models、regimes、annual_changes、decomposition、gradient_diagnostics 对应市场模型、方向、年度暴露差、归因及剩余集中度。models 中常数均值模型的调整R²无解释意义。
- simulations 的block=0是等量随机日期，其余为联合循环区块。每组10,000次，保存观察量、模拟分位数、尾概率和Holm值。residual_null 为附加的固定斜率区块残差换号检验，每种区块10,000次。其beta行是零模型参考范围，不作斜率显著性检验。
- oos、oos_comparison、validation、primary_hypotheses 为历史可得参数下的条件归因误差和主检验；residual_band 为图10的逐窗区间。CSV.GZ可由pandas直接读取。
