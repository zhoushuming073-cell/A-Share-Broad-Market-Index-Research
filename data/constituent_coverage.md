# 六条指数成分股覆盖说明

截至2026-11-18。数据库为 `a_share_indices.sqlite`，本页描述已实际落库的成分股资料。文件日期是其有效观察日期，抓取日期见 `constituent_source`。这些快照之间的日期不能自动填充为逐日真实成分。

| 指数 | 已取得历史阶段名单 | 最新官方名单 | 已取得权重 |
|---|---|---|---|
| 国证1000 399311 | 暂无可复核的完整历史阶段文件 | 2026-10-30，998只，含权重 | 2026-10-30快照，百分比四舍五入合计100.18% |
| 中证A100 000903 | 暂无 | 2026-11-17，100只 | 2026-10-30，100只，合计99.999% |
| 上证100 000132 | 暂无 | 2026-11-17，100只 | 2026-10-30，100只，合计99.998% |
| 沪深300 000300 | BaoStock在2015-01-05及2015—2025年每次6月、12月定期调样后交易日共23份观察快照，每份300只 | 2026-11-17，300只 | 2026-10-30，300只，合计99.99% |
| 上证指数 000001 | 暂无 | 2026-11-17，2238只 | 2026-10-30，2233只，合计100.01% |
| 中证全指 000985 | 暂无 | 2026-11-17，5117只 | 2026-10-30，5117只，合计100.045% |

**历史资料缺口很大。** 沪深300的23份是定期调样后观察快照，并不证明两个观察日之间没有临时调整。BaoStock原始字段 `updateDate` 有时早于查询日，已在逐份CSV中保存，不能将查询日一概视为公告生效日。其余五条指数2015—2025年的阶段名单尚未取得。所有指数均无连续历史权重，因此这些资料目前不能用于全样本日度抵消率计算。

国证1000的国证官方 `download-adjustment` 接口在本次探索时曾返回约18,592条调整记录，覆盖2021-12至2026-10；正式归档时连续返回“样本历史调样文件不存在”，所以**未把先前未保存的预览结果写入数据库**。`scripts/build_constituent_db.py`保留重试入口；成功时应检查每阶段成员数和原始文件校验值后再使用。官方历史数据服务也可能需要授权取得。

数据库表：`constituent_source`记录URL、抓取时间、文件路径、SHA-256；`constituent_snapshot`保存各时点全名单、来源类别、可得权重；`constituent_adjustment_record`预留国证调样原始记录，目前为空。中证和国证官网原始Excel文件在 `raw_constituents/`，BaoStock观察快照CSV也逐份保存。中证官网文件的成分日与权重日并不相同，不能直接拼为同日成分权重。

质量检查：SQLite `integrity_check=ok`；23份沪深300历史观察各300只。当前权重和与100%的小幅差异反映文件中的逐股权重取整，国证1000 998只名单的成因仍待官方确认。2026-10-30与2026-11-17文件中的成分数量变化可能与期间调样、上市及退市有关，不能仅凭此归因。

查询例：

```sql
SELECT as_of_date, security_code, security_name
FROM constituent_snapshot
WHERE index_code='000300'
  AND snapshot_kind='baostock_rebalance_observation'
  AND as_of_date='2015-06-15'
ORDER BY security_code;
```

来源：[中证指数公开成分文件](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/000300cons.xls)、[中证指数公开权重文件](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/closeweight/000300closeweight.xls)、[国证指数数据服务说明](https://www.cnindex.com.cn/zh_services/data_services/index.html?act_menu=null&act_service=2)、[BaoStock历史沪深300成分查询接口](http://baostock.com/baostock/index.php/首页)。每条实际入库数据的确切来源以 `constituent_source` 表为准。
