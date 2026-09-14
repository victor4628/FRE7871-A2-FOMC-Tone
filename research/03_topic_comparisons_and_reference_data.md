# Topic model、声明对照与参考数据审计

阅读日期：2026-09-13。页码均为本地 PDF 物理页码。此文件记录剩余本地材料的完整阅读结果，不使用旧项目代码或旧报告。

## 1. Kherwa & Bansal (2019), Topic Modeling: A Comprehensive Review

来源：`reference/Topic_Modeling_A_Comprehensive_Review.pdf`，17 页，全文含参考文献已读，p.8 表格已视觉核查。

- **定位（pp.2–6）**：综述 LSA、NMF、PLSA、LDA。LSA 以 term-document matrix 做 SVD；NMF 施加非负约束；PLSA/LDA 是概率生成模型。LDA 假设每篇文档由多个主题混合产生、每个主题是词分布。
- **共同限制**：经典方法多为 bag of words，忽略词序，因此不会自然识别否定、条件句或“高通胀/低失业”在政策语境中的不同方向。
- **推断（p.6）**：variational inference 较快但带近似偏差；Gibbs sampling 更慢。算法选择不是本作业的主要经济问题。
- **文中实验（pp.7–9）**：电影评论 5,000 条、NIH grants 仅 100 条；作者预先设定 K=10，随后因 NIH 样本小改为 K=5。其 coherence 表显示结果依数据与 K 改变，不能据此声称 LDA 普遍优于 LSA。
- **不适用的预处理（p.7）**：作者删除所有数字；货币政策文本中利率、通胀和数量变化是核心信息，本作业不能照用。
- **扩展（pp.9–10）**：supervised、correlated、hierarchical、dynamic、syntactic topic models分别处理标签、主题相关、层次、时间演化和词序问题。
- **评估与挑战（pp.12–13）**：可解释性、可视化、计算成本、随机初始化不稳定、主题数和先验选择均未解决；held-out likelihood 与人类可解释性不必一致。
- **本作业用途**：可用预先定义的经济主题给句级鹰鸽分数分组，解释“分数来自通胀、就业还是政策行动”。不建议把无监督 LDA 加入主要 tone 估计或回归，否则增加一个需要调参、稳定性检验且没有鹰鸽方向的模型。

## 2. 两份 2023-07-26 与 2023-09-20 声明对照

来源：`reference/FOMC_Statement_comparison.pdf` 与 `reference/FOMC_Statement_Comparison_Strikethrough.pdf`，各 1 页；全文及图像已读。

- 并排版本将 2023-09-20 新文字标黄；strikethrough 版本以删除/新增颜色展示差异。
- 有经济描述变化：activity 从 `moderate` 改 `solid`；jobs 从 `robust` 改为 `slowed ... but remain strong`。
- 有实际政策行动变化：July 的 `raise ... to 5-1/4 to 5-1/2` 改为 September 的 `maintain ... at 5-1/4 to 5-1/2`。
- 其余政策指引、资产负债表缩减和风险段落基本相同；投票名单新增 Adriana D. Kugler。
- **测量启发**：全文水平分数可能被大量不变 boilerplate 淹没。必须同时报告全文 stance 与相对前次声明的 change/novelty，并将名单变化和程序性文本排除。`maintain` 不应自动等同 dovish；它相对前次加息是更鸽，但作为政策水平可能仍偏紧。

## 3. `FOMC_Data_2011_2024.xlsx` 只读审计

六个工作表、19,748 条数据记录全部扫描；没有公式、图表、批注、超链接或 Excel 错误值。所有表按日期倒序、无重复日期、日期连续性符合各市场交易日来源的差异。

| 工作表 | 记录数 | 日期范围 | 主要内容 |
|---|---:|---|---|
| GT10 | 3,330 | 2011-12-30 至 2024-10-04 | 10Y yield，B/E 与 C/F、D/G 两组完全重复 |
| GT2 | 3,330 | 2011-12-30 至 2024-10-04 | 2Y yield，同样存在重复列 |
| 2s10s_Spread | 3,330 | 2011-12-30 至 2024-10-04 | 10Y−2Y，单位约为 bp |
| Gold_Prices | 3,318 | 2011-12-30 至 2024-10-04 | gold price |
| VIX | 3,229 | 2011-12-30 至 2024-10-04 | VIX |
| SP500 | 3,211 | 2011-12-30 至 2024-10-04 | S&P 500 |

- 各表只有最近 1,099 条记录填有 Change 与 % Change；更早记录仍有 level，但变化列为空。变化值和相邻交易日 level 能逐条对上。
- 2s10s spread 与 `100 × (GT10−GT2)` 的 3,330 个共同日期全部在 0.11bp 内，偏差来自显示/数据精度。spread 跨过零时，文件的 `% Change` 使用前值绝对值作分母；这种百分比对跨零利差缺乏经济可解释性，正式作业应使用 bp 变化。
- 文件截止 2024-10-04，且没有 DXY、DGS1、DGS3MO、IWF 或 IWN，也没有 FOMC 文本与发布时间。它不能覆盖本次作业要求的市场变量和 2026 预测时点。
- **处理决定**：仅作为历史收益计算和 10s2s 单位的交叉检查；主数据从题目指定的一手来源重新拉取。最终 GitHub 不提交原始数据文件，遵守作业要求。

完整机器审计结果保存在临时文件 `tmp/literature_text/workbook_audit.json`，不属于最终交付物。
