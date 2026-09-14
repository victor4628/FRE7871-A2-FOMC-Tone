# FOMC NLP 研究阅读档案

本目录保存 2026-09-13 的独立阅读与方案设计。未参考仓库中的旧 notebook、旧代码、旧报告或旧方法说明。

## 完成情况

- 本地 PDF：12/12，238 页，全文阅读完成；扫描/图表型页面已视觉核查。
- 本地 Excel：1/1，6 个工作表、19,748 条记录，完整只读扫描完成。
- 补充核查：ACL 2023 论文、作者代码库、模型访问条件、现任 Chair/FOMC 日历、FRED 与 ETF 指标定义。
- 当前阶段：研究与拟议方案完成；尚未开始最终数据采集、模型运行、notebook 或 PDF。

## 笔记索引

1. [核心 FOMC 文献](01_core_readings.md)
2. [FinBERT 与 BERT](02_finbert_and_bert.md)
3. [Topic model、声明对照与 Excel 审计](03_topic_comparisons_and_reference_data.md)
4. [方法选择与拟议实施方案](04_method_choice_and_proposed_plan.md)

## 当前建议

主要比较采用：

1. Gorodnichenko–Pham–Talavera 已有政策词典（透明基准）；
2. ACL 2023 FOMC-RoBERTa（FOMC 鹰鸽监督模型）；
3. ProsusAI/finbert（用户要求正式纳入的逐句金融情绪模型）。

三种方法均纳入趋势比较、Table 2 和 Table 3。FinBERT 主要分数为逐句 `P(positive)−P(negative)` 的均值，另保留 positive-probability 均值与《Parsing the Fed》对照；不将 positive/negative 直接改名为 dovish/hawkish。主题模型只用于解释主题来源，不加入主要 tone 回归。
