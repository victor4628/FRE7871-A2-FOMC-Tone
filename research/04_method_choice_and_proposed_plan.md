# 方法选择与拟议实施方案

日期：2026-09-13。此方案结合全部本地 reference 及补查的一手论文、作者代码库、模型卡和官方数据说明。尚未开始生成最终数据、模型结果、notebook 或 PDF；等待方案确认。

## 1. 方法选择

### 方法 A：现有货币政策词典（透明基准）

采用 Gorodnichenko, Pham & Talavera (2021/2023) 的公开规则，经 Shah, Paturi & Chava (ACL 2023) Table 1 完整列示并在 FOMC 三分类数据上检验。

- 原样使用四组 noun/direction terms 和一组 negation phrases，不自行增词、删词或按最终回归结果调词。
- 按原文组合规则逐句标 hawkish/dovish/neutral；保留句级匹配词、否定翻转及无法判定原因，文档分数为 `(hawkish句数−dovish句数)/target句数`。
- 同时报告 coverage：target sentences / eligible sentences。低 coverage 的文档不把 0 解释成真正中性。
- 这份规则在 ACL 论文中的 weighted F1 约 0.50–0.54，定位是透明、可审计的基准，不包装为高准确模型。
- Apel–Blix Grimaldi 词典可做小型稳健性检查，但它最初为 Riksbank/ECB 语境设计，且公开复现通常是约 25/26 个正负 stems。不会将 Loughran–McDonald 单独当鹰鸽词典；LM 是金融正负情绪词典，需政策主题语境才能解释。

主要来源：

- [Shah, Paturi & Chava (2023), ACL paper](https://aclanthology.org/2023.acl-long.368.pdf)，Table 1、Table 5。
- [作者代码库](https://github.com/gtfintechlab/fomc-hawkish-dovish)。
- [Apel & Blix Grimaldi (2012)](https://www.econstor.eu/bitstream/10419/81866/1/715517821.pdf)。

### 方法 B：FOMC-RoBERTa（主要语言模型）

使用 `gtfintechlab/FOMC-RoBERTa`，即 ACL 2023 的 RoBERTa-large，直接以 FOMC 句子标注为 dovish/hawkish/neutral，而不是把金融情绪标签改名。

- 作者标签为 `LABEL_0=Dovish, LABEL_1=Hawkish, LABEL_2=Neutral`；实施时仍从模型 `config.id2label` 读取并固定 revision，避免顺序错误。
- 论文原始样本：minutes 1,070 句、press conference 315 句、speeches 994 句；两位独立标注者，split-data agreement 约 89.0%–95.0%。
- combined 数据的 RoBERTa-large weighted F1 为 0.7171，时间切分 1996–2019 train、2020–2022 test 的 F1 为 0.7114。press-conference 单独测试约 0.55，不能掩盖这一弱点。
- 文档聚合遵循作者方法：逐句预测，以 `(hawkish−dovish)/eligible sentences` 得到 level。另计算相对上次同类型文档的 tone change 和语义 novelty；这两个派生量分别命名，不与 level 混用。
- 对模型遗漏或概率接近的句子做人工错误审计，按 ACL 论文已公开的 annotation guide 判断，不创造新词典。
- 模型目前在 Hugging Face 是 gated、CC BY-NC 4.0；下载前需要登录并接受共享联系信息的条件。课程研究用途与非商业许可一致，但访问步骤需要账户完成。

主要来源：

- [ACL 2023 论文](https://aclanthology.org/2023.acl-long.368/)。
- [官方代码与标签说明](https://github.com/gtfintechlab/fomc-hawkish-dovish)。
- [FOMC-RoBERTa 模型页](https://huggingface.co/gtfintechlab/FOMC-RoBERTa)。

### 方法 C：ProsusAI/finbert（正式纳入比较的金融情绪模型）

根据用户补充要求，FinBERT 改为必做的第三种比较方法，采用作业明确举例的 `ProsusAI/finbert`，逐句 sentiment scoring。与方法 A、B 一起用于趋势比较、Warsh 发布逐项展示和四个资产回归；不再限于可选附录。

- 使用预训练模型及匹配的 tokenizer，固定模型 revision，从配置核对 positive、negative、neutral 的映射。官方模型卡说明其在 Financial PhraseBank 上做金融情绪微调：[ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)。
- 每句保留三个类别的概率。主要文档指标 `finbert_sentiment = mean(P(positive) − P(negative))`，取值 [−1,1]。在同一组纳入分析的句子上与方法 A、B 比较，明确分母和 coverage；无有效句时标为缺失。
- 另保存 `mean(P(positive))`，用于与本地《Parsing the Fed》的 positive-probability 用法直接比较；这是同一方法的辅助规格，不额外算作第四种方法。
- 保留完整句子的否定、条件、转折与数字；长文逐句处理，不只取整篇前 512 tokens。超长单句的拆分规则与覆盖情况单独记录。首轮使用现成 checkpoint，不先做无依据的微调或 TF-IDF 加权。
- 正值代表金融情绪较正面，负值代表较负面；不预设与政策鹰鸽存在固定正向或反向映射。先比较与词典/FOMC-RoBERTa 的相关性，检查代表性分歧句，并比较同一样本上的市场解释力。
- Table 2 同时列出两种 hawkishness 与 FinBERT sentiment；Table 3 包含三种主要分数各对四项资产的回归。即使 FinBERT 效果较弱，也完整报告。
- ACL 2023 将 positive→dovish、negative→hawkish 直接映射后的 combined F1 约 0.425，只说明该任务/映射的局限，不能提前判定 FinBERT 对本作业的市场回归无价值。最终报告区分“鹰鸽测量是否准确”与“情绪是否解释市场反应”。

## 2. 文本样本与清洗

### 样本边界

- 观察期：2018-02-01 至 2026-09-13 23:59 ET。预测目标是 2026-09-15/16 FOMC meeting，任何截止时间后的材料不得进入预测特征。
- 现实核查：Federal Reserve 官方资料显示 Kevin Warsh 于 2026-05-22 就任 Chair；最近完成的 FOMC meeting 是 2026-07-28/29，下一次为 2026-09-15/16。因此作业中的 Warsh 设定与当前官方事实一致。
- 主席分期：Powell（样本起点至 2026-05-21）与 Warsh（2026-05-22 起）。Warsh 样本很短，均值比较和 Table 2 以描述性证据为主，不用显著性包装小样本。

官方核查：[Warsh biography](https://www.federalreserve.gov/aboutthefed/bios/board/warsh.htm)；[FOMC calendar](https://www.federalreserve.gov/monetarypolicy.htm)。

### 文档类型

1. FOMC statements：公告发布日期和 2:00 p.m. ET 时间；正文，排除投票名单、联系方式和网页导航。
2. Minutes：以**公开发布日期**作事件日，不用会议日期；保留经济展望与政策讨论，排除目录、行政性项目和重复附件。
3. Chair speeches/testimony：只收 Chair 本人的正式讲稿、国会证词；记录官方页面发布时间。问答只有官方逐字稿存在时另行标记。
4. Press conferences：prepared remarks 与 Chair answers 分开；记者问题作为上下文保留但不计入 Chair tone。这样避免记者的鹰鸽措辞进入主席分数。

每份文档保存 URL、标题、document type、speaker、release date/time、meeting date（如适用）、Chair、原始文本 hash、清洗后句数。数据抓取代码公开，但遵照作业要求不把原始数据文件提交 GitHub。

## 3. 分数、比较与验证

每份文档保留三种主要分数，并区分各自含义：

- 方法 A、B 的 `tone_level`：文档本身的 hawkishness；`tone_change` 是相对前次同类型文档的变化。
- 方法 C 的 `finbert_sentiment`：金融情绪水平；其前后差分表示情绪变得更正面或负面，不自动称为更鹰或更鸽。
- `novelty`：可选的同类型文档文本变化指标，独立于三种分数；不冒充 surprise，也不使用未经验证的 FinBERT 原始向量余弦作为政策距离。

Figure 1 按文档类型展示方法 A、B 的 tone level，标出 Warsh 上任日；FinBERT sentiment 在独立 panel 展示，使用金融情绪坐标标签，同样比较 Powell 与 Warsh。差分及平滑曲线作辅助。两份 2023 对照 PDF 用于检查政策行动、经济展望等局部信号，不预先规定整篇声明的鹰鸽排序。

验证分三层：

1. 核实两个语言模型的标签映射、训练数据和 checkpoint 来源；仅在确认未用于该 checkpoint 训练的数据上报告独立测试指标，不能直接把作者公开 test split 当成发布模型的独立测试集。
2. 用论文给出的典型句、两份声明 redline，以及否定/条件/数字句做预先列明的 sanity checks。
3. 随机抽查近期各类型句子和所有 Warsh 文件中的模型分歧句；人工判断只用于报告误差，不回头改词表或按市场结果调模型。

## 4. 市场变量与事件对齐

按题目指定口径重新取数：

| 因变量/控制 | 数据源 | 变换 |
|---|---|---|
| DXY | Yahoo Finance `DX-Y.NYB` | 事件交易日 close-to-close % return |
| 10s2s | FRED `T10Y2Y` | 事件日变化，bp |
| 1Y yield | FRED `DGS1` | 事件日变化，bp |
| Growth−Value | Yahoo Finance IWF、IWN | `return(IWF)−return(IWN)`，百分点 |
| 3M control | FRED `DGS3MO` | 事件日变化，bp |

IWF 跟踪 Russell 1000 Growth（large/mid cap），IWN 跟踪 Russell 2000 Value（small cap）。所以题目的 Growth−Value 同时混入 size exposure，报告中会明确说明，不能将系数解释为纯 growth/value 因子。

官方产品说明：[IWF](https://www.ishares.com/us/products/239706/)；[IWN](https://www.ishares.com/us/products/239712/ishares-russell-2000-value-etf)；[FRED T10Y2Y](https://fred.stlouisfed.org/series/T10Y2Y)。

事件规则预先固定：市场开盘前或 4:00 p.m. ET 前发布，用当日收盘相对前一交易日；4:00 p.m. 后、周末或假日发布，使用下一交易日。Statement 与同日 press conference 的日频反应无法拆开，因此不在同一回归中当两个独立冲击。

## 5. 回归设计

对三种主要文本分数和四个市场变量估计基准式，至少形成 12 个“方法 × 指标”对应结果。方法 A、B 使用 hawkishness，方法 C 使用 finbert_sentiment，分别回归；三种方法在同一规格下使用共同有效事件样本，便于公平比较：

`ΔAsset_i = α + β Tone_i + γ ΔDGS3MO_i + ε_i`

此处 `Tone` 是文本分数的通用占位符；FinBERT 回归系数解释为金融情绪变化的关联，不改称鹰派冲击。

- `Tone` 先用样本内标准差标准化，使 β 表示 tone 增加 1 SD 的资产变化。
- 主规格按 document type 分开估计，避免 statement、minutes、speech 的信息集和发布时间混在一起。
- 另给 pooled 描述性规格，加入 document-type fixed effects 与 Chair indicator；同日多文档的标准误按事件日期聚类。小样本下报告 HC3，并用 leave-one-event-out 检查是否由单次危机事件驱动。
- Table 3 报 β、standard error、p-value、N、adjusted R²；正文强调日频关联无法识别纯文本因果，也无法完全剥离同日政策行动和宏观新闻。
- Warsh 文件逐项列入 Table 2，但不会单独跑样本极小的 Warsh 回归。

## 6. 预测与最终交付

预测只使用 2026-09-13 截止信息：

- cut/hold/hike 三项概率合计 100%；记录基准市场概率来源，再用最近 Warsh communication、通胀/就业数据和模型分数解释调整。
- 声明比 2026-07-29 更鹰的概率，围绕 `tone_change` 定义。
- 四个资产分别给上涨概率和条件期望变动，单位与历史回归一致。
- 一笔 position 写明方向、规模逻辑、持有窗口及能证伪观点的结果；不把样本内拟合当确定性交易建议。

交付结构：一个可从头运行、保存输出的 notebook；`AI_USE.md`；短 PDF，含作业指定 Table 1、Figure 1、Table 2、Table 3、文献比较和预测。PDF 使用 Roc Investments forest green `#254C45` 作标题、表头与图形强调色。GitHub 只提交代码、notebook、报告与必要说明，不提交原始数据。

## 7. 已知风险及处理

- FOMC-RoBERTa 需要 Hugging Face gated access；若无法访问，不静默换成普通 FinBERT。备选顺序是：先由用户完成授权；仍不可用时，从作者公开标注 splits 微调相同 RoBERTa-large 并完整披露与原 checkpoint 的差别。
- press-conference F1 较低，必须单独报告，不用 overall F1 掩盖。
- speeches 的发布时间、盘后发布和同日重复事件会改变收益对齐，所有规则在看回归前冻结。
- 2020 疫情期等极端事件可能主导日频结果，采用 leave-one-event-out 及危机期指示作为稳健性，不删除不利观察。
- reference Excel 截止 2024 且资产不匹配，仅作交叉检查。
