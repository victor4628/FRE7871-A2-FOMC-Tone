# 核心文献阅读笔记

阅读日期：2026-09-13。所有页码均为本地 PDF 的物理页码（从 1 开始），不是期刊印刷页码。这里保存文献事实与研究判断，尚未实施作业模型。未读取旧代码、旧 notebook 或旧报告。

## 1. How You Say It Matters — Doh, Kim & Yang (2021)

来源：`reference/How_you_say_it_Matters.pdf`，16 页，全文含尾注、参考文献已读。

- **问题与样本（pp.1–4,9）**：FOMC 声明中的经济展望、风险解释，是否能产生独立于数值决策的信息？样本为 2004-03 至 2014-12 的 87 次例会，有正式声明及 staff 编写的 A/B/C（有时 D）备选声明。
- **标签来源（pp.3–4,15）**：A 较宽松，B 通常代表共识，C/D 较紧缩，存在 D 时取 D。备选全文有五年公开滞后；这是回顾性研究的重要限制，不能把当年尚未公开的备选声明用于实时预测。
- **为什么不是普通词频（pp.4–7）**：宽松和紧缩备选声明的高频词高度重叠；inflation 本身无法给出方向。USE 编码上下文，计算相似度；论文还用分句顺序互换展示同词不同语义。
- **三个量（pp.7–9）**：tone 为相对鹰鸽备选声明的方向；novelty 为与前次正式声明的语义距离；stance 为 tone × novelty。surprise 还需扣除市场预期，其权重利用公告附近债券价格估计，不能把 stance 当成未预期冲击。
- **版本细节（p.8）**：本文描述的市场窗口是公告前后各 10 分钟；不能与 2023 修订论文的 5 分钟 Eurodollar 窗口混用。
- **证据（pp.10–13）**：2007-09，正式声明与 A 均降息 50bp，但因理由/风险不同，正式声明与 C 的 USE 相似度 0.983，高于与 A 的 0.968（C 降息 25bp）。2013-10：USE 对正式声明与 A/C 相似度为 0.895/0.990，词频为 0.975/0.972。
- **相关性（p.13 Table 6）**：surprise 与 Swanson 的 FFR、forward guidance、LSAP 因子相关性为 0.20、0.52、−0.12；这是该版本、该样本的结果。
- **对作业的启发**：保留否定、转折、限定语、数字；分清水平、变化和意外；对关键句进行人工核查。不能把该文的高频结果解释为我们日频回归必然有相同效果。

## 2. Deciphering Federal Reserve Communication — Doh, Song & Yang

来源：`reference/Deciphering_the_Federal_Reserve_Commentary_2020.pdf`，48 页，全文含 online appendix 已读。**实际版本：2023-11-14 修订稿**（封面写 June 2022; updated November 2023），RWP 20-14，不能仅按文件名当作 2020 原稿。

### 方法与可复现要点

- **模型（pp.7–14）**：Transformer 版 USE，512 维；FinBERT 768 维作为比较基准，另比 TF-IDF/LSA。这里的 FinBERT 用于 embedding cosine similarity，不是直接使用 positive 概率。
- **数值微调（pp.11–14,48）**：在 USE 上添加 512 维 ReLU 全连接层。252 个含数字句和 688 个不含数字句，分别形成 63,504 和 473,344 个句对，共 536,848 个训练对。数字损失将 cosine distance 对齐 `2|x1-x2|/(|x1|+|x2|)`，非数字损失保留原模型距离。未提供于本地文件中的完整训练数据和代码仍需另行追溯。
- **例子（pp.13–14 Table 2）**：原始 USE/FinBERT 可能把数值差距方向排错；微调保留定性语义同时改善数值次序。表中数值经过归一化，不能把大于 1 的表值理解成原始 cosine similarity 大于 1。
- **文本标签与现实预期（pp.15–18,20–21）**：以 2011-08 primary dealer survey、FOMC 内部讨论解释备选语言。论文没有假设市场实时看到了保密备选全文，而是假设备选政策边界与市场理解较接近。
- **公式（pp.18–22, Eq.4–14）**，记 `s` 为余弦相似度、`F` 为正式声明、`A/C` 为当次鸽/鹰备选：
  - `Novelty_t = 1-s(F_t,F_{t-1})`
  - `Tone_t = [s(C_t,F_t)-s(A_t,F_t)]/[1-s(A_t,C_t)]`
  - `Stance_t = Novelty_t × Tone_t`
  - `Stance_dove = -(1-s(A_t,F_{t-1}))`；`Stance_hawk = 1-s(C_t,F_{t-1})`
  - 正式 stance 写为两种备选 stance 的凸组合，解出 dovish weight `w_t`。
  - 基准预期边界取 ±Novelty；`MPS_t = Novelty_t × (Tone_t-1+2p_{t-Δ})`。
- **边界注意（pp.19,26 footnote 8）**：tone 落在 [−1,1] 需要相应语义距离关系，不能视作任意余弦向量的恒等性质。作者自己对 2014-09 例外指定最鸽分数；复现必须披露异常处理，不可默默截断。
- **市场识别（pp.22–24,27–29）**：用 12 个月期限 Eurodollar futures、公告附近 5 分钟收益，通过最大秩相关估计 `p` 和尺度，`p` 限 [0,1]。5 个边界事件；该 surprise 包含市场价格信息，不能再以同一价格上的高拟合度当作独立文本预测验证。
- **样本（p.25）**：2004-03 至 2016-12，共 99 次正式声明；排除两次临时公告（2007-08、2008-01）及四次没有备选全文的会议（2005-09、2005-12、2008-08、2009-04）。
- **预期稳健性（pp.25,28）**：Lawrence Meyer 会前 newsletter 的声明草稿，自 2008-09 起用作预期代理；novelty 相关约 0.75，两种 MPS 相关 0.87。这份收费历史材料并不在 reference 中。
- **结果（p.29 Table 5）**：MPS 与 Bauer–Swanson 0.85，orthogonalized 0.74，Bu et al. 0.56，Nakamura–Steinsson 0.81；Swanson 合计 0.73，FFR 0.36，FG 0.69，LSAP −0.20。文本 stance 与这些 surprise 的相关性接近零，进一步说明概念区别。
- **资产回归（pp.30–32 Table 6）**：将冲击尺度标准化为对应一年期零息收益率上升 25bp。股票 10/30/60/90 分钟 β 为 −2.14/−1.81/−3.38/−3.22，R² 0.27/0.14/0.31/0.23；日频 CRSP β −3.99，R² 0.15；生成回归量的标准误用 bootstrap。不能把“25bp”误解为实际每次决策幅度。
- **宏观验证（pp.32–34）**：12 阶月度四变量 proxy-SVAR，工业生产、CPI、excess bond premium、2Y yield，VAR 1973–2019、工具样本 2004–2016；区间宽，需保留不确定性。
- **反事实（pp.34–37）**：固定市场预期、替换个别段落再算 stance。2011-08 全换成 A 会得到很大股票涨幅（8.75%）；作者明确强调仅小幅政策反事实更可信。不能把这些外推数值拿来直接生成作业预测。Table 7 注释截距 0.19 与正文/Table 6 的 0.18 不一致；Table 8 日期标题也应核对，引用时不机械抄写。
- **附录（pp.43–48）**：tokenization、自注意力、训练任务、段落 embedding 加权近似，段落权重非负且和为一；包含跨段落相似度项，并非只做同位置段落比较。

### 我们的判断（不冒充原文结论）

1. 原文的 contemporaneous alternatives 在当前样本末端不可实时获得，完整复制不适合作为 2026 主模型；可以借鉴对照语义、novelty 与预期分解。
2. 文中 embedding 维数与长文本适用性的论述不足以确定模型 token 上限。USE 也不能仅因采用 Transformer 就归入 BERT 家族；实施要依据原始模型文档。
3. 日频 + DGS3MO 控制只检验条件关联，不能承诺识别纯语言因果效应。

## 3. Parsing the Fed (2021 presentation)

来源：`reference/Parsing the Fed.pdf`，34 页，全文文字及全部幻灯片图像已查看。无清晰作者信息；作业称为 2021 presentation。

- **pipeline（p.5）**：collect → clean/parse/tag → factor similarity / word list / FinBERT sentiment → 四资产市场回归。
- **factor similarity（pp.6–10）**：逐句 FinBERT 向量，分别对 “Interest rates will rise” 和 “Inflation will rise” 算 cosine similarity，文档内等权平均，两个分数一起回归。p.8 标题说 distance，但公式是 similarity。模型 checkpoint、pooling、否定识别性能未完整披露。
- **词典（pp.11–16）**：作者从发布会开场白手工构建 topic+direction 的非连续短语表；主题包括 interest rate、economy、job market、sentiment。句内按短语长度与方向合成并取 sign，随后按类别汇总。PDF 仅提供示例，没有完整机器可读词表；不可据此自行补齐后称“现有词典”。公式的匹配指示/分母说明不充分，需要代码或附录才能严格复现。
- **词典回归（p.16）**：2011–2021 press transcript，四指标 R² 分别 24.9%、5.9%、8.6%、23.2%；属于该样本的解释度。
- **FinBERT sentiment（pp.17–24）**：positive probability；可整文句均值、三个 segment 均值、TF-IDF 加权、用词典产生训练信号的微调。弱标签模型和词典并非独立证据；不能用词典标签自测后称人工验证。
- **结果（pp.25–28,32）**：原始 FinBERT 2016–2021 为 47 份声明，R² 3.0%、11.2%、0.9%、3.7%；加权后 7.1%、14.8%、1.0%、6.2%。微调 2011–2021 是 10.3%、0.3%、10.6%、9.8%。跨页/跨时期的数字不能直接当同样本优劣比较。
- **加权公式（p.31）**：词权重 `(1+log tf)/(1+log document_word_count) × log(N/df)`（tf≥1）；句权重怎样由词权重汇总仍未充分披露。预测应用 IDF 必须只在训练期估计。
- **样本与表格疑点（pp.10,32–33）**：factor similarity 的 G−V R² p.10 为 0.216，p.32 两时期为 0.237/0.015，不能自行推定同一规格。p.33 将部分“one-day”窗口写为前一日收盘到次日收盘，存在与作业单日定义不同/歧义的地方。
- **文献准确性（p.3）**：把 2011 LM 与 BERT 改善联系起来是年代错位，不采信这句话；必须回查一手资料。
- **适用性**：很贴合作业指标，可借鉴结构；不是严密可复现论文，也未展示独立时间外预测或作业要求的 DGS3MO 控制。不能把情绪概率当鹰派概率。

## 4. Bloomberg Intelligence (September 2021)

来源：`reference/Bloomberg Intelligence BI Sentiment Index Methodology.pdf`，14 页。扫描图像为主，全部页面已视觉阅读，pp.3–9 为研究正文与参考文献；其余为封面、目录、免责声明、服务介绍。作者 Alex Montiel、Ira Jersey、Angelo Manolatos。

- **数据（p.4）**：1993 至 2021-04 的 minutes；只取 Participants’ Views on Current Conditions and the Economic Outlook、Committee Policy Action。市场目标为 FF4–FF1 期货价差与 3M/5Y Treasury curve，minutes 发布日对前一交易日收盘变化。不能把会议日期当公开日期。
- **主题选择（pp.5–6）**：noun phrases → TF-IDF → 四个 LASSO 线性模型（两种市场变化及各自正负二元目标），用非零系数提取重要主题，平均绝对系数作主题权重。
- **确实使用的现有词典（p.6）**：Loughran–McDonald 金融正负词典。只在重要主题句中计数 positive−negative；再乘主题权重与共识强度（例如 some/all participants 的权重），汇总句均值。并不是公布了一份独立“Bloomberg 鹰鸽词典”。
- **平滑（p.7）**：8 期指数移动平均，对应约一年会议；只能用于趋势展示，不能当即时原始事件分数。
- **结果（p.8）**：与 federal funds futures 变化相关 0.45，与 3M/5Y curve 相关 −0.29，与 monetary policy index 差分相关 0.40。比较使用八期移动平均；不能与单事件未平滑预测性能比较。
- **局限（pp.6–8）**：完整 topic 清单、LASSO 系数、共识映射权重未公布；作者说用了全部数据并建议未来 rolling window。若照抄全样本市场监督选主题再做同样本回归，会有循环验证/过拟合风险。LM 反映经济情绪，需要主题语境才能映射政策倾向，不能单独当 monetary-policy lexicon。
- **对作业的价值**：章节筛选、participants 共识语气、金融词典来源、趋势平滑可借鉴；完整模型不能仅凭这份 brochure 声称复现。
