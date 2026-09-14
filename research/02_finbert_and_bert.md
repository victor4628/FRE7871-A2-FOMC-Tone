# FinBERT 与 BERT：模型身份、任务及使用边界

阅读日期：2026-09-13。页码均为本地 PDF 物理页码。以下四篇全文、附录及参考文献已读；额外查看了长篇论文 pp.65–66 代码截图和 Araci p.10 混淆矩阵。尚未下载或运行模型。

## 1. Araci (2019), FinBERT: Financial Sentiment Analysis with Pre-trained Language Models

来源：`reference/FinBERT_Intro.pdf`，11 页，arXiv:1908.10063v1。

- **身份与任务（pp.1–5）**：在 BERT 上继续金融领域预训练，再微调金融句子情绪分类。标签反映文本对所述公司的股票价格影响；不是 FOMC 鹰鸽标签。
- **语料（p.5）**：Reuters TRC2 金融子集 46,143 篇、约 29M words、400k sentences，2008–2010；Financial PhraseBank 4,845 句，positive 28.1%、negative 12.4%、neutral 59.4%，16 名标注者。文献版本间 4,845/4,840 计数需分别记录。
- **验证（pp.5–6）**：20% test，再从剩余取 20% validation；文中实验还报告十折。FiQA 1,174 条连续情绪值，作者自行十折，与别人官方 test 结果不完全可比。不是跨时间预测。
- **架构（pp.4–6）**：最后层 `[CLS]` 接分类层和 softmax，回归改 MSE。全部参数可微调，使用 gradual unfreezing、discriminative learning rates、slanted triangular schedule 等减轻遗忘。微调标签决定输出的经济含义。
- **实验参数（p.6）**：最大长度 64、learning rate 2e-5、batch 64、6 epochs、dropout 0.1、warmup 0.2。这里的 64 是该实验设置，不是 BERT 固有限制。类别加权交叉熵、macro-F1 应对不平衡。
- **结果（pp.7–9）**：全数据 accuracy 0.86 / macro-F1 0.84，全标注一致子集 0.97/0.95；FiQA MSE 0.07、R² 0.55。Table 4 vanilla BERT 0.85/0.84，domain adaptation 0.86/0.84，增益很小。作者在结论明确表示，无法断定继续金融预训练在本实验显著优于不做。
- **误差（pp.9–10）**：仍会误判“亏损由 2.2m 降到 0.3m”等数字比较。p.9 错误类别叙述有疑似笔误；p.10 混淆矩阵及解释显示 positive/neutral 区分是主要问题，不机械引用那句文字。
- **使用判断**：可作金融情绪基准。不能把 positive 直接改名 hawkish，也不能把金融情绪 accuracy 当 FOMC accuracy。

## 2. Yang, Uy & Huang (2020), FinBERT: A Pretrained Language Model for Financial Communications

来源：`reference/FinBert_Intro1.pdf`，5 页，arXiv:2006.08097v2。

- **另一条 FinBERT 路线（pp.1–2）**：HKUST 团队，论文链接 `yya518/FinBERT`；与 Araci 模型不应混称为同一 checkpoint。
- **预训练（pp.2–3）**：4.9B tokens：10-K/Q 2.5B、earnings calls 1.3B、analyst reports 1.1B。60,490 份 10-K 与 142,622 份 10-Q（1994–2019）；136,578 份 calls（2004–2019）；488,494 份 analyst reports（1995–2008）。这些具体数量与 2022 长篇版本不同。
- **词表**：cased FinVocab 28,573、uncased 30,873，和通用词表约 41% 重合。这里是 tokenizer vocabulary，**不是用于鹰鸽打分的 word list**。
- **训练**：BaseVocab 继续 BERT 250k steps；FinVocab 从头 1M steps。长度先 128 后 512，4 个 P100，batch 128；这不是本作业应重复的训练规模。
- **下游数据（p.3）**：PhraseBank 4,840；AnalystTone 10,000（3,580 positive、1,830 negative、4,590 neutral）；FiQA 1,111，连续标签转二分类。随机 90/10 划分重复十次，报告 mean accuracy。
- **结果（p.4）**：uncased FinVocab 在三个数据集 accuracy 0.872/0.844/0.887，uncased BERT 0.835/0.730/0.850。FinVocab 相比 BaseVocab 增益有限且依任务而变；不能用这些数字推断 FOMC 鹰鸽性能。

## 3. Huang, Wang & Yang (2022), FinBERT: A Large Language Model for Extracting Information from Financial Text

来源：`reference/FinBERT_a_Financial_Language_Model.pdf`，74 页，最终稿 September 2022，Contemporary Accounting Research。全文含 online appendices 已读。

### 训练、标签与准确率

- **预训练（pp.13–15）**：同为 4.9B tokens；该版本为 476,633 份 analyst reports（2003–2012），其余 filings/calls 时间与上述路线相近。不要把两个版本的数据计数拼接。
- **标注（pp.15–16）**：10,000 个 analyst-report sentences，3,577 positive、4,586 neutral、1,837 negative；8,100 train、900 validation、1,000 test，随机划分。部分句子在无标签预训练语料出现过，作者披露这一点，并做 PhraseBank 检验。
- **公平比较（pp.16–21,50–55）**：LM 基准规则为遇 negative 则负，否则遇 positive 则正，否则中性；按净计数替代准确率接近（62.2% vs 62.1%）。FinBERT accuracy 88.2%、macro-F1 87.8%；BERT 85.0%/84.2%；LM 62.1%/58.1%。FinBERT 的 89.7% 是 negative recall，不能改称总准确率。
- **小样本与语序**：10% 训练数据时 FinBERT accuracy 81.3%，BERT 62%；随机打乱测试词序后 FinBERT 76.9%，下降 11.3 个百分点，BERT 下降 14.8 个百分点。这支持语境与领域迁移的重要性，不等于验证了央行标签。
- **其他任务（pp.25–26）**：ESG 四类各 500，FinBERT 89.5% vs BERT 87%；同样需要相应人工标签。

### 长文本与市场验证

- **文本聚合（pp.27–30,49）**：对财报电话会议中管理层发言逐句分类，presentation 与 Q&A 都包括，排除分析师问句；文档 tone 为 `(positive句数-negative句数)/全部管理层句数`，再标准化。不能只截取开头 512 tokens 代表整篇。
- **市场样本**：28,873 次 calls、712 家公司、2003–2020；三日 CAR `[-1,+1]` 为收益减市场，控制基本面，季度固定效应，标准误按公司聚类，winsorize 1%。这不是 FOMC 公告日设计。
- **结果（pp.60–61）**：每一标准差 tone 对 CAR 的系数，FinBERT 0.907 百分点、BERT 0.873、LM 0.640、LSTM 0.743；adj.R² 0.080、0.079、0.069、0.073。用 5,000 次 bootstrap 和 Vuong 比较。不能把 88.2% 分类准确率写成市场预测准确率。
- **原文提醒（pp.27–28）**：市场回归是“文本测量是否好”及“市场是否定价”的联合检验；回归不显著未必说明模型失效，显著也不能替代人工标签验证。
- **Online Appendix（pp.70–71）**：只取头/尾 512 tokens，系数 0.301/0.243，adj.R² 0.062/0.061；逐句计数 0.907/0.080，逐句 `P(pos)-P(neg)` 均值 0.893/0.079。两种逐句聚合接近，是该样本证据，不是普遍定理。

### 实施细节与身份核对

- **附录（pp.38–48）**：保留原始句子语序、数字和 stopwords；匹配 tokenizer 与模型。FinVocab uncased 30,873，频率阈值 8,500；BaseVocab 重合 12,498。下游 5 epochs、lr 2e-5、batch 32、最大 512、全部参数微调。金融 tokenizer 不是情绪词典。
- **不同模型迁移（p.41 footnote 43）**：Araci 的已有模型直接迁到本文 analyst-report test 的 accuracy 74.25%，低于本文专门微调的 88.2%；标签语料匹配比只看“FinBERT”名称重要。
- **教程截图（pp.65–66，已视觉核查）**：加载 `yiyanghkust/finbert-tone`，对应 tokenizer；logits 经 softmax；此 checkpoint 例示顺序为 `0=neutral, 1=positive, 2=negative`。这是该模型的顺序，不能套给 ProsusAI 或别的 checkpoint。实施读取 `id2label`，固定版本。
- **例句（pp.67–68）**：LM 容易把中性金融词和否定语境误作情绪；说明不得在 Transformer 前去掉否定词或破坏句法。
- **附录笔误（p.74）**：概率聚合例子将两个概率都写 negative，显然有标签文字错误；依据明确公式和模型配置解释，不照搬笔误。

## 4. Devlin et al. (2019), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

来源：`reference/BERT_overview.pdf`，16 页，arXiv:1810.04805v2，含附录全文已读。

- **预训练目标（pp.1–5）**：MLM + NSP；MLM 选 15% token，其中 80% mask、10% random、10% unchanged；NSP 一半真正下一句、一半随机。BooksCorpus 800M words，Wikipedia 2,500M。通用语言学习没有自带 monetary-policy polarity。
- **结构（pp.3–4）**：Base 12 层、hidden 768、12 heads、110M 参数；Large 24/1,024/16/340M。WordPiece，token+segment+position embeddings，`[CLS]`/`[SEP]`。
- **直接相关的警示（p.4 footnote 6）**：作者明确说，未微调的 `[CLS]` 向量不是有意义的句子表示，因为它按 NSP 目标训练。即使已做情绪微调，也不能无验证地假定余弦距离就是政策立场距离。
- **任务适配（pp.5–9）**：下游专门输出层与标签，全部参数微调；GLUE、QA、SWAG、NER 的成功不等于鹰鸽标签验证。NER 的拼接最后四层特征实验不是通用句相似度背书。
- **长度（p.13）**：最长 512 tokens，预训练 90% 步骤长度 128、最后 10% 长度 512；长文件应分句/分段并保留顺序、审计切分。
- **小样本（p.14）**：小数据集对初始化和超参数更敏感，作者建议多次运行及 validation 选择；不能用测试期收益反复选模型。

## 对本作业的共同结论（研究判断）

1. 明确区分金融情绪、经济展望、政策鹰鸽、相对前次变化、相对市场预期的意外；这五者不可共用一个名称。
2. 优先检验已有央行鹰鸽标签微调的模型是否可复用；FinBERT 原生 sentiment 适合作为独立维度/基准，而不是直接改标签。只有找到可复核的政策微调数据与评估，才考虑把 FinBERT 改用于鹰鸽分类。
3. 若采用语义相似度，需核实 sentence embedding 的训练方式和 pooling，使用有文献依据的完整鹰/鸽对照，并验证否定、条件句、数字、双重信号。两句临时写的 anchor 不足以证明量表有效。
4. 对 Chair Q&A 只统计 Chair 回答，单独保留问题作为语境；minutes 用章节定位而不是全文混入行政内容。输出句级概率与文档聚合便于追溯。
5. 人工语义验证与资产反应检验分开；模型与聚合规则在看测试期市场结果前确定。
