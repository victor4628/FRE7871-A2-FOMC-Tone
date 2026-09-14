# Assignment 2 复现说明

建议使用 Python 3.11 或 3.12。在仓库根目录运行：

```powershell
python -m venv .venv
& .venv\Scripts\python.exe -m pip install -r requirements.txt
& .venv\Scripts\hf.exe auth login
& .venv\Scripts\python.exe -m src.train_temporal_roberta
$env:FOMC_MODEL_PATH = "data/models/fomc-roberta-temporal"
& .venv\Scripts\python.exe scripts\run_v2.py --refresh-documents --refresh-market --rescore
& .venv\Scripts\python.exe scripts\build_notebook_v2.py
& .venv\Scripts\python.exe scripts\build_report_v2.py
```

上述命令会用作者公开的 1996-2019 时间训练集复现当前保存的 RoBERTa
结果。训练文件由脚本从固定的作者仓库版本下载，模型保存在 Git 忽略目录。

官方 FOMC-RoBERTa 需要先在模型页面申请访问：
<https://huggingface.co/gtfintechlab/FOMC-RoBERTa>。令牌只需读取权限，不应
写入仓库或聊天。获批后删除 `FOMC_MODEL_PATH` 环境变量，评分脚本会直接
使用官方 checkpoint。

抓取的数据、模型缓存和所有派生 CSV 位于被 `.gitignore` 排除的 `data/`
和 `outputs/`。GitHub 仅保留代码、已执行 notebook、最终 PDF 和说明文件。
