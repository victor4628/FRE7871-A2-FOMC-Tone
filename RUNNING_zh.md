# Assignment 2 复现说明

在 `assignment2` 目录创建 Python 3.11/3.12 环境，然后运行：

```powershell
python -m pip install -r requirements.txt
python scripts/run_all.py --refresh --rescore
python scripts/build_notebook.py
python scripts/build_report.py
```

首次运行会从 Federal Reserve、Federal Reserve H.15、Yahoo Finance 和
Hugging Face 下载公开数据与 `ProsusAI/finbert`。这些文件保存在被
`.gitignore` 排除的 `data/` 目录，不会提交到仓库。Notebook 和最终 PDF
包含已保存的结果。

