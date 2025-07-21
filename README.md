
# Anaconda データ分析開発環境セットアップ手順

このリポジトリは、Anaconda（conda）ベースのデータサイエンス・分析のための開発環境（Dev Container + conda）を簡単に構築できるテンプレートです。

## 前提条件

- Docker Desktop（Linux, Windows, Mac いずれも可）
- VS Code（Dev Containers拡張をインストール）

---

## 構成
- Dev Container 対応
- conda環境（Python, numpy, pandas, matplotlib, jupyterlab, geopandas, gdal, scikit-learn, seaborn, japanize-matplotlib など）

---

## セットアップ手順
1. このリポジトリを任意の場所に clone してください。
2. VS Code でリポジトリを開きます。
3. コマンドパレット（F1）から「Dev Containers: Reopen in Container」を選択します。
4. 初回起動時にconda環境が自動構築されます。

---

## データの利用例
コンテナ内で `/workspaces/ResearchEnv/workspace/data/` ディレクトリを自由に利用できます。

例: Jupyter Notebook
```python
import pandas as pd
df = pd.read_csv('/workspaces/ResearchEnv/workspace/data/sample.csv')
```

---

## 注意事項
- `.env` ファイルなど個人設定ファイルは **公開リポジトリには含めないでください**（`.gitignore`推奨）
- Windows/WSL, Mac, Linux いずれも利用可能です

---

## 推奨環境
- VS Code + Dev Containers拡張
- Docker Desktop
- Anaconda/Miniconda（コンテナ内で自動セットアップされます）

---
