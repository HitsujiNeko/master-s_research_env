---
mode: agent
---

タスクには、Github Copilot が行うタスクを記述する。
このファイルは、Github Copilot がタスクを理解し、適切なコードを生成するための指示を含んだ
ドキュメントです。

# 進捗状況
- [〇] データの理解と初期分析
- [ ] データの前処理
- [ ] モデルの選定と実装
- [ ] モデルの学習と評価
- [ ] 提出ファイルの作成
- [ ] AIによる解釈用ファイルの作成

# タスク
MUFG_competition/prompt/interpretation.txt
を読み込み、その中に記載されているデータをすべて理解し、以下のタスクを実行する。  

1. interpretation.txtの <AIが解釈結果を記述> の部分に、AIが解釈した内容を記述する。
2. interpretation.txtに基づきデータの前処理・モデルの選定・実装に関する方針をまとめたMUFG_competition/prompt/preprocessing_and_modeling_policy.md を作成する。



# コンペティションの目的
中小企業向け融資データをもとに、各企業が債務を返済できるかどうか（デフォルトするかどうか）を予測するモデルを構築し、提出フォーマット（sample_submit.csv）に従って予測結果を出力する。

# 使用データセット
- train.csv: 学習用データ
- test.csv: 予測用データ
- sample_submit.csv: 提出フォーマット

データの詳細は description.txt を参照。

データのパス:
- MUFG_competition/data/raw/train.csv
- MUFG_competition/data/raw/test.csv
- MUFG_competition/data/raw/sample_submit.csv

# データ出力場所
- 前処理済みデータ: MUFG_competition/data/processed/
- 予測結果・提出ファイル: MUFG_competition/data/output/

# 評価方法
- F1スコアを使用
- 暫定評価: 評価用データセットの50%で評価
- 最終評価: 残り50%で評価
- リーダーボードは最終評価で順位決定

# 分析・実装方法
- データ分析はJupyter Notebookで行う
  - Notebookの場所: MUFG_competition/notebooks/
  - 各セルには、何を行うかコメントで記載する
- モデル実装はPythonファイルで行う
  - Pythonファイルの場所: MUFG_competition/src
  - 各関数には、役割や処理内容をコメントで記載する

# データ理解・初期分析
- 分析結果は以下の場所に保存している。適宜参照すること。
  - MUFG_competition/prompt/interpretation.txt

# AIによる解釈用ファイル
- 分析結果をAIが解釈できるよう、interpretation.txtを作成する
- 含める内容:
  - データの概要
  - データの前処理方法
  - モデルの選定理由
  - モデルの評価方法
  - 結果の解釈

# 追加指示（現状データ理解前の段階）
- 仮説や着眼点（どの特徴量が重要そうか等）をNotebookに記載する
- データの可視化・分布確認を優先する
- データ辞書の作成を推奨

# 実行環境・再現性
- Python3, pandas, numpy, scikit-learn等の主要ライブラリを使用
- シード値固定で再現性を担保

# 提出ファイルフォーマット
- sample_submit.csvの形式に従う（id, 予測値[0/1]）

# チーム開発・バージョン管理
- Gitによるバージョン管理を推奨
- 主要な分析・実装の進捗はコミットメッセージで記録
