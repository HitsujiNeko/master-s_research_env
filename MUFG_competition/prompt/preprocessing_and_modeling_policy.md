# MUFG Competition: 前処理・モデル選定方針まとめ

## 1. データの前処理方針

- 欠損値は存在しないため、欠損値補完は不要。
- 数値特徴量（GrossApproval, SBAGuaranteedApproval, InitialInterestRate, TermInMonths, CongressionalDistrict, JobsSupported）は分布の幅が広く、外れ値が多い。
  - GrossApproval, SBAGuaranteedApproval, TermInMonths, JobsSupported で外れ値が多い。必要に応じて外れ値除去・変換やロバストな手法を検討。
  - InitialInterestRate, CongressionalDistrict は外れ値が少ない。
- カテゴリ変数（Subprogram, FixedOrVariableInterestInd, NaicsSector, BusinessType, BusinessAge, CollateralInd, RevolverStatus, ApprovalFiscalYear）は一部の値に偏りがある。
  - 各カテゴリ変数の分布・解釈結果に基づき、最適なエンコーディング手法を選択する：
    - Subprogram（ローン種別）：Count EncodingまたはTarget Encoding（デフォルト率エンコーディング）
    - FixedOrVariableInterestInd（固定/変動金利）：Label Encoding（0/1変換）
    - NaicsSector（業種大分類）：Count EncodingまたはTarget Encoding
    - BusinessType（事業形態）：Count EncodingまたはTarget Encoding
    - BusinessAge（事業年数区分）：Label EncodingまたはCount Encoding
    - CollateralInd（担保有無）：Label Encoding（0/1変換）
    - RevolverStatus（リボルビング/定期貸付）：0/1の数値型としてそのまま利用
    - ApprovalFiscalYear（承認年度）：Label EncodingまたはCount Encoding（数値型としても利用可）
    ※多値カテゴリはOne-Hot Encodingは次元が増えすぎるため非推奨。LightGBMではCount Encodingが高速・安定。
  
- 連続値は標準化（StandardScaler）を基本とし、必要に応じて正規化も検討。
- 多重共線性（GrossApprovalとSBAGuaranteedApprovalの相関が高い）に注意し、特徴量選択やモデル解釈時に考慮。
- データの分布はtrain/test間で大きな差はないが、NaicsSectorのPublic administrationはtestのみ。

### 補足：
### カテゴリ変数ごとのエンコーディング方針
1. **Subprogram（ローン種別）**
   - 少数派カテゴリも含めて分布が偏っているため、モデルの解釈性・汎化性を重視し「Count Encoding（頻度エンコーディング）」または「Target Encoding（デフォルト率エンコーディング）」が有効。
   - LightGBMではCount Encodingが高速・安定。解釈性重視ならTarget Encodingも検討。

2. **FixedOrVariableInterestInd（固定/変動金利）**
   - 2値（F/V）のみなので「Label Encoding（0/1変換）」または「One-Hot Encoding」どちらでも可。シンプルなLabel Encoding推奨。

3. **NaicsSector（業種大分類）**
   - 多数のカテゴリがあり、分布に偏りがある。少数派も重要な説明変数となるため「Count Encoding」または「Target Encoding」が有効。
   - One-Hotは次元が増えすぎるため非推奨。

4. **BusinessType（事業形態）**
   - 圧倒的多数（CORPORATION）と少数派が混在。Count EncodingまたはTarget Encodingが有効。

5. **BusinessAge（事業年数区分）**
   - 5カテゴリ程度で分布に偏りあり。Label EncodingまたはCount Encodingが有効。

6. **CollateralInd（担保有無）**
   - 2値（Y/N）のみなのでLabel Encoding（0/1変換）推奨。

7. **RevolverStatus（リボルビング/定期貸付）**
   - 0/1の数値型なのでそのまま利用可能。追加エンコーディング不要。

8. **ApprovalFiscalYear（承認年度）**
   - 年度（2020～2024）で数値型として扱えるが、カテゴリとして扱う場合はLabel EncodingまたはCount Encodingが有効。


---


## 2. モデル選定方針

- 目的変数LoanStatusは2値分類（1=デフォルト, 0=非デフォルト）でクラス不均衡（1が約12%）。
- 評価指標はF1スコア。
- ロジスティック回帰、決定木系（RandomForest, LightGBM）を候補。
  - 外れ値や分布の偏りに強い決定木系モデルが有力。
  - 複数モデルで比較し、F1スコアが高いものを選定。
- 重要な特徴量（融資額、金利、期間、業種、事業形態、担保有無、年度など）はモデル解釈・改善に活用。

## 3. 実装・再現性

- Python3, pandas, numpy, scikit-learn, lightgbm等の主要ライブラリを使用。
- シード値固定で再現性を担保。
- データ分析はJupyter Notebook、モデル実装はPythonファイルで管理。
- 主要な分析・実装の進捗はGitで管理。

## 4. 前処理・モデル構築の流れ

1. データ読み込み
2. 外れ値処理（必要に応じて）
3. 特徴量エンジニアリング（カテゴリ変数のOne-Hot化、数値特徴量の標準化）
4. 学習データ・テストデータの分割
5. モデル選定・学習（ロジスティック回帰、RandomForest、LightGBM等）
6. 評価（F1スコア）
7. 重要特徴量の可視化・解釈
8. 予測結果の提出ファイル作成

---

この方針に基づき、データの前処理からモデル構築までを行うプログラムを作成します。
