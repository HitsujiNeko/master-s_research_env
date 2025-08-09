import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import optuna
from category_encoding import bayesian_target_encoding

"""
モデルはLightGBMを使用し、Optunaでハイパーパラメータチューニングを行う。

"""



# 1. データ読み込み
train = pd.read_csv("MUFG_competition/data/raw/train.csv")
test = pd.read_csv("MUFG_competition/data/raw/test.csv")

test_ID = test['id'] .copy() 
cat_features = [
    "Subprogram", 
    "FixedOrVariableInterestInd", 
    "NaicsSector", 
    "BusinessType", 
    "BusinessAge", 
    "CollateralInd", 
    "RevolverStatus", 
    "ApprovalFiscalYear"
]
num_features = [
    "GrossApproval", 
    "SBAGuaranteedApproval", 
    "InitialInterestRate", 
    "TermInMonths", 
    "CongressionalDistrict", 
    "JobsSupported"
]
target = "LoanStatus"

# 2. データの前処理

train['train'] = True
test['train'] = False
all_df = pd.concat([train, test], ignore_index=True)
print(all_df.head() )
print(all_df.info())

""" 数値特徴量の前処理方針
AIが提案した処理方針

### GrossApproval（融資承認額）
**前処理方針:**  
- 対数変換（log1p）+ 標準化  
**理由:**  
- 右裾が長い分布で外れ値が多い。  
- 対数変換で分布の歪み・外れ値の影響を緩和し、標準化でスケールを揃える。
---
### SBAGuaranteedApproval（補償金額）
**前処理方針:**  
- 対数変換（log1p）+ 標準化  
**理由:**  
- GrossApprovalと同様に右裾が長く外れ値が多い。  
- 高い相関があるため、同じ処理で多重共線性対策にもなる。
---
### InitialInterestRate（初期金利）
**前処理方針:**  
- 標準化のみ  
**理由:**  
- 分布は比較的まとまっており、外れ値がほぼ存在しない。  
- 標準化で十分。
---

### TermInMonths（融資期間）
**前処理方針:**  
- 標準化+（必要に応じて）カテゴリ分割（例：5年/10年/25年などの区分）  
**理由:**  
- 2ピーク型分布で外れ値が多い。  
- 標準化でスケール調整し、区分化でモデルの解釈性向上も期待できる。
---

### JobsSupported（借入者事業者の雇用数（借入者の自己申告））
**前処理方針:**  
- 対数変換（log1p）+ 標準化  
**理由:**  
- 小規模事業者中心で右裾が長い分布、外れ値も多い。  
- 対数変換で分布を整え、標準化でスケール調整。
---

#### まとめ
- **右裾が長い分布・外れ値多い特徴量**：対数変換 + 標準化
- **分布がまとまっている特徴量**：標準化のみ
"""
# 数値特徴量の前処理
for col in num_features:
    if col in ['GrossApproval', 'SBAGuaranteedApproval', 'JobsSupported']:
        all_df[col] = np.log1p(all_df[col])
    all_df[col] = StandardScaler().fit_transform(all_df[[col]])


""" カテゴリ変数のエンコーディング
Subprogram（ローン種別）：ベイズ Target Encoding
FixedOrVariableInterestInd（固定/変動金利）：Label Encoding（0/1変換）
NaicsSector（業種大分類）：Count EncodingまたはTarget Encoding
BusinessType（事業形態）：Count EncodingまたはTarget Encoding
BusinessAge（事業年数区分）：Label EncodingまたはCount Encoding
CollateralInd（担保有無）：Label Encoding（0/1変換）
RevolverStatus（定期貸付（0）かリボルビング（1）か）：0/1の数値型としてそのまま利用
ApprovalFiscalYear（承認年度）：Label EncodingまたはCount Encoding（数値型としても利用可）
CongressionalDistrict（借入者の住所が属する選挙区）： 
"""

# Subprogramのベイズターゲットエンコーディング （エンコーディング後はSubprogram_bteに追加される。）
all_df = bayesian_target_encoding(all_df, 'Subprogram', target, prior_weight=100, n_splits=10, random_state=42)
all_df.drop(columns=['Subprogram'], inplace=True)
all_df.rename(columns={'Subprogram_bte': 'Subprogram'}, inplace=True)

# FixedOrVariableInterestIndのLabel Encoding
all_df['FixedOrVariableInterestInd'] = all_df['FixedOrVariableInterestInd'].map({'Fixed': 0, 'Variable': 1})

# NaicsSectorのCount Encoding
naics_count = all_df['NaicsSector'].value_counts()
all_df['NaicsSector'] = all_df['NaicsSector'].map(naics_count)

# BusinessTypeのCount Encoding
business_count = all_df['BusinessType'].value_counts()
all_df['BusinessType'] = all_df['BusinessType'].map(business_count)

# BusinessAgeのLabel Encoding
all_df['BusinessAge'] = all_df['BusinessAge'].astype('category').cat.codes

# CollateralIndのLabel Encoding
all_df['CollateralInd'] = all_df['CollateralInd'].map({'Yes': 1, 'No': 0})

# RevolverStatusのそのまま利用

# ApprovalFiscalYearのLabel Encoding
all_df['ApprovalFiscalYear'] = all_df['ApprovalFiscalYear'].astype('category').cat.codes

# CongressionalDistrictのベイズターゲットエンコーディング
all_df = bayesian_target_encoding(all_df, 'CongressionalDistrict', target, prior_weight=100, n_splits=10, random_state=42)
all_df.drop(columns=['CongressionalDistrict'], inplace=True)
all_df.rename(columns={'CongressionalDistrict_bte': 'CongressionalDistrict'}, inplace=True)

print('エンコーディング後のデータフレーム:')
print(all_df.info())


print('前処理後のデータフレーム:')
print(all_df.head(10))


# 3. データの分割
train_df = all_df[all_df['train']].drop(columns=['id', 'train', target])
test_df = all_df[~all_df['train']].drop(columns=['id', 'train', target])

y = all_df[all_df['train']][target]
X_train, X_val, y_train, y_val = train_test_split(
    train_df, y, test_size=0.2, random_state=42, stratify=y
)

# 4. モデルの定義とOptunaによるハイパーパラメータチューニング
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', -1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        'random_state': 42,
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(params, dtrain, valid_sets=[dval],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=10),
                        lgb.log_evaluation(period=100)
                    ])

    y_pred = model.predict(X_val)
    y_pred_binary = np.where(y_pred > 0.5, 1, 0)
    
    f1 = f1_score(y_val, y_pred_binary)
    return f1

def tune_hyperparameters():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    print('Best trial:', study.best_trial)
    return study.best_params

# ハイパーパラメータのチューニング
best_params = tune_hyperparameters()
# 5. 最終モデルの学習
final_model = lgb.LGBMClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)
# 6. テストデータの予測
test_predictions = final_model.predict(test_df)
# 7. 結果の保存
submission = pd.DataFrame({
    'Id': test_ID,
    'LoanStatus': test_predictions
})
submission.to_csv("MUFG_competition/data/processed/submission.csv", index=False, header=True)
print("Submission saved to MUFG_competition/data/processed/submission.csv")