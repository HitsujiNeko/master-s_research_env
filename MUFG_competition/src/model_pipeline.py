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


# 1. データ読み込み
train = pd.read_csv("MUFG_competition/data/raw/train.csv")
test = pd.read_csv("MUFG_competition/data/raw/test.csv")

cat_cols = [
    "Subprogram", 
    "FixedOrVariableInterestInd", 
    "NaicsSector", 
    "BusinessType", 
    "BusinessAge", 
    "CollateralInd", 
    "RevolverStatus", 
    "ApprovalFiscalYear"
]
num_cols = [
    "GrossApproval", 
    "SBAGuaranteedApproval", 
    "InitialInterestRate", 
    "TermInMonths", 
    "CongressionalDistrict", 
    "JobsSupported"
]

# 2. データの前処理

""" カテゴリ変数のエンコーディング
Subprogram（ローン種別）：Holdout Target Encoding
FixedOrVariableInterestInd（固定/変動金利）：Label Encoding（0/1変換）
NaicsSector（業種大分類）：Count EncodingまたはTarget Encoding
BusinessType（事業形態）：Count EncodingまたはTarget Encoding
BusinessAge（事業年数区分）：Label EncodingまたはCount Encoding
CollateralInd（担保有無）：Label Encoding（0/1変換）
RevolverStatus（リボルビング/定期貸付）：0/1の数値型としてそのまま利用
ApprovalFiscalYear（承認年度）：Label EncodingまたはCount Encoding（数値型としても利用可）
"""
# Subprogram の Holdout Target Encoding
def holdout_target_encoding(train, test, col, target, n_splits=5):
    train_encoded = pd.Series(index=train.index, dtype=float)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(train, train[target]):
        # 他分割で平均計算
        mean_map = train.iloc[train_idx].groupby(col)[target].mean()
        # 検証分割に割り当て
        train_encoded.iloc[val_idx] = train.iloc[val_idx][col].map(mean_map)
    # テストデータには全体平均を割り当て
    global_mean_map = train.groupby(col)[target].mean()
    test_encoded = test[col].map(global_mean_map)
    return train_encoded, test_encoded

# 例：Subprogramに適用
train_subprogram_enc, test_subprogram_enc = holdout_target_encoding(
    train, test, col="Subprogram", target="LoanStatus", n_splits=5
)


# # ワンホットエンコーディング
# train_cat = pd.get_dummies(train[cat_cols].astype(str), drop_first=True)
# test_cat = pd.get_dummies(test[cat_cols].astype(str), drop_first=True)
## カテゴリ変数の次元合わせ
#train_cat, test_cat = train_cat.align(test_cat, join='left', axis=1, fill_value=0)

# 数値特徴量の標準化
scaler = StandardScaler()
num_features = ["GrossApproval", "SBAGuaranteedApproval", "InitialInterestRate", "TermInMonths", "CongressionalDistrict", "JobsSupported"]
train_num = scaler.fit_transform(train[num_features])
test_num = scaler.transform(test[num_features])

# 特徴量結合
X_train = np.hstack([train_num, train_cat.values])
X_test = np.hstack([test_num, test_cat.values])
y_train = train["LoanStatus"].values

# 4. 学習データ分割
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)


# 5. OptunaによるLightGBMハイパーパラメータ最適化
def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
best_params = study.best_params
best_model = lgb.LGBMClassifier(**best_params)
best_model.fit(X_train, y_train)

# 6. テストデータ予測
pred_test = best_model.predict(X_test)

# 7. 提出ファイル作成
submit = pd.DataFrame({"id": test["id"], "LoanStatus": pred_test})
submit.to_csv("MUFG_competition/data/output/sample_submit.csv", index=False, header=True)

# 8. 重要特徴量の可視化
importances = best_model.feature_importances_
feature_names = num_features + list(train_cat.columns)
feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
feat_imp.sort_values("importance", ascending=False).to_csv("MUFG_competition/data/output/feature_importance.csv", index=False)

print(f"Optuna best params: {best_params}")
print("Prediction and feature importance files saved.")
