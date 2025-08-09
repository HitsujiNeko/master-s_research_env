""" カテゴリ変数のエンコーディングを行う関数をまとめたモジュール

# 1 . Hold-out ターゲットエンコーディング
# 2 . ベイズターゲットエンコーディング


注意点
<#1. #2. >
以下の記事を参考に実装
https://note.com/crn_datascience/n/n9bc781d3ff4f

・各引数に渡す値
- df：学習データとテストデータを縦にくっつけたデータフレーム
- column_name：エンコーディングしたいカラム名
- target_name：目的変数のカラム名
- n_splits、random_state：モデル学習時のクロスバリデーションと同じ値
- prior_weight：ベイズターゲットエンコーディングで中央に寄せる度合い（100が基準、ベイズターゲットエンコーディングのみ）

・学習データとテストデータを縦にくっつけたデータフレームを渡す想定（第一引数）
・学習データとテストデータの見分け方は、「train」カラムが'True'か'False'で判別する想定です。
・リークの観点からモデル学習のクロスバリデーションと分割方法を合わせる必要があります。
なのでn_splits、random_stateを学習時の分割と合わせてください。

・関数はStratifiedKFold想定で作成しています。
各自のクロスバリデーションの分割手法に合わせて変更してください。

・エンコーディング前のカラムはそのまま残して、エンコーディング後のカラムは「元のカラム名+_hte（or bte）」という新しいカラムをデータフレームに追加します。
※hte：hold-out target encoding
 bte：bayesian target encoding

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

#1. Hold-outターゲットエンコーディング
def holdout_target_encoding(df, column_name, target_name, n_splits=10, random_state=123):
    """
    Hold-outターゲットエンコーディングを実施する関数。
    クロスバリデーションを考慮して、trainデータとtestデータを分けてエンコーディングを実施。
    :param df: 縦に繋げたDataFrame (trainとtestが一緒に含まれている)
    :param column_name: ターゲットエンコーディングを適用する列の名前
    :param target_name: ターゲット変数の列の名前
    :param n_splits: クロスバリデーションの分割数
    :param random_state: クロスバリデーションのランダムシード
    :return: ターゲットエンコーディングが適用されたDataFrame
    """
    # trainデータとtestデータを分割
    train_df = df[df['train'] == True].copy()
    test_df = df[df['train'] == False].copy()

    # クロスバリデーションのセットアップ
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_df[f'{column_name}_hte'] = np.nan

    # 各foldごとにターゲットエンコーディングを実施
    for train_idx, val_idx in skf.split(train_df, train_df[target_name]):
        X_train_fold, X_val_fold = train_df.iloc[train_idx], train_df.iloc[val_idx]

        # 訓練fold内の平均ターゲット値を計算
        mean_target = X_train_fold.groupby(column_name)[target_name].mean()

        # 検証foldにエンコーディングを適用
        train_df.loc[train_df.index[val_idx], f'{column_name}_hte'] = X_val_fold[column_name].map(mean_target)

    # testデータにはtrainデータ全体の平均値を適用
    global_mean_target = train_df.groupby(column_name)[target_name].mean()
    test_df[f'{column_name}_hte'] = test_df[column_name].map(global_mean_target)

    # trainとtestのエンコード結果を統合
    df = pd.concat([train_df, test_df], axis=0)

    return df



#2. ベイズターゲットエンコーディング
def bayesian_target_encoding(df, column_name, target_name, prior_weight=100, n_splits=10, random_state=300):
    """
    ベイズターゲットエンコーディングを実施する関数。
    クロスバリデーションを考慮して、trainデータとtestデータを分けてエンコーディングを実施。
    :param df: 縦に繋げたDataFrame (trainとtestが一緒に含まれている)
    :param column_name: ターゲットエンコーディングを適用する列の名前
    :param target_name: ターゲット変数の列の名前
    :param prior_weight: ベイズターゲットエンコーディングの際の全体平均に対する重み
    :param n_splits: クロスバリデーションの分割数
    :param random_state: クロスバリデーションのランダムシード
    :return: ターゲットエンコーディングが適用されたDataFrame
    """
    # trainデータとtestデータを分割
    train_df = df[df['train'] == True].copy()
    test_df = df[df['train'] == False].copy()

    # クロスバリデーションのセットアップ
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_df[f'{column_name}_bte'] = np.nan

    # 各foldごとにターゲットエンコーディングを実施
    for train_idx, val_idx in skf.split(train_df, train_df[target_name]):
        X_train_fold, X_val_fold = train_df.iloc[train_idx], train_df.iloc[val_idx]

        # 全体のターゲット平均を計算
        global_mean = X_train_fold[target_name].mean()

        # 訓練fold内のカテゴリごとのターゲット平均とサンプル数を計算
        category_stats = X_train_fold.groupby(column_name)[target_name].agg(['mean', 'count'])

        # ベイズターゲットエンコーディングの計算
        smooth = (category_stats['mean'] * category_stats['count'] + global_mean * prior_weight) / (category_stats['count'] + prior_weight)

        # 検証foldにエンコーディングを適用
        train_df.loc[train_df.index[val_idx], f'{column_name}_bte'] = X_val_fold[column_name].map(smooth)

    # testデータにはtrainデータ全体のベイズターゲットエンコーディングを適用
    global_smooth = (category_stats['mean'] * category_stats['count'] + global_mean * prior_weight) / (category_stats['count'] + prior_weight)
    test_df[f'{column_name}_bte'] = test_df[column_name].map(global_smooth)

    # trainとtestのエンコード結果を統合
    df = pd.concat([train_df, test_df], axis=0)

    return df