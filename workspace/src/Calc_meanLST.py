"""
Geotiff形式のLSTデータを読み込み、平均値を計算するスクリプト


・入力: 各月のLSTデータ（GeoTIFF形式）
・出力: 各月の平均LST値を計算し、CSVファイルに保存

- 入力データの概要
・形式: GeoTIFF
・相対パス： workspace/data/geotiff/LST_{YEAR}/LST_{YEAR}_{MONTH}.tif
例: workspace/data/geotiff/LST_2023/LST_2023_01.tif
・ LST値が0のピクセルは無視して平均値を計算

- 出力データの概要
・形式: CSV
・相対パス： workspace/data/csv/LST_mean_{YEAR}.csv

"""

import os
import numpy as np
import rasterio
import pandas as pd

def calculate_mean_lst(year):
    """
    指定された年のLSTデータから平均値を計算し、CSVファイルに保存する関数
    :param year: 年（例: 2023）
    """
    monthly_means = []

    for month in range(1, 13):
        # ファイルパスの生成
        file_path = f"workspace/data/geotiff/LST_{year}/LST_{year}_{month:02d}.tif"
        
        if not os.path.exists(file_path):
            print(f"ファイルが存在しません: {file_path}")
            monthly_means.append(np.nan)
            continue
        
        # GeoTIFFファイルの読み込み
        with rasterio.open(file_path) as src:
            lst_data = src.read(1)  # 1バンド目を読み込み
            
            # LST値が0のピクセルを無視して平均値を計算
            valid_lst_data = lst_data[lst_data > 0]
            if valid_lst_data.size > 0:
                mean_value = np.mean(valid_lst_data)
            else:
                mean_value = np.nan  # 有効なデータがない場合はNaN
            
            monthly_means.append(mean_value)

    # 結果をDataFrameに変換
    df = pd.DataFrame({
        'Month': [f"{month:02d}" for month in range(1, 13)],
        'Mean_LST': monthly_means
    })

    # CSVファイルに保存
    output_path = f"workspace/data/csv/LST_mean_{year}.csv"
    df.to_csv(output_path, index=False)
    print(f"平均LST値を保存しました: {output_path}")

if __name__ == "__main__":
    # 年を指定して関数を実行
    year = 2023  # 例: 2023年のデータを処理
    calculate_mean_lst(year)