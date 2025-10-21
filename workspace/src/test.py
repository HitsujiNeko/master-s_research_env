"""
geotiff 形式のファイルを読み込み、処理を学ぶためのサンプルコード



"""


import os
import rasterio
import numpy as np
import pandas as pd
from glob import glob

# -------------------------------
# ファイルの読み込み
# -------------------------------
FILE_PATH = 'workspace/data/geotiff/Landsat8/2023/L8_20230808_032316_Hanoi_Reflectance.tif'

with rasterio.open(FILE_PATH) as src:
    data = src.read()
    print("データの形状:", data.shape)
    print("メタデータ:", src.meta)
    print("バンド情報:", src.descriptions)
    print("座標参照系:", src.crs)
    print("変換情報:", src.transform)
    print("各バンドの統計情報:")
    for i in range(data.shape[0]):
        band = data[i]
        print(f" バンド {i+1}: min={np.min(band)}, max={np.max(band)}, mean={np.mean(band)}, std={np.std(band)}")   