"""
ベトナムハノイのLST（地表面温度）の時系列分析を行うプログラム

グラフとしてLSTの1年間の変化を表示し、昼間と夜間のLSTを比較する。
LSTは範囲内の画像の平均を計算する

使用データ：

MODIS MOD11A2（LST Day 1km）

https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A2

- 解像度：1km
- 観測周期：8日合成
データセットの利用可能時期
2000-02-18 T00:00:00 ~ 

"""

import ee
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from datetime import datetime


# 認証・初期化
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

# ROI（関心領域）を定義
roi = ee.Geometry.Rectangle([105.27, 20.55, 106.03, 21.40])

# MODIS LSTコレクション（8日合成）
collection = ee.ImageCollection("MODIS/061/MOD11A2") \
    .select("LST_Day_1km") \
    .filterDate("2021-01-01", "2023-08-31") \
    .filterBounds(roi)

# LST値を摂氏に変換
def convert_lst(img):
    lst = img.multiply(0.02).subtract(273.15)
    lst = lst.rename('LST_C')
    return lst.copyProperties(img, ["system:time_start", "system:time_end"])

LSTDay = collection.map(convert_lst)

# 領域平均LSTを画像プロパティとして追加
def add_mean_property(img):
    stats = img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=1000,
        maxPixels=int(1e8)
    )
    mean = stats.get('LST_C')
    return img.set({'mean_LST_C': mean})

LSTDay_mean = LSTDay.map(add_mean_property)

# DOY（年内通日）と年ごとのLSTを抽出
info = LSTDay_mean.toList(LSTDay_mean.size())
dates = []
years = []
doys = []
values = []

for i in range(info.size().getInfo()):
    img = ee.Image(info.get(i))
    timestamp = img.get('system:time_start').getInfo()
    if timestamp is not None:
        dt = datetime.utcfromtimestamp(timestamp / 1000)
        doy = dt.timetuple().tm_yday
        year = dt.year
        val = img.get('mean_LST_C').getInfo()
        if val is not None:
            dates.append(dt)
            years.append(year)
            doys.append(doy)
            values.append(val)

df = pd.DataFrame({'date': dates, 'year': years, 'doy': doys, 'LST_C': values})

# 年ごとにDOYでプロット
plt.figure(figsize=(12,6))
for y in sorted(df['year'].unique()):
    sub = df[df['year'] == y]
    plt.plot(sub['doy'], sub['LST_C'], label=str(y))
plt.xlabel('Day of Year (DOY)')
plt.ylabel('Land Surface Temperature (°C)')
plt.title('Seasonal Variation of Land Surface Temperature (2019-2023)')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('hanoi_lst_time_series.png')