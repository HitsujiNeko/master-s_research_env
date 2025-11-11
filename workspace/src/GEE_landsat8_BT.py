# -*- coding: utf-8 -*-
"""
GEE を用いて Landsat 8 C2 Level-1 (Tier 1) から
Band 10 の輝度温度（BT, °C：大気補正・放射率補正なし）を算出し、
ROI 内の統計を出力、必要に応じて GeoTIFF を Google Drive にエクスポートする。

ベース：ユーザー提供の LST スクリプトと同じ構成・パラメータ設計
  - 期間/ROI/CLOUD_COVER/出力設定 などを踏襲
  - ただしデータセットは Level-1（LANDSAT/LC08/C02/T1）を使用
  - ST ではなく B10 から Radiance→BT を算出（補正なし）
"""

from datetime import datetime
import ee
import geemap

# ==== プロジェクト/初期化 ====
GEE_PROJECT = 'master-research-465403'   # 必要に応じて変更
try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)

# ==== Parameters ====
START_DATE = '2023-07-05'   # 例：検証用に 2023-07-07 付近
END_DATE   = '2023-07-09'
CLOUD_COVER = 80            # Level-1 のBTは雲影も含めた分布把握が目的なので緩めに

# 行政境界（GAUL）からハノイを取得（ユーザースクリプトと同様の流儀）
admin = ee.FeatureCollection("FAO/GAUL/2015/level2")
ROI = admin.filter(ee.Filter.eq('ADM2_NAME', 'Ha Noi')).geometry()

FILE_NAME_PREFIX = 'hanoi_bt_l8l1_20230707'
FOLDER_NAME = 'EarthEngine'   # Google Drive フォルダ
SCALE = 30
CRS = 'EPSG:4326'
MAX_PIXELS = 1e9

# 出力を有効化（True: エクスポート実行, False: 統計のみ）
ENABLE_EXPORT = True

# QA マスクを適用するか（False: “生のBT”に忠実 / True: 雲・影・巻雲を除去して比較用）
APPLY_QA_MASK = False


# ==== 関数群 ====
def mask_clouds(img: ee.Image) -> ee.Image:
    """
    QA_PIXEL ビットで雲/雲影/雪/巻雲を除去（0=clear）
    Level-1 にも QA_PIXEL は付与されているため利用可能。
    """
    qa = img.select('QA_PIXEL')
    cirrus  = qa.bitwiseAnd(1 << 2).eq(0)
    cloud   = qa.bitwiseAnd(1 << 3).eq(0)
    shadow  = qa.bitwiseAnd(1 << 4).eq(0)
    snow    = qa.bitwiseAnd(1 << 5).eq(0)
    mask = cirrus.And(cloud).And(shadow).And(snow)
    return img.updateMask(mask)


def add_bt_band(img: ee.Image) -> ee.Image:
    """
    Level-1 の B10 から Radiance → Brightness Temperature (BT, °C) を計算して
    'BT_C' バンドとして追加（補正なし）。
      Lλ = ML * DN + AL
      BT(K) = K2 / ln(K1 / Lλ + 1)
      BT(°C) = BT(K) - 273.15
    係数は画像プロパティから取得。
    """
    ml = ee.Number(img.get('RADIANCE_MULT_BAND_10'))
    al = ee.Number(img.get('RADIANCE_ADD_BAND_10'))
    k1 = ee.Number(img.get('K1_CONSTANT_BAND_10'))
    k2 = ee.Number(img.get('K2_CONSTANT_BAND_10'))

    rad = img.select('B10').multiply(ml).add(al)                 # Radiance
    bt_k = k2.divide((k1.divide(rad)).add(1).log())              # Kelvin
    bt_c = bt_k.subtract(273.15).rename('BT_C')                  # Celsius
    return img.addBands(bt_c)


def image_info_summary(im: ee.Image) -> dict:
    keys = ['LANDSAT_SCENE_ID', 'WRS_PATH', 'WRS_ROW',
            'CLOUD_COVER', 'DATE_ACQUIRED', 'SCENE_CENTER_TIME']
    return im.toDictionary(keys).getInfo()


def reduce_stats(im: ee.Image, geom: ee.Geometry, scale: int = 30) -> dict:
    reducers = ee.Reducer.percentile([2,5,25,50,75,95,98]).combine(
        reducer2=ee.Reducer.minMax(), sharedInputs=True
    )
    res = im.select('BT_C').reduceRegion(
        reducer=reducers,
        geometry=geom,
        scale=scale,
        maxPixels=MAX_PIXELS,
        bestEffort=True
    )
    return res.getInfo()


# ==== データセット（Level-1, Tier 1） ====
col = (ee.ImageCollection('LANDSAT/LC08/C02/T1')
       .filterDate(START_DATE, END_DATE)
       .filterBounds(ROI)
       .filter(ee.Filter.lte('CLOUD_COVER', CLOUD_COVER)))

# 画像ごとに BT を追加、必要なら QA マスク
def prep(im):
    if APPLY_QA_MASK:
        im = mask_clouds(im)
    return add_bt_band(im)

col_bt = col.map(prep)

# 撮影時刻に最も近い 1 シーン（中央日付に近い画像）を選択して詳細確認
center = ee.Date(datetime.strptime('2023-07-07', '%Y-%m-%d'))
col_near = col_bt.map(lambda im: im.set('timeDiff',
                                        ee.Number(im.date().difference(center, 'second')).abs()))
img_bt = ee.Image(col_near.sort('timeDiff').first())

# === 統計の表示 ===
print('=== Selected Scene ===')
print(image_info_summary(img_bt))
print('=== BT (°C) stats over ROI ===')
stats = reduce_stats(img_bt, ROI, SCALE)
for k in sorted(stats.keys()):
    print(f'{k}: {stats[k]:.3f}')

# === エクスポート（任意） ===
if ENABLE_EXPORT:
    task = ee.batch.Export.image.toDrive(
        image=img_bt.select('BT_C').reproject(crs=CRS, scale=SCALE),
        description=f'{FILE_NAME_PREFIX}_BT_C',
        folder=FOLDER_NAME,
        fileNamePrefix=f'{FILE_NAME_PREFIX}_BT_C',
        region=ROI,
        scale=SCALE,
        crs=CRS,
        maxPixels=MAX_PIXELS
    )
    task.start()
    print('[Export] Started to Google Drive:', f'{FILE_NAME_PREFIX}_BT_C')
