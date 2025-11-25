""" 
Google Earth Engineを使用して、
ハノイ周辺のLandsat 8データからLST（地表面温度）を取得し
、GeoTIFF形式でエクスポートするスクリプト 




対象領域一覧（仮）
- ハノイ
ROI = [105.27, 20.55, 106.03, 21.40] 
- ホーチミン
ROI = [106.60, 10.75, 106.85, 10.95]  
- ダナン
ROI = [108.20, 16.00, 108.40, 16.20]  
- ハイフォン
ROI = [106.70, 20.80, 106.90, 21.00]  
- カントー
ROI = [105.80, 10.00, 106.00, 10.20]  

"""
from datetime import datetime
import ee
import geemap

GGE_PROJECT = 'master-research-465403'  # Google Earth EngineプロジェクトID
# Google Earth Engine APIの初期化
try:
       ee.Initialize(project=GGE_PROJECT)
except Exception as e:
       ee.Authenticate()
       ee.Initialize(project=GGE_PROJECT)


### 
##  Parameters
###
# 取得するデータの期間
START_DATE = '2025-01-01'  # 開始日 例: '2025-01-01'
END_DATE = '2025-01-31'  # 終了日 例: '2025-01-31'


# 雲量の閾値（%） これ以下の画像を対象とする
CLOUD_COVER = 20  

# GEEの行政境界データセット（GAUL）
admin = ee.FeatureCollection("FAO/GAUL/2015/level2")
# ハノイの行政区画だけを抽出
ROI = admin.filter(ee.Filter.eq('ADM2_NAME', 'Ha Noi'))

#rect = ee.Geometry.Rectangle(ROI)

FILE_NAME_PREFIX = 'hanoi_lst8_202501'  # 出力ファイル名の接頭辞


FOLDER_NAME = 'EarthEngine'  # Google Drive上の保存先フォルダ
SCALE = 30  # 出力画像の解像度（メートル単位）
CRS = 'EPSG:4326'  # 出力画像の座標参照系（CRS） 基本的にはWGS84を使用
MAX_PIXELS = 1e9  # エクスポート可能な最大ピクセル数
###
##
###


# Landsat 8データセット
dataset = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .filterDate(START_DATE, END_DATE) \
    .filterBounds(ROI) \
    .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER))  # 雲量の閾値でフィルタリング

# LSTバンドを計算する関数
# 参考： https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products
def compute_lst_and_reflectance(img):
    thermal_bands  = img.select(['ST_B10']).multiply(0.00341802).add(149.0)
    optical_bands = img.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).multiply(0.0000275).add(-0.2)
    return thermal_bands.addBands(optical_bands).set('system:time_start', img.get('system:time_start'))

# 画像のマスク処理（雲、影、雪を除去）
# 参考：https://www.usgs.gov/media/files/landsat-8-9-collection-2-level-2-science-product-guide 
def mask_clouds(img):
    qa = img.select('QA_PIXEL')
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    shadow = qa.bitwiseAnd(1 << 4).eq(0)
    snow = qa.bitwiseAnd(1 << 5).eq(0)
    mask = cloud.And(shadow).And(snow)
    return img.updateMask(mask)

# LSTと反射率を計算し、マスク処理を適用
processed_dataset = dataset.map(compute_lst_and_reflectance).map(mask_clouds)







