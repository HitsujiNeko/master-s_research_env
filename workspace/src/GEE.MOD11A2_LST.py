"""
Google Earth Engineを使用して、
MODIS MOD11A2（8日合成LST） 昼間  データから地表面温度（LST）を取得し
GeoTIFF形式でエクスポートするスクリプト

対象領域一覧（仮）
- ハノイ
REGION = [105.27, 20.55, 106.03, 21.40]
- ホーチミン
REGION = [106.60, 10.75, 106.85, 10.95]
- ダナン
REGION = [108.20, 16.00, 108.40, 16.20]
- ハイフォン
REGION = [106.70, 20.80, 106.90, 21.00]
- カントー
REGION = [105.80, 10.00, 106.00, 10.20]
"""

from datetime import datetime
import ee

GGE_PROJECT = 'master-research-465403'  # Google Earth EngineプロジェクトID

# 取得するデータの期間
START_DATE = '2025-01-01'
END_DATE = '2025-01-31'

# 対象領域（例：ハノイ）
REGION = [105.27, 20.55, 106.03, 21.40]

FILE_NAME_PREFIX = 'hanoi_modis_lst_202501'
FOLDER_NAME = 'EarthEngine'
SCALE = 1000  # MODISの空間解像度（m）
CRS = 'EPSG:4326'
MAX_PIXELS = 1e9



# Earth Engine API初期化
try:
    ee.Initialize(project=GGE_PROJECT)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=GGE_PROJECT)

rect = ee.Geometry.Rectangle(REGION)

# MODIS MOD11A2 LSTデータセット
dataset = ee.ImageCollection('MODIS/061/MOD11A2') \
    .filterDate(START_DATE, END_DATE) \
    .filterBounds(rect)



# LSTバンド（LST_Day_1km）を摂氏に変換
# スケール: 0.02, Kelvin → Celsius
def calc_modis_lst(img):
    lst = img.select('LST_Day_1km').multiply(0.02).subtract(273.15)
    lst = lst.rename('LST_C')
    return lst.copyProperties(img, ['system:time_start', 'system:time_end'])


lst_images = dataset.map(calc_modis_lst)
mean_lst = lst_images.mean()


# 期間内の画像の日付一覧を表示
dates = dataset.aggregate_array('system:time_start').getInfo()
for t in dates:
    dt = datetime.utcfromtimestamp(t / 1000)
    print(dt.strftime('%Y-%m-%d %H:%M:%S'))


# GeoTIFFでエクスポート
task = ee.batch.Export.image.toDrive(
    image=mean_lst,
    description='Hanoi_MODIS_LST_Mesh',
    folder=FOLDER_NAME,
    fileNamePrefix=FILE_NAME_PREFIX,
    region=rect,
    scale=SCALE,
    crs=CRS,
    maxPixels=MAX_PIXELS
)

task.start()
print("Export task started. Check your Google Drive's 'EarthEngine'フォルダ.")

