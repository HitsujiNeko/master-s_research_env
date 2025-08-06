""" 
Google Earth Engineを使用して、
ハノイ周辺のLandsat 8データからLST（地表面温度）を取得し
、GeoTIFF形式でエクスポートするスクリプト 

大気補正済みのLandsat 8 Collection 2 Level 2データを使用

注意点：放射率は考慮していない



現状のコードは期間内の平均LST画像を出力するコード



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

GGE_PROJECT = 'master-research-465403'  # Google Earth EngineプロジェクトID

### 
##  Parameters
###
# 取得するデータの期間
START_DATE = '2025-01-01'  # 開始日 例: '2025-01-01'
END_DATE = '2025-01-31'  # 終了日 例: '2025-01-31'


# 雲量の閾値（%） これ以下の画像を対象とする
CLOUD_COVER = 20  

# 範囲：　　緯度1, 経度1, 緯度2, 経度2の順で指定
ROI = [105.27, 20.55, 106.03, 21.40]  

FILE_NAME_PREFIX = 'hanoi_lst8_202501'  # 出力ファイル名の接頭辞

FOLDER_NAME = 'EarthEngine'  # Google Drive上の保存先フォルダ
SCALE = 30  # 出力画像の解像度（メートル単位）
CRS = 'EPSG:4326'  # 出力画像の座標参照系（CRS） 基本的にはWGS84を使用
MAX_PIXELS = 1e9  # エクスポート可能な最大ピクセル数
###
##
###


# Google Earth Engine APIの初期化
try:
       ee.Initialize(project=GGE_PROJECT)
except Exception as e:
       ee.Authenticate()
       ee.Initialize(project=GGE_PROJECT)


rect = ee.Geometry.Rectangle(ROI)

# Landsat 8 Collection 2, Level 2
dataset = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .filterDate(START_DATE, END_DATE) \
    .filterBounds(rect) \
    .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER))  # 雲量の閾値でフィルタリング

# 画像のマスク処理（雲、影、雪を除去）
def mask_clouds(img):
    qa = img.select('QA_PIXEL')
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    shadow = qa.bitwiseAnd(1 << 4).eq(0)
    snow = qa.bitwiseAnd(1 << 5).eq(0)
    mask = cloud.And(shadow).And(snow)
    return img.updateMask(mask)


# LSTバンド（ST_B10）を取得し、摂氏に変換
# 参考： https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products
def calc_lst(img):
    lst = img.select(['ST_B10']).multiply(0.00341802).add(149.0).subtract(273.15)
    lst = lst.rename('LST_C')
    return lst.copyProperties(img, ['system:time_start', 'system:time_end'])


dataset_masked = dataset.map(mask_clouds)
lst_images = dataset_masked.map(calc_lst)
mean_lst = lst_images.mean()


# 期間内の画像の日付一覧を表示
dates = dataset.aggregate_array('system:time_start').getInfo()
for t in dates:
    dt = datetime.utcfromtimestamp(t / 1000)
    print(dt.strftime('%Y-%m-%d %H:%M:%S'))


# Earth EngineのタスクとしてGeoTIFFでエクスポート（Google Driveに保存）
task = ee.batch.Export.image.toDrive(
    image=mean_lst, # エクスポートする画像
    description='Hanoi_LST_Mesh', # タスクの説明。Earth Engineの「Tasks」タブで表示される名前
    folder=FOLDER_NAME, # Google Drive内の保存先フォルダ名
    fileNamePrefix=FILE_NAME_PREFIX, # 出力ファイル名の接頭辞
    ROI=rect, 
    scale=SCALE,   # 出力画像の解像度（メートル単位）
    crs=CRS,
    maxPixels=MAX_PIXELS 
)


task.start()
print("Export task started. Check your Google Drive's 'EarthEngine'フォルダ.")
