"""
Google Earth Engineを使用してLandsat 8データからバンドデータをgeotiff形式でエクスポートするプログラム

LST（地表面温度）および
バンドから様々な指標を計算するために、
各反射バンド情報をエクスポートする

エクスポートする条件は以下の通り
1. 有効ピクセル率がCLOUD_THRESHOLD以上（例：0.5 = 50%）
2. 全ピクセル数がTOTAL_PIXEL_THRESHOLD以上

注意点：
有効ピクセル率はST_B10を使用して計算している
有効ピクセルの割合はバンドにより異なる可能性がある
例： SR_B4（赤色）は、ST_B10より雲の影響を受けやすい
"""

import ee
import geemap
import pandas as pd
import geopandas as gpd
import os

# --------------------------------------
# 設定値（定数管理）
# --------------------------------------
CONFIG = {
    'GGE_PROJECT': 'master-research-465403',
    'YEAR': 2019,
    'CLOUD_THRESHOLD': 0.5,
    'TOTAL_PIXEL_THRESHOLD': 1700000,
    'EXPORT_SCALE': 30,
    'EXPORT_FOLDER_LST': 'Landsat8_LST',
    'EXPORT_FOLDER_REF': 'Landsat8_反射バンド',
    'ROI_SHP_PATH': 'workspace/data/SHP/研究対象領域/研究対象都市_行政区画.shp',
    'REFLECTANCE_BANDS': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
}

START_DATE = f"{CONFIG['YEAR']}-01-01"
END_DATE = f"{CONFIG['YEAR']}-12-31"
CSV_OUTPUT = f'image_metadata_{CONFIG["YEAR"]}.csv'
# --------------------------------------
# Earth Engine初期化
# --------------------------------------
try:
    ee.Initialize(project=CONFIG['GGE_PROJECT'])
except Exception as e:
    print(f"EE初期化エラー: {e}")
    ee.Authenticate()
    ee.Initialize(project=CONFIG['GGE_PROJECT'])

# --------------------------------------
# ROI取得
# --------------------------------------
try:
    boundary_df = gpd.read_file(CONFIG['ROI_SHP_PATH'])
    hanoi_geom = boundary_df[boundary_df['TinhThanh'] == 'Hà Nội'].iloc[0].geometry
    ROI = ee.Geometry(hanoi_geom.__geo_interface__)
except Exception as e:
    print(f"ROI取得エラー: {e}")
    raise

# --------------------------------------
# 関数定義
# --------------------------------------
def cloud_mask(image):
    qa = image.select('QA_PIXEL')
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    shadow = qa.bitwiseAnd(1 << 4).eq(0)
    cirrus = qa.bitwiseAnd(1 << 2).eq(0)
    return image.updateMask(cloud.And(shadow).And(cirrus))

def apply_scale_factors(image):
    optical = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
    lst_celsius = image.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST_Celsius')
    return image.addBands(optical, None, True).addBands(lst_celsius)

def get_valid_pixel_ratio(image):
    try:
        total = image.select('ST_B10').unmask(1).reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=ROI,
            scale=CONFIG['EXPORT_SCALE'],
            maxPixels=1e13
        ).get('ST_B10')
        masked = image.select('ST_B10').reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=ROI,
            scale=CONFIG['EXPORT_SCALE'],
            maxPixels=1e13
        ).get('ST_B10')
        valid_ratio = ee.Number(masked).divide(ee.Number(total))
        return total, valid_ratio
    except Exception as e:
        print(f"有効ピクセル率計算エラー: {e}")
        return 0, 0

def export_lst_to_drive(image, time):
    try:
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=f'L8_{time}_Hanoi_LST',
            folder=f"{CONFIG['EXPORT_FOLDER_LST']}/{CONFIG['YEAR']}",
            fileNamePrefix=f'L8_{time}_Hanoi_LST',
            scale=CONFIG['EXPORT_SCALE'],
            region=ROI,
            maxPixels=1e13,
            fileFormat='GeoTIFF'
        )
        task.start()
    except Exception as e:
        print(f"LSTエクスポートエラー: {e}")

def export_reflectance_to_drive(image, time):
    try:
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=f'L8_{time}_Hanoi_Reflectance',
            folder=f"{CONFIG['EXPORT_FOLDER_REF']}/{CONFIG['YEAR']}",
            fileNamePrefix=f'L8_{time}_Hanoi_Reflectance',
            scale=CONFIG['EXPORT_SCALE'],
            region=ROI,
            maxPixels=1e13,
            fileFormat='GeoTIFF'
        )
        task.start()
    except Exception as e:
        print(f"反射バンドエクスポートエラー: {e}")

def create_metadata(date_str, total, valid_ratio, exported, time_csv):
    return {
        '日時': date_str,
        '全体ピクセル数': int(total),
        '有効ピクセル率': round(valid_ratio, 3),
        '出力有無': exported,
        '観測時刻': time_csv
    }

def export_image_task(image, date_str, metadata_list):
    total, valid_ratio = get_valid_pixel_ratio(image)
    total = total.getInfo() if hasattr(total, 'getInfo') else total
    valid_ratio = valid_ratio.getInfo() if hasattr(valid_ratio, 'getInfo') else valid_ratio
    exported = False

    if valid_ratio >= CONFIG['CLOUD_THRESHOLD'] and total >= CONFIG['TOTAL_PIXEL_THRESHOLD']:
        lst_img = image.select('LST_Celsius').clip(ROI)
        reflectance_img = image.select(CONFIG['REFLECTANCE_BANDS']).clip(ROI)
        time = ee.Date(image.get('system:time_start')).format('YYYYMMdd_HHmmss').getInfo()
        export_lst_to_drive(lst_img, time)
        export_reflectance_to_drive(reflectance_img, time)
        exported = True

    time_csv = ee.Date(image.get('system:time_start')).format('HH:mm:ss').getInfo()
    metadata_list.append(create_metadata(date_str, total, valid_ratio, exported, time_csv))

# --------------------------------------
# メイン処理
# --------------------------------------
os.makedirs(CONFIG['EXPORT_FOLDER_LST'], exist_ok=True)
os.makedirs(CONFIG['EXPORT_FOLDER_REF'], exist_ok=True)

collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .filterBounds(ROI) \
    .filterDate(START_DATE, END_DATE) \
    .map(cloud_mask) \
    .map(apply_scale_factors)

image_list = collection.toList(collection.size())
metadata = []

for i in range(collection.size().getInfo()):
    img = ee.Image(image_list.get(i))
    date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    try:
        export_image_task(img, date, metadata)
    except Exception as e:
        print(f"画像処理エラー: {e}")

df = pd.DataFrame(metadata)
df = df.drop_duplicates(subset=['日時', '観測時刻'])
df.to_csv(CSV_OUTPUT, index=False)