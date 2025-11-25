"""
Landsat8のバンドから各種指標を計算し、GeoTIFFで保存するスクリプト

算出する指標：
- NDVI (Normalized Difference Vegetation Index)
- NDWI (Normalized Difference Water Index)
- NDBI (Normalized Difference Built-up Index)


"""
import os
import rasterio
import numpy as np
import pandas as pd
from glob import glob

# -------------------------------
# パラメータ設定
# -------------------------------
YEAR = 2023
INPUT_FOLDER = f'workspace/data/geotiff/Landsat8/reflectance/{YEAR}'
OUTPUT_FOLDER = f'workspace/data/geotiff/Landsat8/indexes/{YEAR}'
CSV_OUTPUT = f'workspace/data/csv/index_statistics_{YEAR}.csv'

# -------------------------------
# 作成する指標関数
# -------------------------------

def calculate_ndvi(red, nir):
    # 正規化植生指数
    ndvi = (nir - red) / (nir + red + 1e-10)
    return ndvi

def calculate_ndwi(green, nir):
    # 正規化水分指数
    ndwi = (green - nir) / (green + nir + 1e-10)
    return ndwi

def calculate_ndbi(swir, nir):
    # 正規化建物指数
    ndbi = (swir - nir) / (swir + nir + 1e-10)
    return ndbi

# -------------------------------
# 画像ごとの処理
# -------------------------------
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
records = []

for path in sorted(glob(os.path.join(INPUT_FOLDER, '*.tif'))):
    with rasterio.open(path) as src:
        profile = src.profile
        bands = src.read()
        print(np.nanmin(bands), np.nanmax(bands))
        print (f'バンド名の確認: {src.descriptions} ')

        band_dict = {
            'SR_B2': bands[1],  # Blue
            'SR_B3': bands[2],  # Green
            'SR_B4': bands[3],  # Red
            'SR_B5': bands[4],  # NIR
            'SR_B6': bands[5],  # SWIR1
        }


        # 必要なバンドが揃っているか確認
        if all(b is not None for b in band_dict.values()):
            ndvi = calculate_ndvi(band_dict['SR_B4'], band_dict['SR_B5'])
            ndwi = calculate_ndwi(band_dict['SR_B3'], band_dict['SR_B5'])
            ndbi = calculate_ndbi(band_dict['SR_B6'], band_dict['SR_B5'])

            index_stack = np.stack([ndvi, ndwi, ndbi])
            index_names = ['NDVI', 'NDWI', 'NDBI']

            for i, name in enumerate(index_names):
                output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(path).replace('.tif', f'_{name}.tif'))
                profile.update(dtype=rasterio.float32, count=1)
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(index_stack[i].astype(np.float32), 1)

            # 統計量
            stats = {
                'filename': os.path.basename(path),
                'NDVI_min': float(np.nanmin(ndvi)),
                'NDVI_max': float(np.nanmax(ndvi)),
                'NDVI_mean': float(np.nanmean(ndvi)),
                'NDWI_min': float(np.nanmin(ndwi)),
                'NDWI_max': float(np.nanmax(ndwi)),
                'NDWI_mean': float(np.nanmean(ndwi)),
                'NDBI_min': float(np.nanmin(ndbi)),
                'NDBI_max': float(np.nanmax(ndbi)),
                'NDBI_mean': float(np.nanmean(ndbi)),
            }
            records.append(stats)
            print(f"{path}内の指標計算と保存が完了しました。")
        else:
            print(f"{path}内のファイルに必要なバンドが揃っていません。")

# -------------------------------
# 統計CSV出力
# -------------------------------
os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
df = pd.DataFrame(records)
df.to_csv(CSV_OUTPUT, index=False)
