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
YEAR = 2022
INPUT_FOLDER = f'workspace/data/geotiff/LST_{YEAR}'
OUTPUT_FOLDER = f'workspace/data/indexes/{YEAR}'
CSV_OUTPUT = f'workspace/data/indexes/statistics_{YEAR}.csv'

# -------------------------------
# 作成する指標関数（拡張可能）
# -------------------------------

def calculate_ndvi(red, nir):
    ndvi = (nir - red) / (nir + red + 1e-10)
    return ndvi

def calculate_ndwi(green, nir):
    ndwi = (green - nir) / (green + nir + 1e-10)
    return ndwi

def calculate_ndbi(swir, nir):
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

        band_dict = {
            'SR_B2': None,  # Blue
            'SR_B3': None,  # Green
            'SR_B4': None,  # Red
            'SR_B5': None,  # NIR
            'SR_B6': None,  # SWIR1
        }

        for i, name in enumerate(src.descriptions):
            if name in band_dict:
                band_dict[name] = bands[i]

        # 必要なバンドが揃っているか確認
        if all(band_dict.values()):
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

# -------------------------------
# 統計CSV出力
# -------------------------------
df = pd.DataFrame(records)
df.to_csv(CSV_OUTPUT, index=False)
