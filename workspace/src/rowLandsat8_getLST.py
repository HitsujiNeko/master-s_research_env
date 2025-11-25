#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_bt_l8_b10_only.py
Landsat 8 (Collection 2, Tier 1, Level-1) の Band10 と MTL.txt から
輝度温度（BT, °C）の GeoTIFF を作る最小構成スクリプト。

【使い方】


【出力】
- L8_B10_BT_C.tif（輝度温度, °C, float32, NaN=nodata）
"""

import os
import re
import argparse
import numpy as np
import rasterio

DIR = "workspace/data/geotiff/Landsat8/level1_Landsat8"

# --------------------
# MTL パーサ（KEY = VALUE をざっくり辞書化）
# --------------------
def parse_mtl(filepath: str) -> dict:
    kv = {}
    with open(filepath, "r") as f:
        for line in f:
            if "=" not in line:
                continue
            k, v = [s.strip() for s in line.strip().split("=", 1)]
            # 文字列のクオートを外す／数値は変換
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            else:
                try:
                    if any(c in v for c in (".", "e", "E")):
                        v = float(v)
                    else:
                        v = int(v)
                except Exception:
                    pass
            kv[k] = v
    return kv

# --------------------
# 計算ユーティリティ
# --------------------
def calc_TOA(dn: np.ndarray, MULT: float, AL: float) -> np.ndarray:
    """Radiance: TOA = MULT * DN + AL"""
    return MULT * dn + AL

def radiance_to_btK(TOA: np.ndarray, K1: float, K2: float) -> np.ndarray:
    """Brightness Temperature (K): BT = K2 / ln(K1/TOA + 1)"""
    out = np.full_like(TOA, np.nan, dtype="float64")
    valid = TOA > 0
    out[valid] = K2 / np.log((K1 / TOA[valid]) + 1.0)
    return out

# --------------------
# パス推定（ディレクトリから自動検出）
# --------------------
def guess_paths_from_dir(d: str):
    mtl = None
    b10 = None
    for fn in os.listdir(d):
        up = fn.upper()
        if up.endswith("_MTL.TXT"):
            mtl = os.path.join(d, fn)
        elif re.search(r"_B10\.TIF$", up):
            b10 = os.path.join(d, fn)
    return mtl, b10

# --------------------
# メイン
# --------------------


    # --- # ...existing code...
def main():
    ap = argparse.ArgumentParser(description="Compute BT (°C) from Landsat 8 L1 (C2 T1) Band10.")
    ap.add_argument("--dir", type=str, default=None, help="シーンフォルダ（MTL/B10 を自動検出）")
    ap.add_argument("--mtl", type=str, default=None, help="MTL.txt パス（--dir未使用時）")
    ap.add_argument("--b10", type=str, default=None, help="Band10 TIF パス（--dir未使用時）")
    ap.add_argument("--out", type=str, default="L8_B10_BT_C.tif", help="出力 GeoTIFF（BT, °C）")
    args = ap.parse_args()

    # 引数優先 → 個別指定（--mtl/--b10） → デフォルトDIRを参照
    if args.dir:
        mtl_path, b10_path = guess_paths_from_dir(args.dir)
    elif args.mtl or args.b10:
        mtl_path, b10_path = args.mtl, args.b10
    else:
        mtl_path, b10_path = guess_paths_from_dir(DIR)

    if not mtl_path or not os.path.exists(mtl_path):
        raise FileNotFoundError(f"MTL.txt が見つかりません。--dir または --mtl を確認してください。検索パス: {args.dir or DIR}")
    if not b10_path or not os.path.exists(b10_path):
        raise FileNotFoundError(f"Band10 が見つかりません。--dir または --b10 を確認してください。検索パス: {args.dir or DIR}")

    # ...existing code...MTL 読み取り（必要な定数） ---
    mtl = parse_mtl(mtl_path)
    try:
        MULT = float(mtl["RADIANCE_MULT_BAND_10"])
        ADD = float(mtl["RADIANCE_ADD_BAND_10"])
        K1 = float(mtl["K1_CONSTANT_BAND_10"])
        K2 = float(mtl["K2_CONSTANT_BAND_10"])
    except KeyError as e:
        raise KeyError(f"MTL に必要なキーが見つかりません: {e}")

    # --- Band10 読み込み ---
    with rasterio.open(b10_path) as src10:
        profile = src10.profile
        dn10 = src10.read(1).astype("float64")
        nodata10 = src10.nodata

    # マスク（NoData）
    mask = np.zeros_like(dn10, dtype=bool)
    if nodata10 is not None:
        mask |= (dn10 == nodata10)

    # DN -> Radiance
    L10 = calc_TOA(dn10, MULT, ADD)
    mask |= (L10 <= 0)  # 物理的に不正な画素も除外

    # Radiance -> BT (K) -> BT (°C)
    btK = radiance_to_btK(L10, K1, K2)
    btK[mask] = np.nan
    btC = btK - 273.15


    # 出力（float32 / NaN を nodata 扱い）
    out_profile = profile.copy()
    out_profile.update(dtype="float32", nodata=np.nan)
    with rasterio.open(args.out, "w", **out_profile) as dst:
        dst.write(btC.astype("float32"), 1)

    print(f"[OK] BT saved (°C): {args.out}")
    print(f"  MTL: MULT={MULT}, ADD={ADD}, K1={K1}, K2={K2}")
    print(f"  Input: {b10_path}")

if __name__ == "__main__":
    main()
