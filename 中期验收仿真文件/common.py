# -*- coding: utf-8 -*-
"""
公共模块：太阳位置天文算法、晴空辐照模型、倾斜面辐照(POA)、绘图样式。
所有仿真脚本共享该模块，保证参数与口径一致。
"""
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 站点与系统参数
LAT = 43.88          # 长春 纬度
LON = 125.35         # 长春 经度
TZ = 8.0             # 东八区

PANEL_AREA = 1.0     # m^2
PANEL_ETA = 0.19     # 组件效率
SYSTEM_LOSS = 0.88   # 线损+失配+温升综合系数

FIXED_TILT = 40.0    # 固定式最佳倾角(长春)
GROUND_ALBEDO = 0.20

DIFFUSE_RATIO = 0.30   # 晴空散射与直射水平分量之比

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "out", "figures")
DATA_DIR = os.path.join(BASE_DIR, "out", "data")
for d in (FIG_DIR, DATA_DIR):
    os.makedirs(d, exist_ok=True)

METRICS_PATH = os.path.join(BASE_DIR, "out", "metrics.json")


# ---------------------------------------------------------------- 绘图样式
def setup_style():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linestyle": "--",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.edgecolor": "#3a3a3a",
        "figure.facecolor": "white",
    })


C = {
    "fixed": "#8c8c8c",
    "dual": "#c0392b",
    "astr": "#2c7fb8",
    "photo": "#e67e22",
    "comp": "#27ae60",
    "hold": "#7f8c8d",
    "pid": "#c0392b",
    "iso": "#2980b9",
    "dead": "#27ae60",
    "pso": "#8e44ad",
    "po": "#d35400",
}


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print("  [saved]", path)
    return path


# ---------------------------------------------------------------- 天文算法
def solar_position(doy, hour_utc, lat=LAT, lon=LON):
    """NOAA 简化太阳位置算法（矢量版）。

    参数
    ----
    doy      : 年积日 (1..365)，标量或数组
    hour_utc : UTC 小数小时，标量或数组
    返回
    ----
    el : 高度角(deg)   az : 方位角(deg, 自正北顺时针)
    """
    doy = np.asarray(doy, dtype=float)
    hour_utc = np.asarray(hour_utc, dtype=float)

    g = 2.0 * np.pi / 365.0 * (doy - 1.0 + (hour_utc - 12.0) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))

    tst = hour_utc + lon / 15.0 + eqtime / 60.0        # 真太阳时
    ha = np.deg2rad((tst - 12.0) * 15.0)               # 时角
    phi = np.deg2rad(lat)

    up = np.sin(decl) * np.sin(phi) + np.cos(decl) * np.cos(phi) * np.cos(ha)
    east = -np.cos(decl) * np.sin(ha)
    north = np.sin(decl) * np.cos(phi) - np.cos(decl) * np.sin(phi) * np.cos(ha)

    el = np.rad2deg(np.arcsin(np.clip(up, -1.0, 1.0)))
    az = np.rad2deg(np.arctan2(east, north)) % 360.0
    return el, az


def sun_vector(el, az):
    """单位太阳矢量，ENU 分量 (east, north, up)。"""
    e = np.cos(np.deg2rad(el))
    return (e * np.sin(np.deg2rad(az)),
            e * np.cos(np.deg2rad(az)),
            np.sin(np.deg2rad(el)))


def airmass(el_deg):
    el = np.asarray(el_deg, dtype=float)
    elc = np.maximum(el, 0.0)
    am = 1.0 / (np.sin(np.deg2rad(elc)) + 0.50572 * (elc + 6.07995) ** -1.6364)
    return np.where(el > 0.0, am, np.inf)


def clear_sky(el_deg, doy, tau=0.70):
    """晴空模型：Meinel 直射 + 简化散射。

    返回 dni, dhi, ghi  [W/m^2]
    """
    el = np.asarray(el_deg, dtype=float)
    doy = np.asarray(doy, dtype=float)
    e0 = 1367.0 * (1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.0))
    am = airmass(el)
    dni = e0 * tau ** (np.power(am, 0.678))
    sin_el = np.sin(np.deg2rad(np.maximum(el, 0.0)))

    ghi_beam = dni * sin_el
    dhi = DIFFUSE_RATIO * ghi_beam
    mask = el <= 0.0
    dni = np.where(mask, 0.0, dni)
    dhi = np.where(mask, 0.0, dhi)
    ghi = dni * sin_el + dhi
    return dni, dhi, ghi


def poa_irradiance(dni, dhi, ghi, el, az, tilt_deg, surface_az_deg=None,
                   mode="fixed", albedo=GROUND_ALBEDO):
    """倾斜面总辐照 (POA)。

    mode='fixed'     : 固定倾角 tilt_deg，朝向 surface_az_deg（自北顺时针）
    mode='dual_axis' : 双轴，法线始终指向太阳，tilt = 90 - el
    """
    el = np.asarray(el, dtype=float)
    az = np.asarray(az, dtype=float)

    if mode == "dual_axis":
        beta = np.deg2rad(np.clip(90.0 - el, 0.0, 90.0))
        cos_aoi = np.where(el > 0.0, 1.0, 0.0)
    else:
        beta = np.full_like(el, np.deg2rad(tilt_deg), dtype=float)
        sa = np.deg2rad(surface_az_deg if surface_az_deg is not None else 180.0)
        cos_aoi = (np.sin(np.deg2rad(el)) * np.cos(beta)
                   + np.cos(np.deg2rad(el)) * np.sin(beta) * np.cos(az - sa))
        cos_aoi = np.clip(cos_aoi, 0.0, None)

    poa_beam = dni * cos_aoi
    poa_diff = dhi * (1.0 + np.cos(beta)) / 2.0
    poa_grnd = ghi * albedo * (1.0 - np.cos(beta)) / 2.0
    return poa_beam + poa_diff + poa_grnd


# ---------------------------------------------------------------- 指标收集
def update_metrics(key, value):
    data = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data[key] = value
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
