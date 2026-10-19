# -*- coding: utf-8 -*-
"""
成果 1：太阳位置天文算法验证与全年视日轨迹
对应申请书：研究内容 2「视日运动轨迹算法」、特色 2「内置太阳位置数据表」
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from common import (LAT, LON, C, setup_style, save, solar_position,
                    update_metrics)


def local_hours(n=1440, hour_start=0.0, hour_end=24.0):
    return np.linspace(hour_start, hour_end, n)


def doy_of(month, day):
    # 2026 非闰年
    cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    return cum[month - 1] + day


CASES = [
    ("春分 03-20", doy_of(3, 20), "#e67e22"),
    ("夏至 06-21", doy_of(6, 21), "#c0392b"),
    ("秋分 09-23", doy_of(9, 23), "#27ae60"),
    ("冬至 12-21", doy_of(12, 21), "#2c7fb8"),
]


def main():
    setup_style()

    # ------ 全年逐时轨迹 ------
    days = np.arange(1, 366)
    hours = np.arange(0, 24, 0.25)
    D, H = np.meshgrid(days, hours, indexing="ij")
    el_all, az_all = solar_position(D, H - 8.0)   # 本地时间 → UTC
    daylight = el_all > 0

    # ------ 逐日日出/日落/正午高度角 ------
    sunrise, sunset, noon_el = [], [], []
    for d in days:
        hh = np.arange(0, 24, 1 / 60.0)
        e, _ = solar_position(np.full_like(hh, d), hh - 8.0)
        up = np.where(e > 0)[0]
        sunrise.append(hh[up[0]] if len(up) else np.nan)
        sunset.append(hh[up[-1]] if len(up) else np.nan)
        noon_el.append(e.max())

    sunrise = np.array(sunrise)
    sunset = np.array(sunset)
    noon_el = np.array(noon_el)

    # ------ 理论值校验：正午高度角 = 90 - |lat - decl| ------
    g = 2 * np.pi / 365.0 * (days - 1)
    decl = np.rad2deg(0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
                      - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
                      - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    noon_theory = 90.0 - np.abs(LAT - decl)
    # 仅在正午太阳位于南侧（长春全年成立）时适用；取日最大值对应时刻比较
    err = np.abs(noon_el - noon_theory)
    err = err[~np.isnan(err)]

    # =========================================================== 绘图
    fig = plt.figure(figsize=(13.2, 8.6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 1.0],
                  hspace=0.34, wspace=0.26)

    # (a) 典型日 高度角-方位角 轨迹（极坐标式直角投影）
    ax = fig.add_subplot(gs[0, 0])
    for label, d, col in CASES:
        hh = np.arange(0, 24, 1 / 60.0)
        e, a = solar_position(np.full_like(hh, d), hh - 8.0)
        m = e > 0
        ax.plot(a[m], e[m], color=col, lw=2.0, label=label)
        i = np.where(m)[0]
        ax.scatter([a[i[0]], a[i[-1]]], [e[i[0]], e[i[-1]]],
                   color=col, s=26, zorder=5)
        ax.annotate(label.split()[0], (a[i[len(i) // 2]], e[i[len(i) // 2]]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=8.5, color=col)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlim(60, 300)
    ax.set_ylim(0, 72)
    ax.set_xlabel("方位角 (°, 自正北顺时针)")
    ax.set_ylabel("高度角 (°)")
    ax.set_title("(a) 典型日太阳视运动轨迹（长春 43.88°N）")
    ax.legend(loc="upper left", ncol=2, framealpha=0.92)

    # (b) 全年日照窗口 + 昼长
    ax = fig.add_subplot(gs[0, 1])
    ax.fill_between(days, sunrise, sunset, color="#f6c85f", alpha=0.55,
                    label="可日照时段")
    ax.plot(days, sunrise, color="#4a4a4a", lw=1.1, label="日出时刻")
    ax.plot(days, sunset, color="#4a4a4a", lw=1.1, label="日落时刻")
    ax.set_xlim(1, 365)
    ax.set_ylim(0, 21)
    ax.set_xlabel("年积日 (day of year)")
    ax.set_ylabel("本地时刻 (h)")
    ax.set_title("(b) 全年日出、日落时刻与昼长")
    ax.axhline(12, color="k", lw=0.6, ls=":")
    ax.text(8, 12.3, "12:00", fontsize=8, color="k")

    ax2 = ax.twinx()
    daytime = sunset - sunrise
    ax2.plot(days, daytime, color="#c0392b", lw=1.6, ls="--", label="昼长")
    ax2.set_ylabel("昼长 (h)", color="#c0392b")
    ax2.tick_params(axis="y", colors="#c0392b")
    ax2.set_ylim(6, 17)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, fontsize=8.2)

    # (c) 正午高度角：算法 vs 解析解
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(days, noon_theory, color="#2c7fb8", lw=2.2, label="解析解 90°-|φ-δ|")
    ax.plot(days[::4], noon_el[::4], "o", ms=3.0, color="#c0392b",
            label="NOAA 算法（抽样）")
    ax.set_xlim(1, 365)
    ax.set_xlabel("年积日")
    ax.set_ylabel("正午太阳高度角 (°)")
    ax.set_title(f"(c) 算法校验：最大偏差 {err.max():.3f}°，平均 {err.mean():.3f}°")
    ax.legend(loc="lower center", ncol=2, fontsize=8.5)

    # (d) 分轴角速度——两轴需求相反，是分轴非均匀调度的依据
    ax = fig.add_subplot(gs[1, 1])
    hh = np.arange(4, 20, 1 / 60.0)
    style = {172: ("夏至", "-", "#c0392b"), 355: ("冬至", "--", "#2c7fb8")}
    for d, (nm, ls, col) in style.items():
        e, a = solar_position(np.full_like(hh, d), hh - 8.0)
        a_unwrap = np.unwrap(np.deg2rad(a)) * 180.0 / np.pi
        de = np.abs(np.gradient(e, hh))
        da = np.abs(np.gradient(a_unwrap, hh)) * np.cos(np.deg2rad(e))
        m = e > 1
        ax.plot(hh[m], de[m], ls=ls, color=col, lw=2.0,
                label=f"{nm} 高度角速率 |dε/dt|")
        ax.plot(hh[m], da[m], ls=ls, color="#e67e22" if d == 172 else "#16a085",
                lw=1.4, alpha=0.95, label=f"{nm} 方位角等效速率 cosε·|dA/dt|")
    ax.axvline(12, color="k", lw=0.7, ls=":")
    ax.set_xlim(4, 20)
    ax.set_ylim(0, 26)
    ax.set_xlabel("本地时刻 (h)")
    ax.set_ylabel("角速度 (°/h)")
    ax.set_title("(d) 分轴角速度：高度角早晚快、方位角正午快")
    ax.legend(loc="upper center", ncol=2, fontsize=7.6)

    fig.suptitle("图 1  太阳位置天文算法验证与全年视日轨迹", fontsize=15,
                 fontweight="bold", y=0.985)
    save(fig, "fig1_sun_path.png")

    # ------ 导出查找表样例 ------
    tbl_days = np.array([80, 172, 266, 355])
    tbl_hours = np.arange(6, 18.01, 0.25)
    tab = np.zeros((len(tbl_days), len(tbl_hours), 2))
    for i, d in enumerate(tbl_days):
        e, a = solar_position(np.full_like(tbl_hours, d), tbl_hours - 8.0)
        tab[i, :, 0] = e
        tab[i, :, 1] = a
    np.savez_compressed(f"{__import__('common').DATA_DIR}/sun_table_sample.npz",
                        days=tbl_days, hours=tbl_hours, el=tab[:, :, 0],
                        az=tab[:, :, 1])

    update_metrics("1_sun_algorithm", {
        "站点": "长春 43.88°N, 125.35°E",
        "算法": "NOAA 简化太阳位置算法",
        "正午高度角最大偏差(deg)": round(float(err.max()), 4),
        "正午高度角平均偏差(deg)": round(float(err.mean()), 4),
        "全年可照时数(h)": round(float(np.nansum(sunset - sunrise)), 1),
        "最长昼长(h)": round(float(np.nanmax(sunset - sunrise)), 3),
        "最短昼长(h)": round(float(np.nanmin(sunset - sunrise)), 3),
        "查找表分辨率(min)": 15,
    })
    print(f"  正午高度角偏差 max={err.max():.4f}° mean={err.mean():.4f}°")
    print(f"  全年可照时数 = {np.nansum(sunset - sunrise):.1f} h")


if __name__ == "__main__":
    main()
