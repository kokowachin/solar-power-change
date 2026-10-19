# -*- coding: utf-8 -*-
"""
成果 2：双轴复合追踪 vs 固定式 / 单轴 的年发电量对比
对应申请书：立项背景「双轴跟踪较固定式提升 20%~30% 年发电量」的数据支撑
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from common import (LAT, LON, PANEL_AREA, PANEL_ETA, SYSTEM_LOSS,
                    FIXED_TILT, GROUND_ALBEDO, C, setup_style, save,
                    solar_position, clear_sky, update_metrics, FIG_DIR,
                    DATA_DIR)


def build_year(year=2026):
    """逐时太阳位置与辐照。晴空模型按长春典型年逐月总辐照标定晴空指数。"""
    local = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h")
    utc = local - pd.Timedelta(hours=8)
    doy = utc.dayofyear.values.astype(float)
    hu = utc.hour.values + utc.minute.values / 60.0
    el, az = solar_position(doy, hu)
    dni, dhi, ghi = clear_sky(el, doy)

    # 长春典型年逐月水平面总辐照 (kWh/m2/月)
    TARGET_MONTH = np.array([55, 75, 115, 145, 165, 160, 145, 140, 130, 100, 65, 45],
                            dtype=float)
    mon = pd.DatetimeIndex(local).month
    cs_month = np.bincount(mon, weights=ghi, minlength=13)[1:] / 1000.0
    k = np.where(cs_month > 1e-6, TARGET_MONTH / np.maximum(cs_month, 1e-6), 1.0)
    k = np.clip(k, 0.25, 1.0)          # 晴空指数物理上限为 1
    k_hour = k[mon - 1]

    dni = dni * k_hour
    dhi = dhi * k_hour
    ghi = dni * np.sin(np.deg2rad(np.maximum(el, 0))) + dhi
    return local, doy, hu, el, az, dni, dhi, ghi


def poa_fixed(el, az, dni, dhi, ghi, tilt=FIXED_TILT):
    beta = np.deg2rad(tilt)
    cos_aoi = (np.sin(np.deg2rad(el)) * np.cos(beta)
               + np.cos(np.deg2rad(el)) * np.sin(beta)
               * np.cos(np.deg2rad(az) - np.pi))
    cos_aoi = np.clip(cos_aoi, 0, None)
    return (dni * cos_aoi
            + dhi * (1 + np.cos(beta)) / 2
            + ghi * GROUND_ALBEDO * (1 - np.cos(beta)) / 2)


def poa_single_axis_ns(el, az, dni, dhi, ghi):
    """单轴：水平南北向轴，绕轴跟踪东西（法线位于东西-高度平面）。"""
    e = np.cos(np.deg2rad(el))
    s_e = e * np.sin(np.deg2rad(az))      # east
    s_u = np.sin(np.deg2rad(el))          # up
    gamma = np.arctan2(s_e, s_u)          # 法线偏离天顶角
    cos_aoi = np.sqrt(np.clip(s_e ** 2 + s_u ** 2, 0, 1))
    beta = np.abs(gamma)
    return (dni * cos_aoi
            + dhi * (1 + np.cos(beta)) / 2
            + ghi * GROUND_ALBEDO * (1 - np.cos(beta)) / 2)


def poa_dual(el, az, dni, dhi, ghi):
    beta = np.deg2rad(np.clip(90.0 - el, 0.0, 90.0))
    cos_aoi = np.where(el > 0, 1.0, 0.0)
    return (dni * cos_aoi
            + dhi * (1 + np.cos(beta)) / 2
            + ghi * GROUND_ALBEDO * (1 - np.cos(beta)) / 2)


def to_power(poa):
    return poa * PANEL_AREA * PANEL_ETA * SYSTEM_LOSS


SCEN = [
    ("固定式 40°倾角", "fixed", C["fixed"]),
    ("单轴（水平南北轴）", "single", "#2c7fb8"),
    ("双轴复合追踪", "dual", C["dual"]),
]


def main():
    setup_style()
    local, doy, hu, el, az, dni, dhi, ghi = build_year()

    poa = {
        "fixed": poa_fixed(el, az, dni, dhi, ghi),
        "single": poa_single_axis_ns(el, az, dni, dhi, ghi),
        "dual": poa_dual(el, az, dni, dhi, ghi),
    }
    power = {k: to_power(v) for k, v in poa.items()}

    month = pd.DatetimeIndex(local).month
    monthly = {k: np.bincount(month, weights=power[k], minlength=13)[1:] / 1000.0
               for k in power}                     # kWh
    annual = {k: float(v.sum() / 1000.0) for k, v in power.items()}   # kWh
    daily_avg = {k: v.sum() / 365.0 / 1000.0 for k, v in power.items()}

    gain_dual = (annual["dual"] / annual["fixed"] - 1) * 100
    gain_single = (annual["single"] / annual["fixed"] - 1) * 100

    # 单独统计晴空条件下「跟踪自身带来的增益」= 仅看辐照，不受组件模型影响
    poa_annual = {k: float(v.sum() / 1000.0) for k, v in poa.items()}
    gain_poa = (poa_annual["dual"] / poa_annual["fixed"] - 1) * 100

    # ---------------- 绘图 ----------------
    fig = plt.figure(figsize=(13.4, 8.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.24)

    # (a) 月度发电量
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(1, 13)
    w = 0.27
    for i, (name, key, col) in enumerate(SCEN):
        ax.bar(x + (i - 1) * w, monthly[key], width=w, color=col,
               label=name, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}" for m in x])
    ax.set_xlabel("月份")
    ax.set_ylabel("月发电量 (kWh)")
    ax.set_title("(a) 逐月发电量对比（1 m² 组件）", loc="left")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_ylim(0, max(monthly["dual"]) * 1.42)

    ax2 = ax.twinx()
    m_gain = (monthly["dual"] / monthly["fixed"] - 1) * 100
    ax2.plot(x, m_gain, "o--", color="#8e44ad", ms=4, lw=1.4,
             label="双轴相对固定增益")
    ax2.set_ylabel("增益 (%)", color="#8e44ad")
    ax2.tick_params(axis="y", colors="#8e44ad")
    ax2.set_ylim(0, 90)
    ax2.grid(False)
    ax2.legend(loc="upper right", fontsize=8.5)

    # (b)(c) 典型日功率曲线
    days = [("(b) 夏至日（6-21）功率曲线", 172), ("(c) 冬至日（12-21）功率曲线", 355)]
    local_doy = pd.DatetimeIndex(local).dayofyear.values
    for slot, (title, d) in zip([gs[0, 1], gs[1, 0]], days):
        ax = fig.add_subplot(slot)
        m = np.where(local_doy == d)[0]
        hh = (local[m].hour + local[m].minute / 60.0).values
        order = np.argsort(hh)
        hh = hh[order]
        m = m[order]
        for name, key, col in SCEN:
            ax.plot(hh, power[key][m], color=col, lw=2.0, label=name)
        ax.fill_between(hh, 0, power["dual"][m], color=C["dual"], alpha=0.10)
        ax.set_xlim(0, 24)
        ax.set_xlabel("本地时刻 (h)")
        ax.set_ylabel("直流功率 (W)")
        ax.set_title(title, loc="left")
        ax.legend(loc="upper left", fontsize=8.5)

    # (d) 年发电量汇总
    ax = fig.add_subplot(gs[1, 1])
    names = [s[0] for s in SCEN]
    vals = [annual[s[1]] for s in SCEN]
    cols = [s[2] for s in SCEN]
    bars = ax.barh(names, vals, color=cols, height=0.52, edgecolor="white")
    for b, v, key in zip(bars, vals, [s[1] for s in SCEN]):
        pct = (annual[key] / annual["fixed"] - 1) * 100
        tag = "基准" if key == "fixed" else f"+{pct:.1f}%"
        ax.text(v + max(vals) * 0.015, b.get_y() + b.get_height() / 2,
                f"{v:.0f} kWh/年  ({tag})", va="center", fontsize=9.5)
    ax.set_xlim(0, max(vals) * 1.42)
    ax.set_xlabel("年发电量 (kWh)")
    ax.set_title("(d) 年发电量与相对增益", loc="left")
    ax.invert_yaxis()

    fig.suptitle("图 2  不同跟踪方式的年发电量对比（长春，1 m² 组件，晴空模型）",
                 fontsize=14.5, fontweight="bold", y=0.985)
    save(fig, "fig2_energy_gain.png")

    # ---------------- 导出 ----------------
    df = pd.DataFrame({
        "local_time": local,
        "el": el, "az": az, "dni": dni, "dhi": dhi, "ghi": ghi,
        "poa_fixed": poa["fixed"], "poa_single": poa["single"], "poa_dual": poa["dual"],
        "p_fixed": power["fixed"], "p_single": power["single"], "p_dual": power["dual"],
    })
    df.to_csv(f"{DATA_DIR}/year_power.csv", index=False, encoding="utf-8-sig")

    update_metrics("2_energy_gain", {
        "站点": "长春 43.88°N, 125.35°E",
        "年水平面总辐照(kWh/m2)": round(float(ghi.sum() / 1000.0), 1),
        "固定式年发电量(kWh)": round(annual["fixed"], 1),
        "单轴年发电量(kWh)": round(annual["single"], 1),
        "双轴年发电量(kWh)": round(annual["dual"], 1),
        "单轴相对固定增益(%)": round(gain_single, 2),
        "双轴相对固定增益(%)": round(gain_dual, 2),
        "双轴相对固定辐照增益(%)": round(gain_poa, 2),
        "双轴日均发电量(kWh)": round(daily_avg["dual"], 2),
        "最大月增益(%)": round(float(m_gain.max()), 1),
        "最小月增益(%)": round(float(m_gain.min()), 1),
    })
    print(f"  固定 {annual['fixed']:.0f} kWh  单轴 {annual['single']:.0f} kWh  "
          f"双轴 {annual['dual']:.0f} kWh")
    print(f"  双轴增益 {gain_dual:.1f}%  单轴增益 {gain_single:.1f}%  "
          f"月增益 {m_gain.min():.0f}%~{m_gain.max():.0f}%")


if __name__ == "__main__":
    main()
