# -*- coding: utf-8 -*-
"""
成果 5：非均匀间歇跟踪调度与系统功耗优化
对应申请书：创新点 2、研究内容第 3 项「自适应唤醒策略」

核心思想：太阳角速度随时间变化（高度角早晚快、方位角正午快），
固定采样周期要么在最坏时刻失准、要么在全天大量冗余动作。
按瞬时角速度反解采样周期，可在满足 0.5° 误差约束的前提下显著减少动作次数。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from common import (C, setup_style, save, solar_position, update_metrics, DATA_DIR)
from fig2_energy import build_year, poa_dual, to_power

DT = 30.0
DOY = 172
EPS_MAX = 0.5              # 允许跟踪偏差 (°)

# ---- 功耗模型（20 W ~ 100 W 级小型双轴云台）----
P_SLEEP = 16.5e-6          # W
E_MCU = 80e-3 * 0.6        # J  唤醒计算
E_DRIVER = 0.15 * 1.0      # J  驱动静态
E_MOTOR = 12.0 * 1.2 * 1.5  # J 单次转动（1.5 轴平均）

SCHEMES = [
    ("固定 5 min", 300.0, "#8c8c8c"),
    ("固定 2 min", 120.0, "#2980b9"),
    ("非均匀自适应", None, "#27ae60"),
]


def sun_rate():
    """返回当日逐时太阳分轴角速度 (°/h) 与时间轴。"""
    t = np.arange(0, 24, DT / 3600.0)
    el, az = solar_position(np.full_like(t, DOY), t - 8.0)
    az_u = np.unwrap(np.deg2rad(az)) * 180 / np.pi
    d_el = np.gradient(el, t)
    d_az = np.gradient(az_u, t) * np.cos(np.deg2rad(el))
    omega = np.maximum(np.abs(d_el), np.abs(d_az))
    return t, el, az, d_el, d_az, omega


def main():
    setup_style()
    t, el, az, d_el, d_az, omega = sun_rate()
    day = el > 1.0
    t_day = t[day]
    w_day = omega[day]
    daylight_h = float(t_day[-1] - t_day[0])

    # 自适应周期
    T_adapt = np.clip(EPS_MAX / np.maximum(w_day, 1e-6) * 3600.0, 60.0, 900.0)

    rows = []
    for name, Tfix, col in SCHEMES:
        if Tfix is None:
            T = T_adapt
            n_wake = float(np.trapezoid(1.0 / T, t_day * 3600.0))
        else:
            T = np.full_like(t_day, Tfix)
            n_wake = daylight_h * 3600.0 / Tfix
        err_env = w_day * T / 3600.0
        e_day = n_wake * (E_MOTOR + E_MCU + E_DRIVER) + P_SLEEP * 86400.0
        rows.append(dict(name=name, T=T, n=n_wake, e=e_day,
                         err_max=float(err_env.max()),
                         p_avg=e_day / 86400.0, color=col))

    # 当日发电量（沿用成果 2 的辐照与组件模型）
    local, doy, hu, el2, az2, dni, dhi, ghi = build_year()
    import pandas as pd
    m = pd.DatetimeIndex(local).dayofyear.values == DOY
    gen_wh = float(to_power(poa_dual(el2, az2, dni, dhi, ghi))[m].sum())

    fig = plt.figure(figsize=(13.6, 8.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.26)

    # (a) 采样周期
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_day, T_adapt / 60.0, color=C["comp"], lw=2.2, label="自适应周期")
    ax.axhline(5.0, color="#8c8c8c", ls="--", lw=1.2, label="固定 5 min")
    ax.axhline(2.0, color="#2980b9", ls="--", lw=1.2, label="固定 2 min")
    ax.set_xlim(4, 20)
    ax.set_ylim(0, 8)
    ax.set_xlabel("本地时刻 (h)")
    ax.set_ylabel("采样周期 (min)")
    ax.set_title("(a) 自适应采样周期（约束：单次间隔偏差 ≤ 0.5°）", loc="left")
    ax.legend(loc="upper right", ncol=3, fontsize=8.2)
    ax.annotate("正午方位角速率最大\n→ 周期最短", xy=(12.0, T_adapt.min() / 60.0),
                xytext=(13.2, 4.4), fontsize=8, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))
    ax.annotate("清晨/傍晚综合角速度较低\n→ 周期可放宽", xy=(5.2, T_adapt[t_day < 5.6].max() / 60.0),
                xytext=(6.4, 6.2), fontsize=8, color="#2c7fb8",
                arrowprops=dict(arrowstyle="->", color="#2c7fb8", lw=0.8))

    # (b) 误差包络
    ax = fig.add_subplot(gs[0, 1])
    for r in rows:
        ax.plot(t_day, r["T"] * w_day / 3600.0, color=r["color"], lw=1.9,
                label=f"{r['name']}（最大 {r['err_max']:.2f}°）")
    ax.axhline(EPS_MAX, color="#8e44ad", ls="--", lw=1.1)
    ax.text(4.1, EPS_MAX + 0.05, "约束 0.5°", fontsize=8.5, color="#8e44ad")
    ax.set_xlim(4, 20)
    ax.set_ylim(0, 2.4)
    ax.set_xlabel("本地时刻 (h)")
    ax.set_ylabel("单次间隔内最大偏差 (°)")
    ax.set_title("(b) 跟踪偏差包络：固定 5 min 在正午超差", loc="left")
    ax.legend(loc="upper left", fontsize=8.2)

    # (c) 日耗电与动作次数
    ax = fig.add_subplot(gs[1, 0])
    names = [r["name"] for r in rows]
    e_wh = [r["e"] / 3600.0 for r in rows]
    n_w = [r["n"] for r in rows]
    x = np.arange(len(names))
    bars = ax.bar(x, e_wh, width=0.5, color=[r["color"] for r in rows],
                  edgecolor="white")
    for b, v, nw in zip(bars, e_wh, n_w):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015,
                f"{v:.2f} Wh\n{nw:.0f} 次", ha="center", fontsize=8.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("日耗电 (Wh)")
    ax.set_ylim(0, max(e_wh) * 1.32)
    ax.set_title("(c) 日耗电与电机动作次数", loc="left")

    # (d) 汇总表
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    cell = [
        [f"{r['n']:.0f}" for r in rows],
        [f"{r['e']/3600.0:.2f}" for r in rows],
        [f"{r['p_avg']*1000:.1f}" for r in rows],
        [f"{r['err_max']:.2f}" for r in rows],
        [f"{r['e']/3600.0/gen_wh*100:.2f}" for r in rows],
    ]
    rlab = ["日动作次数", "日耗电 (Wh)", "平均功率 (mW)",
            "最大跟踪偏差 (°)", "占日发电量 (%)"]
    tb = ax.table(cellText=cell, rowLabels=rlab, colLabels=names,
                  loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(9.4)
    tb.scale(1.0, 1.95)
    for (r, c), cc in tb.get_celld().items():
        cc.set_edgecolor("#cccccc")
        if r == 0:
            cc.set_facecolor("#eeeeee")
            cc.set_text_props(fontweight="bold")
        if c == 3 and r > 0:
            cc.set_facecolor("#eaf7ef")
    ax.set_title(f"(d) 调度方案综合对比（当日发电 {gen_wh/1000:.2f} kWh）", loc="left")

    fig.suptitle("图 5  非均匀间歇跟踪调度与功耗优化（长春夏至日，双轴 100 W 组件）",
                 fontsize=14.5, fontweight="bold", y=0.985)
    save(fig, "fig5_schedule_power.png")

    import pandas as pd
    pd.DataFrame({"hour": t_day, "omega_deg_h": w_day,
                  "T_adaptive_min": T_adapt / 60.0}
                 ).to_csv(f"{DATA_DIR}/schedule_trace.csv", index=False,
                          encoding="utf-8-sig")

    base = rows[1]   # 固定 2 min 作为"满足精度"的基准
    ada = rows[2]
    update_metrics("5_schedule_power", {
        "仿真条件": "长春夏至日，双轴 100 W(1 m²) 组件，误差约束 0.5°",
        "日照时长(h)": round(daylight_h, 2),
        "固定5min_动作次数": round(rows[0]["n"]),
        "固定5min_日耗电(Wh)": round(rows[0]["e"] / 3600, 2),
        "固定5min_最大偏差(deg)": round(rows[0]["err_max"], 2),
        "固定2min_动作次数": round(rows[1]["n"]),
        "固定2min_日耗电(Wh)": round(rows[1]["e"] / 3600, 2),
        "固定2min_最大偏差(deg)": round(rows[1]["err_max"], 2),
        "自适应_动作次数": round(rows[2]["n"]),
        "自适应_日耗电(Wh)": round(rows[2]["e"] / 3600, 2),
        "自适应_最大偏差(deg)": round(rows[2]["err_max"], 2),
        "自适应_平均功率(mW)": round(rows[2]["p_avg"] * 1000, 1),
        "较固定2min_动作次数下降(%)": round((1 - ada["n"] / base["n"]) * 100, 1),
        "较固定2min_耗电下降(%)": round((1 - ada["e"] / base["e"]) * 100, 1),
        "当日发电量(kWh)": round(gen_wh / 1000, 2),
        "追踪耗电占发电量(%)": round(ada["e"] / 3600 / gen_wh * 100, 2),
    })

    for r in rows:
        print(f"  {r['name']:8s} 次数 {r['n']:6.0f}  耗电 {r['e']/3600:.2f} Wh  "
              f"最大偏差 {r['err_max']:.2f}°")
    print(f"  当日发电 {gen_wh/1000:.2f} kWh；自适应较固定2min 动作次数降 "
          f"{(1 - ada['n'] / base['n']) * 100:.1f}%  耗电降 "
          f"{(1 - ada['e'] / base['e']) * 100:.1f}%")


if __name__ == "__main__":
    main()
