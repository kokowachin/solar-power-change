# -*- coding: utf-8 -*-
"""
成果 3：光强置信度评估与双模（天文/光电/保持）平滑切换仿真
对应申请书：创新点 1「双阈值联合判决机制」、拟解决的关键问题 1

判决逻辑：
    C_L = 实测平均光强 / 晴空期望光强          → 阴天时下降
    C_G = 四象限归一化梯度模长                 → 局部阴影时突增
    C_L < θ1        → 天文轨迹（盲追）
    C_G > θ2 且持续 → 先保持(延时重采)，超时后转入天文轨迹
    其余            → 光电精调
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from common import (C, setup_style, save, solar_position, update_metrics, DATA_DIR)

DT = 30.0                      # 仿真步长 (s)
T_START, T_END = 5 * 3600, 19 * 3600
DOY = 172                      # 夏至
TAU_TRACK = 20.0               # 跟踪环时间常数 (s)
ALPHA = 1.0 / TAU_TRACK

TH_LIGHT = 0.35                # 光照置信度阈值 θ1
TH_GRAD = 0.12                 # 梯度异常阈值 θ2（归一化）
KX = 0.06                      # 光强差分对角度误差的灵敏度 (1/deg)
HOLD_MAX_STEPS = 4             # 保持态最长 2 min，超时转天文盲追

MODES = {0: "天文轨迹", 1: "光电精调", 2: "保持/重采"}
MCOL = {0: C["astr"], 1: C["photo"], 2: C["hold"]}


def weather(sec):
    """天气剧本：返回 (透过率 T, 东侧传感器遮挡系数 shadow)。"""
    h = sec / 3600.0
    shadow = 1.0
    if h < 9.0:
        T = 1.00
    elif h < 10.0:
        T = 1.00 - 0.55 * (h - 9.0)          # 云层渐起
    elif h < 12.0:
        # 碎云 / 移动云影：透光率波动 + 单侧传感器被云影扫过
        T = 0.66 + 0.16 * np.sin(2 * np.pi * (h - 10.0) / 1.1)
        shadow = 0.55 if (int((h - 10.0) * 6) % 2 == 0) else 1.0
    elif h < 13.0:
        T = 0.25                             # 全阴天，光照均匀
    elif h < 14.0:
        T = 0.92                             # 持续局部遮挡（云层边缘/建筑）
        shadow = 0.45
    else:
        T = 1.00
    return T, shadow


def sensor_readings(e_az, e_el, T, shadow):
    """四象限传感阵列归一化输出。"""
    base = T
    diff_floor = (1.0 - T) * 0.12
    dx_true = np.clip(KX * e_az, -0.7, 0.7)
    dy_true = np.clip(KX * e_el, -0.7, 0.7)
    L_E = base * (1 + dx_true) * shadow + diff_floor
    L_W = base * (1 - dx_true) + diff_floor
    L_N = base * (1 + dy_true) + diff_floor
    L_S = base * (1 - dy_true) + diff_floor
    return L_E, L_W, L_N, L_S


def simulate(composite=True, seed=20260501):
    """composite=True 为本项目复合控制；False 为传统纯光电闭环。"""
    n = int((T_END - T_START) / DT) + 1
    sec = T_START + np.arange(n) * DT
    hours = sec / 3600.0

    el, az = solar_position(np.full(n, DOY), hours - 8.0)
    az_u = np.unwrap(np.deg2rad(az)) * 180 / np.pi
    rate_el = np.gradient(el, hours) / 3600.0
    rate_az = np.gradient(az_u, hours) / 3600.0

    rng = np.random.default_rng(seed)
    T_arr = np.array([weather(s)[0] for s in sec])
    sh_arr = np.array([weather(s)[1] for s in sec])

    # --- 量测噪声：AR(1)，相关时间 60 s，低光照时恶化 ---
    rho = np.exp(-DT / 60.0)
    sig_n = 0.02 * (1.0 + 2.0 * (1.0 - T_arr) / 0.75)
    n_az = np.zeros(n)
    n_el = np.zeros(n)
    for k in range(1, n):
        n_az[k] = rho * n_az[k - 1] + np.sqrt(1 - rho ** 2) * sig_n[k] * rng.normal()
        n_el[k] = rho * n_el[k - 1] + np.sqrt(1 - rho ** 2) * sig_n[k] * rng.normal()

    e_az = np.zeros(n)
    e_el = np.zeros(n)
    Ls = np.zeros((n, 4))
    mode = np.zeros(n, dtype=int)
    C_Lv = np.zeros(n)
    C_Gv = np.zeros(n)

    e_az[0], e_el[0] = 0.5, -0.4
    anomaly_cnt = 0
    bias_astro = 0.25          # 天文轨迹固有残余误差 (deg)

    for k in range(n):
        T, shadow = T_arr[k], sh_arr[k]
        L_E, L_W, L_N, L_S = sensor_readings(e_az[k], e_el[k], T, shadow)
        Ls[k] = (L_E, L_W, L_N, L_S)
        mean_L = (L_E + L_W + L_N + L_S) / 4.0
        dx = (L_E - L_W) / max(L_E + L_W, 1e-6)
        dy = (L_N - L_S) / max(L_N + L_S, 1e-6)

        C_Lv[k] = np.clip(mean_L, 0, 1.2)
        gmag = float(np.hypot(dx, dy))
        C_Gv[k] = gmag

        # ---------------- 状态机 ----------------
        if k == 0:
            mode[k] = 1
            anomaly_cnt = 0
        elif not composite:
            mode[k] = 1                              # 传统方案：始终光电闭环
        else:
            if C_Lv[k] < TH_LIGHT:
                mode[k] = 0                          # 天文盲追
                anomaly_cnt = 0
            elif gmag > TH_GRAD:
                anomaly_cnt += 1
                mode[k] = 2 if anomaly_cnt <= HOLD_MAX_STEPS else 0
            else:
                anomaly_cnt = max(0, anomaly_cnt - 1)
                mode[k] = 1

        # ---------------- 控制器 ----------------
        if mode[k] == 1:
            # 光电精调：角度误差由四象限差分反演，遮挡/云影会直接污染该估计
            u_az = ALPHA * np.clip(dx / KX + n_az[k], -12, 12)
            u_el = ALPHA * np.clip(dy / KX + n_el[k], -12, 12)
        elif mode[k] == 0:
            u_az = ALPHA * (e_az[k] - bias_astro)
            u_el = ALPHA * (e_el[k] - bias_astro)
        else:
            u_az = u_el = 0.0

        if k + 1 < n:
            e_az[k + 1] = np.clip(e_az[k] + (rate_az[k] - u_az) * DT, -30, 30)
            e_el[k + 1] = np.clip(e_el[k] + (rate_el[k] - u_el) * DT, -30, 30)

    err = np.hypot(e_az, e_el)
    return dict(hours=hours, sec=sec, e_az=e_az, e_el=e_el, err=err,
                mode=mode, C_L=C_Lv, C_G=C_Gv, T=T_arr, shadow=sh_arr, L=Ls)


def main():
    setup_style()
    comp = simulate(composite=True)
    pure = simulate(composite=False, seed=7)

    mask_bad = (comp["hours"] >= 9.0) & (comp["hours"] < 14.0)
    rms = lambda a: float(np.sqrt(np.mean(np.asarray(a) ** 2)))
    rms_comp, rms_pure = rms(comp["err"]), rms(pure["err"])
    rms_comp_bad, rms_pure_bad = rms(comp["err"][mask_bad]), rms(pure["err"][mask_bad])
    max_pure = float(pure["err"].max())
    max_comp = float(comp["err"].max())
    share = {m: float(np.mean(comp["mode"] == m) * 100) for m in MODES}

    fig = plt.figure(figsize=(13.6, 9.0))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1.15, 1],
                  hspace=0.44, wspace=0.26)

    # (a) 四路光强
    ax = fig.add_subplot(gs[0, 0])
    T = comp["T"]
    ax.plot(comp["hours"], T, color="k", lw=1.0, ls=":", label="环境透过率 T")
    styles = [("东向", 0, "#c0392b"), ("西向", 1, "#2980b9"),
              ("北向", 2, "#27ae60"), ("南向", 3, "#8e44ad")]
    for nm, i, col in styles:
        ax.plot(comp["hours"], comp["L"][:, i], color=col, lw=1.2, alpha=0.9,
                label=nm)
    ax.axvspan(10.0, 12.0, color="#f39c12", alpha=0.10, label="云影交替遮挡")
    ax.axvspan(13.0, 14.0, color="#8e44ad", alpha=0.16, label="持续局部遮挡")
    ax.set_xlim(5, 19)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("本地时刻 (h)")
    ax.set_ylabel("归一化光强")
    ax.set_title("(a) 四象限传感阵列响应与天气过程", loc="left")
    ax.legend(loc="lower left", fontsize=6.8, ncol=2, framealpha=0.92)

    # (b) 双阈值
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(comp["hours"], comp["C_L"], color="#27ae60", lw=1.8,
            label="光照置信度 $C_L$")
    ax.axhline(TH_LIGHT, color="#27ae60", ls="--", lw=1.0)
    ax.text(5.2, TH_LIGHT + 0.04, f"$θ_1$ = {TH_LIGHT}", fontsize=8.5, color="#27ae60")
    ax.set_ylim(0, 1.15)
    ax.set_xlim(5, 19)
    ax.set_xlabel("本地时刻 (h)")
    ax.set_ylabel("$C_L$", color="#27ae60")
    ax.tick_params(axis="y", colors="#27ae60")

    ax2 = ax.twinx()
    ax2.plot(comp["hours"], comp["C_G"], color="#8e44ad", lw=1.4,
             label="梯度异常指标 $C_G$")
    ax2.axhline(TH_GRAD, color="#8e44ad", ls="--", lw=1.0)
    ax2.text(5.2, TH_GRAD * 1.25, f"$θ_2$ = {TH_GRAD}", fontsize=8.5, color="#8e44ad")
    ax2.set_ylabel("$C_G$ (归一化梯度)", color="#8e44ad")
    ax2.tick_params(axis="y", colors="#8e44ad")
    ax2.grid(False)
    ax.set_title("(b) 双阈值联合判决量", loc="left")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    # (c) 误差对比（跨两列）+ 底部模式时间条
    ax = fig.add_subplot(gs[1, :])
    mode = comp["mode"]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([MCOL[0], MCOL[1], MCOL[2]])
    ax.imshow(mode[None, :], aspect="auto", cmap=cmap,
              extent=[5, 19, -1.15, -0.55], vmin=-0.5, vmax=2.5,
              interpolation="nearest", zorder=1)
    ax.annotate("模式", xy=(0, 0.024), xycoords="axes fraction",
                ha="right", va="center", fontsize=8.5, color="#333333",
                annotation_clip=False)
    ax.plot(comp["hours"], pure["err"], color=C["photo"], lw=1.6,
            label=f"传统纯光电闭环（RMSE {rms_pure:.2f}°）")
    ax.plot(comp["hours"], comp["err"], color=C["comp"], lw=2.2,
            label=f"本项目复合控制（RMSE {rms_comp:.2f}°）")
    ax.set_xlim(5, 19)
    ax.set_ylim(-1.3, max(max_pure * 1.10, 1.0))
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("本地时刻 (h)")
    ax.set_ylabel("指向误差 (°)")
    ax.set_title("(c) 全天跟踪误差对比（底部色条为复合控制工作模式）", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", fontsize=8.8)
    extra = [Patch(facecolor=MCOL[m], alpha=0.75, label=MODES[m]) for m in [0, 1, 2]]
    ax.legend(handles=extra, loc="upper right", fontsize=8.2)

    # (d) 指标对比
    ax = fig.add_subplot(gs[2, 0])
    cats = ["全天 RMSE", "恶劣天气段 RMSE", "最大误差"]
    pure_v = [rms_pure, rms_pure_bad, max_pure]
    comp_v = [rms_comp, rms_comp_bad, max_comp]
    x = np.arange(len(cats))
    w = 0.34
    b1 = ax.bar(x - w / 2, pure_v, w, color=C["photo"], label="传统纯光电")
    b2 = ax.bar(x + w / 2, comp_v, w, color=C["comp"], label="复合控制")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                    f"{b.get_height():.2f}", ha="center", fontsize=8.4)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("误差 (°)")
    ax.set_ylim(0, max(pure_v) * 1.22)
    ax.set_title("(d) 跟踪精度指标对比", loc="left")
    ax.legend(loc="upper left", fontsize=8.5)

    # (e) 模式占比
    ax = fig.add_subplot(gs[2, 1])
    names = [MODES[m] for m in [0, 1, 2]]
    vals = [share[m] for m in [0, 1, 2]]
    bars = ax.barh(names, vals, color=[MCOL[m] for m in [0, 1, 2]],
                   height=0.5, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("时间占比 (%)")
    ax.set_title("(e) 复合控制模式时间分布", loc="left")
    ax.invert_yaxis()

    fig.suptitle("图 3  置信度评估与双模平滑切换仿真（长春夏至日，30 s 步长）",
                 fontsize=14.5, fontweight="bold", y=0.985)
    save(fig, "fig3_dual_mode.png")

    pd.DataFrame({
        "hour": comp["hours"], "err_composite": comp["err"], "err_pure": pure["err"],
        "mode": comp["mode"], "C_L": comp["C_L"], "C_G": comp["C_G"],
    }).to_csv(f"{DATA_DIR}/dual_mode_trace.csv", index=False, encoding="utf-8-sig")

    update_metrics("3_dual_mode", {
        "仿真条件": "长春夏至日 5:00-19:00，步长 30 s，含阴天与局部遮挡事件",
        "复合控制全天RMSE(deg)": round(rms_comp, 3),
        "纯光电全天RMSE(deg)": round(rms_pure, 3),
        "复合控制恶劣天气RMSE(deg)": round(rms_comp_bad, 3),
        "纯光电恶劣天气RMSE(deg)": round(rms_pure_bad, 3),
        "复合控制最大误差(deg)": round(max_comp, 2),
        "纯光电最大误差(deg)": round(max_pure, 2),
        "全天RMSE降低(%)": round((1 - rms_comp / rms_pure) * 100, 1),
        "恶劣段RMSE降低(%)": round((1 - rms_comp_bad / rms_pure_bad) * 100, 1),
        "模式占比_天文(%)": round(share[0], 1),
        "模式占比_光电(%)": round(share[1], 1),
        "模式占比_保持(%)": round(share[2], 1),
        "判决阈值θ1": TH_LIGHT, "判决阈值θ2": TH_GRAD,
    })

    print(f"  复合 RMSE {rms_comp:.3f}° / 纯光电 {rms_pure:.3f}°  "
          f"降低 {(1 - rms_comp / rms_pure) * 100:.1f}%")
    print(f"  恶劣段 复合 {rms_comp_bad:.2f}° / 纯光电 {rms_pure_bad:.2f}°  "
          f"最大误差 {max_comp:.2f}° / {max_pure:.2f}°")
    print(f"  模式占比 天文 {share[0]:.1f}%  光电 {share[1]:.1f}%  保持 {share[2]:.1f}%")


if __name__ == "__main__":
    main()
