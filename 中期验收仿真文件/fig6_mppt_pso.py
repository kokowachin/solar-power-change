# -*- coding: utf-8 -*-
"""
成果 6：变步长 MPPT 与 PSO 在局部阴影下的寻优性能
对应申请书：研究内容第 5 项「最大功率点追踪算法的设计与融合」

PV 模型：3 块组件串联（含旁路二极管），单块 I-V 采用显式表达式
    Isc(G) = Isc0·G/1000
    Voc(G) = Voc0 + a·ln(G/1000)
    V(I)   = Voc + a·ln(1 − I/Isc) − I·Rs
局部阴影下各块 Isc 不同，串联后 P-V 曲线出现多峰。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from common import C, setup_style, save, update_metrics, DATA_DIR

ISC0, VOC0, A_COEF, RS = 6.0, 22.0, 1.6, 0.15
G_UNIFORM = (1000, 1000, 1000)
G_SHADED = (1000, 600, 300)


def string_v(I, Gs):
    vt = 0.0
    for G in Gs:
        isc = ISC0 * G / 1000.0
        voc = VOC0 + A_COEF * np.log(max(G, 1) / 1000.0)
        if I >= 0.999 * isc:
            vt += -0.7                     # 旁路二极管导通
        else:
            vt += voc + A_COEF * np.log(1.0 - I / isc) - I * RS
    return vt


def pv_curve(Gs, n=1600):
    isc_max = max(ISC0 * G / 1000.0 for G in Gs)
    I = np.linspace(1e-4, isc_max * 0.9995, n)
    V = np.array([string_v(i, Gs) for i in I])
    P = V * I
    order = np.argsort(V)
    return V[order], P[order], I[order]


def peak_info(V, P):
    """返回全局峰与局部峰（P 的极大值点）。"""
    idx = np.where((P[1:-1] > P[:-2]) & (P[1:-1] > P[2:]))[0] + 1
    peaks = [(float(V[i]), float(P[i])) for i in idx]
    gmax = int(np.argmax(P))
    return (float(V[gmax]), float(P[gmax])), peaks


def make_pv_func(V, P):
    return lambda v: float(np.interp(v, V, P))


def po_fixed(V, P, pfunc, step=1.0, v0=5.0, iters=250):
    f = pfunc
    v = v0
    p = f(v)
    hist_v, hist_p = [v], [p]
    for _ in range(iters):
        v2 = min(max(v + step, 0.5), V.max() - 0.5)
        p2 = f(v2)
        if p2 < p:
            step = -step
            v2 = min(max(v + step, 0.5), V.max() - 0.5)
            p2 = f(v2)
        v, p = v2, p2
        hist_v.append(v)
        hist_p.append(p)
    return np.array(hist_v), np.array(hist_p)


def po_variable(V, P, pfunc, k=0.25, v0=5.0, iters=250):
    f = pfunc
    v = v0
    p = f(v)
    hist_v, hist_p = [v], [p]
    dv = 0.8
    cap = 4.0
    for _ in range(iters):
        p_hi = f(min(v + dv, V.max() - 0.5))
        p_lo = f(max(v - dv, 0.5))
        grad = (p_hi - p_lo) / (2 * dv)
        step = float(np.clip(k * abs(grad), 0.08, cap)) * np.sign(grad)
        if step == 0.0:
            step = 0.08
        v2 = min(max(v + step, 0.5), V.max() - 0.5)
        p2 = f(v2)
        if p2 < p:
            cap = max(cap * 0.5, 0.05)
            v2, p2 = v, p
        else:
            cap = min(cap * 1.15, 4.0)
        v, p = v2, p2
        hist_v.append(v)
        hist_p.append(p)
    return np.array(hist_v), np.array(hist_p)


def pso(V, P, pfunc, n_part=12, iters=20, seed=3):
    rng = np.random.default_rng(seed)
    lo, hi = 0.5, float(V.max() - 0.5)
    f = pfunc
    x = rng.uniform(lo, hi, n_part)
    v = rng.uniform(-3, 3, n_part)
    pb = x.copy()
    pbp = np.array([f(xi) for xi in x])
    gb = float(pb[np.argmax(pbp)])
    gbp = float(pbp.max())
    hist = [gbp]
    w, c1, c2 = 0.72, 1.5, 1.5
    for _ in range(iters):
        r1, r2 = rng.random(n_part), rng.random(n_part)
        v = w * v + c1 * r1 * (pb - x) + c2 * r2 * (gb - x)
        x = np.clip(x + v, lo, hi)
        pv = np.array([f(xi) for xi in x])
        better = pv > pbp
        pb[better] = x[better]
        pbp[better] = pv[better]
        if pbp.max() > gbp:
            gbp = float(pbp.max())
            gb = float(pb[np.argmax(pbp)])
        hist.append(gbp)
    return np.array([gb]), np.array(hist)


def main():
    setup_style()

    V1, P1, _ = pv_curve(G_UNIFORM)
    V2, P2, _ = pv_curve(G_SHADED)
    f1, f2 = make_pv_func(V1, P1), make_pv_func(V2, P2)
    g1, peaks1 = peak_info(V1, P1)
    g2, peaks2 = peak_info(V2, P2)

    res = {}
    for tag, V, P, f, gmax in [("uniform", V1, P1, f1, g1),
                               ("shaded", V2, P2, f2, g2)]:
        vf, pf = po_fixed(V, P, f)
        vv, pvv = po_variable(V, P, f)
        vp, php = pso(V, P, f)
        res[tag] = dict(fixed=(vf, pf), var=(vv, pvv), pso=(vp, php),
                        pmax=gmax[1])

    fig = plt.figure(figsize=(13.6, 8.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.26)

    # (a) P-V 曲线
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(V1, P1, color="#27ae60", lw=2.2, label="均匀辐照 1000/1000/1000")
    ax.plot(V2, P2, color="#c0392b", lw=2.2, label="局部阴影 1000/600/300")
    ax.scatter([g1[0]], [g1[1]], color="#27ae60", s=60, zorder=5, marker="*")
    ax.scatter([g2[0]], [g2[1]], color="#c0392b", s=90, zorder=5, marker="*")
    for v, p in peaks2:
        ax.scatter([v], [p], facecolor="none", edgecolor="#c0392b", s=70, zorder=4)
    ax.annotate(f"全局峰 {g2[1]:.0f} W", (g2[0], g2[1]),
                textcoords="offset points", xytext=(10, 8), fontsize=8.5,
                color="#c0392b")
    loc = min(peaks2, key=lambda t: t[1])
    ax.annotate(f"局部峰 {loc[1]:.0f} W", (loc[0], loc[1]),
                textcoords="offset points", xytext=(-8, -18), fontsize=8.5,
                color="#c0392b")
    ax.set_xlim(0, 70)
    ax.set_ylim(0, max(P1.max(), P2.max()) * 1.18)
    ax.set_xlabel("阵列电压 (V)")
    ax.set_ylabel("阵列功率 (W)")
    ax.set_title("(a) 3 组件串联阵列 P-V 特性", loc="left")
    ax.legend(loc="upper right", fontsize=8.5)

    # (b) 阴影下寻优轨迹
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(V2, P2, color="#c0392b", lw=1.6, alpha=0.55, label="P-V 曲线")
    vf, pf = res["shaded"]["fixed"]
    vv, pvv = res["shaded"]["var"]
    vp, php = res["shaded"]["pso"]
    ax.plot(vf, pf, "o-", color=C["po"], ms=2.2, lw=1.0, alpha=0.9,
            label="定步长 P&O（陷入局部峰）")
    ax.plot(vv, pvv, "s-", color=C["iso"], ms=2.2, lw=1.0, alpha=0.9,
            label="变步长 P&O")
    ax.plot(vp[-1], php[-1], "^", color=C["pso"], ms=12, label="PSO 收敛点")
    ax.set_xlim(0, 70)
    ax.set_ylim(0, max(P2.max(), P1.max()) * 1.10)
    ax.set_xlabel("阵列电压 (V)")
    ax.set_ylabel("阵列功率 (W)")
    ax.set_title("(b) 局部阴影下的寻优轨迹", loc="left")
    ax.legend(loc="lower right", fontsize=8.2)

    # (c) 功率收敛过程
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(res["shaded"]["fixed"][1][:220], color=C["po"], lw=1.9,
            label="定步长 P&O")
    ax.plot(res["shaded"]["var"][1][:220], color=C["iso"], lw=1.9,
            label="变步长 P&O")
    ax.plot(np.arange(len(php)) * 12, php, color=C["pso"], lw=2.2,
            label="PSO（12 粒子 × 20 代）")
    ax.axhline(g2[1], color="k", ls="--", lw=1.0)
    ax.text(8, g2[1] * 0.90, f"全局最大 {g2[1]:.0f} W", fontsize=8.4)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, g2[1] * 1.18)
    ax.set_xlabel("累计功率采样次数")
    ax.set_ylabel("输出功率 (W)")
    ax.set_title("(c) 局部阴影下的功率收敛过程", loc="left")
    ax.legend(loc="lower right", fontsize=8.4)

    # (d) 效率对比
    ax = fig.add_subplot(gs[1, 1])
    names = ["定步长 P&O", "变步长 P&O", "PSO"]
    keys = ["fixed", "var", "pso"]
    cols = [C["po"], C["iso"], C["pso"]]
    x = np.arange(len(names))
    w = 0.36
    for i, (tag, lab) in enumerate([("uniform", "均匀辐照"), ("shaded", "局部阴影")]):
        vals = [res[tag][k][1][-1] / res[tag]["pmax"] * 100 for k in keys]
        bars = ax.bar(x + (i - 0.5) * w, vals, w,
                      color=cols if i == 1 else ["#d5d5d5"] * 3,
                      edgecolor="white", hatch="" if i == 1 else "//",
                      label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}",
                    ha="center", fontsize=8.2)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 118)
    ax.set_ylabel("MPPT 效率 (%)")
    ax.set_title("(d) 两种工况下的 MPPT 效率", loc="left")
    ax.axhline(100, color="k", lw=0.8, ls=":")
    ax.legend(loc="lower right", fontsize=8.4)

    fig.suptitle("图 6  变步长 MPPT 与 PSO 在局部阴影下的寻优性能（3 组件串联）",
                 fontsize=14.5, fontweight="bold", y=0.985)
    save(fig, "fig6_mppt_pso.png")

    eff = {tag: {k: res[tag][k][1][-1] / res[tag]["pmax"] * 100 for k in keys}
           for tag in ("uniform", "shaded")}

    update_metrics("6_mppt", {
        "阵列": "3 组件串联，单块 Isc=6 A / Voc=22 V / a=1.6 / Rs=0.15 Ω",
        "均匀辐照_理论最大功率(W)": round(g1[1], 1),
        "阴影辐照_理论最大功率(W)": round(g2[1], 1),
        "阴影下局部峰个数": len(peaks2),
        "均匀_定步长P&O效率(%)": round(eff["uniform"]["fixed"], 1),
        "均匀_变步长P&O效率(%)": round(eff["uniform"]["var"], 1),
        "均匀_PSO效率(%)": round(eff["uniform"]["pso"], 1),
        "阴影_定步长P&O效率(%)": round(eff["shaded"]["fixed"], 1),
        "阴影_变步长P&O效率(%)": round(eff["shaded"]["var"], 1),
        "阴影_PSO效率(%)": round(eff["shaded"]["pso"], 1),
    })

    print(f"  均匀 最大 {g1[1]:.1f} W  效率 "
          f"{eff['uniform']['fixed']:.1f}/{eff['uniform']['var']:.1f}/"
          f"{eff['uniform']['pso']:.1f} %")
    print(f"  阴影 最大 {g2[1]:.1f} W  局部峰 {len(peaks2)} 个  效率 "
          f"{eff['shaded']['fixed']:.1f}/{eff['shaded']['var']:.1f}/"
          f"{eff['shaded']['pso']:.1f} %")


if __name__ == "__main__":
    main()
