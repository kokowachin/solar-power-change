# -*- coding: utf-8 -*-
"""
成果 4：积分分离 PID 与软件动态死区的控制性能对比
对应申请书：创新点 3、项目难点第 1 条「软件动态死区的阈值整定」

被控对象：双轴云台单轴
    J·ω' = K·u − b·ω − Tf·sign(ω),   θ' = ω
    叠加库仑摩擦（静摩擦锁死）与执行器死区 —— 这是产生极限环震荡的根源
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from common import C, setup_style, save, update_metrics, DATA_DIR

DT = 0.01
T_SIM = 8.0
STEP_DEG = 5.0
J, B, KM, TF = 0.5, 2.0, 1.0, 0.08
U_MAX = 2.0

KP, KI, KD = 16.0, 10.0, 1.0
INT_SEP_DEG = 3.0          # 积分分离阈值


def plant_step(theta, omega, u, dt):
    """一步被控对象积分（含静摩擦与执行器死区）。"""
    if abs(u) < 5e-3:
        u = 0.0
    tau = KM * u
    if abs(omega) < 1e-6:
        if abs(tau) <= TF:
            return theta, 0.0, tau
        tau_eff = tau - TF * np.sign(tau)
    else:
        tau_eff = tau - TF * np.sign(omega) - B * omega
    alpha = tau_eff / J
    omega_new = omega + alpha * dt
    if omega * omega_new < 0 and abs(tau) < TF:
        omega_new = 0.0
    return theta + omega_new * dt, omega_new, tau


def run_controller(kind, deadband_deg=0.0, t_end=T_SIM, dt=DT):
    """kind: 'pid' | 'iso' | 'iso_dead'"""
    n = int(t_end / dt) + 1
    t = np.arange(n) * dt
    theta = 0.0
    omega = 0.0
    integ = 0.0
    e_prev = STEP_DEG - np.rad2deg(theta)
    out = np.zeros(n)
    err = np.zeros(n)
    u_hist = np.zeros(n)
    db = np.deg2rad(deadband_deg)

    for k in range(n):
        e_deg = STEP_DEG - np.rad2deg(theta)
        err[k] = e_deg
        out[k] = np.rad2deg(theta)
        e = np.deg2rad(e_deg)

        de = (e - np.deg2rad(e_prev)) / dt
        e_prev = e_deg

        use_int = True
        if kind in ("iso", "iso_dead") and abs(e_deg) > INT_SEP_DEG:
            use_int = False          # 积分分离：误差大时冻结积分（不清零，避免退出时突跳）
        if use_int:
            integ += e * dt

        u = KP * e + KI * integ + KD * de

        # 软件死区：误差进入死区后冻结输出与积分器
        if kind == "iso_dead" and abs(e) < db:
            u = 0.0
            integ = 0.0
        else:
            if kind == "iso_dead" and db > 0:
                # 死区内不清零积分，仅冻结（避免退出死区时突跳）
                pass

        u = float(np.clip(u, -U_MAX, U_MAX))
        u_hist[k] = u
        theta, omega, _ = plant_step(theta, omega, u, dt)

    return dict(t=t, theta=out, err=err, u=u_hist)


def metrics(res, target=STEP_DEG):
    y = res["theta"]
    overshoot = max(0.0, (y.max() - target) / target * 100)
    # 2% 调节时间
    band = 0.02 * target
    outside = np.where(np.abs(y - target) > band)[0]
    settle = res["t"][outside[-1]] if len(outside) else 0.0
    tail = res["t"] >= res["t"][-1] - 3.0
    ripple = float(y[tail].max() - y[tail].min())
    ss_err = float(abs(np.mean(y[tail]) - target))
    # 电机启停次数（角速度跨越阈值的连续运动段数）
    w = np.gradient(y, res["t"])
    m = (np.abs(w) > 1e-3).astype(int)
    starts = int(np.sum(np.diff(np.r_[0, m]) == 1))
    # 稳态换向次数（末尾 3 s 内角速度过零次数，直接对应机械磨损）
    wt = w[tail]
    sg = np.sign(np.where(np.abs(wt) > 2e-5, wt, 0))
    reversals = int(np.sum(np.diff(sg[sg != 0]) != 0)) if np.any(sg != 0) else 0
    energy = float(np.sum(np.abs(res["u"])) * DT)
    motion = float(np.sum(np.abs(np.diff(y[tail]))))
    return dict(overshoot=overshoot, settle=settle, ripple=ripple,
                ss_err=ss_err, starts=starts, reversals=reversals,
                energy=energy, motion=motion)


def main():
    setup_style()

    r_pid = run_controller("pid")
    r_iso = run_controller("iso")
    r_ded = run_controller("iso_dead", deadband_deg=0.08)

    m_pid, m_iso, m_ded = (metrics(r_pid), metrics(r_iso), metrics(r_ded))

    # 死区扫描
    dbs = np.linspace(0.01, 0.60, 25)
    scan = [metrics(run_controller("iso_dead", deadband_deg=d)) for d in dbs]

    fig = plt.figure(figsize=(13.6, 8.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.26)

    # (a) 阶跃响应
    ax = fig.add_subplot(gs[0, 0])
    for res, nm, col in [(r_pid, "常规 PID", C["pid"]),
                         (r_iso, "积分分离 PID", C["iso"]),
                         (r_ded, "积分分离 + 动态死区", C["dead"])]:
        ax.plot(res["t"], res["theta"], color=col, lw=1.9, label=nm)
    ax.axhline(STEP_DEG, color="k", lw=0.9, ls="--")
    ax.axhline(STEP_DEG * 1.02, color="k", lw=0.6, ls=":")
    ax.axhline(STEP_DEG * 0.98, color="k", lw=0.6, ls=":")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 7)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("云台角度 (°)")
    ax.set_title("(a) 5° 阶跃响应对比", loc="left")
    ax.legend(loc="lower right", fontsize=8.6)

    # (b) 稳态误差放大
    ax = fig.add_subplot(gs[0, 1])
    for res, nm, col in [(r_pid, "常规 PID", C["pid"]),
                         (r_iso, "积分分离 PID", C["iso"]),
                         (r_ded, "积分分离 + 动态死区", C["dead"])]:
        m = res["t"] >= 4.0
        ax.plot(res["t"][m], np.abs(STEP_DEG - res["theta"][m]),
                color=col, lw=1.8, label=nm)
    ax.axhline(0.1, color="#8e44ad", ls="--", lw=1.0)
    ax.text(4.05, 0.103, "精度要求 0.1°", fontsize=8, color="#8e44ad")
    ax.axhline(0.05, color="#7f8c8d", ls=":", lw=1.0)
    ax.text(4.05, 0.037, "死区自适应下限 0.05°", fontsize=8, color="#7f8c8d")
    ax.set_xlim(4, 8)
    ax.set_ylim(0, 0.13)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("稳态偏差 |5°−θ| (°)")
    ax.set_title("(b) 稳态段放大：持续微动与死区抑制", loc="left")
    ax.legend(loc="center right", fontsize=8.2)

    # (c) 死区宽度扫描
    ax = fig.add_subplot(gs[1, 0])
    ripple = [s["ripple"] for s in scan]
    ss = [s["ss_err"] for s in scan]
    mo = [s["motion"] for s in scan]
    ax.plot(dbs, ss, "o-", color="#c0392b", lw=1.8, ms=4,
            label="稳态偏差 (°)")
    ax.plot(dbs, mo, "s-", color="#2980b9", lw=1.8, ms=4,
            label="稳态微动行程 (°/3s)")
    ax.axvspan(0.05, 0.30, color="#27ae60", alpha=0.12,
               label="动态死区自适应区间")
    ax.axhline(0.1, color="#8e44ad", ls="--", lw=1.0, label="精度要求 0.1°")
    ax.set_xlabel("死区宽度 (°)")
    ax.set_ylabel("角度量 (°)")
    ax.set_ylim(-0.005, 0.36)
    ax.set_xlim(0, 0.62)
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1, l1, loc="upper center", fontsize=8.0)
    ax.set_title("(c) 死区宽度对稳态偏差与微动行程的权衡", loc="left")

    # (d) 指标表
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ms = [m_pid, m_iso, m_ded]
    rows = ["超调量 σ (%)", "调节时间 t_s (s)", "稳态偏差 (°)",
            "稳态微动行程 (°/3s)", "累计控制量 (A·s)"]
    cells = [
        [f"{m['overshoot']:.1f}" for m in ms],
        [f"{m['settle']:.2f}" for m in ms],
        [f"{m['ss_err']:.3f}" for m in ms],
        [f"{m['motion']:.3f}" for m in ms],
        [f"{m['energy']:.2f}" for m in ms],
    ]
    tb = ax.table(cellText=cells, rowLabels=rows,
                  colLabels=["常规 PID", "积分分离 PID", "积分分离+动态死区"],
                  loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(9.2)
    tb.scale(1.0, 1.85)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(fontweight="bold")
        if c == 3 and r > 0:
            cell.set_facecolor("#eaf7ef")
    ax.set_title("(d) 综合性能指标对比", loc="left")

    fig.suptitle("图 4  积分分离 PID 与动态死区的控制性能对比（含库仑摩擦的云台模型）",
                 fontsize=14.5, fontweight="bold", y=0.985)
    save(fig, "fig4_pid_deadband.png")

    import pandas as pd
    pd.DataFrame({"t": r_pid["t"], "pid": r_pid["theta"],
                  "iso": r_iso["theta"], "iso_dead": r_ded["theta"]}
                 ).to_csv(f"{DATA_DIR}/pid_step_response.csv", index=False,
                          encoding="utf-8-sig")

    update_metrics("4_pid_deadband", {
        "被控对象": "J=0.5, b=2.0, K=1.0, 库仑摩擦 0.08 N·m, 限幅 ±2 A",
        "常规PID_超调(%)": round(m_pid["overshoot"], 2),
        "积分分离PID_超调(%)": round(m_iso["overshoot"], 2),
        "积分分离+死区_超调(%)": round(m_ded["overshoot"], 2),
        "常规PID_调节时间(s)": round(m_pid["settle"], 2),
        "积分分离PID_调节时间(s)": round(m_iso["settle"], 2),
        "积分分离+死区_调节时间(s)": round(m_ded["settle"], 2),
        "常规PID_稳态峰峰值(deg)": round(m_pid["ripple"], 4),
        "积分分离PID_稳态峰峰值(deg)": round(m_iso["ripple"], 4),
        "积分分离+死区_稳态峰峰值(deg)": round(m_ded["ripple"], 4),
        "常规PID_稳态偏差(deg)": round(m_pid["ss_err"], 3),
        "积分分离PID_稳态偏差(deg)": round(m_iso["ss_err"], 3),
        "积分分离+死区_稳态偏差(deg)": round(m_ded["ss_err"], 3),
        "常规PID_稳态微动行程(deg/3s)": round(m_pid["motion"], 3),
        "积分分离PID_稳态微动行程(deg/3s)": round(m_iso["motion"], 3),
        "积分分离+死区_稳态微动行程(deg/3s)": round(m_ded["motion"], 3),
        "稳态微动行程下降(%)": round((1 - m_ded["motion"] / m_pid["motion"]) * 100, 1),
        "累计控制量下降(%)": round((1 - m_ded["energy"] / m_pid["energy"]) * 100, 1),
        "动态死区区间(deg)": "0.05 ~ 0.30",
        "积分分离阈值(deg)": INT_SEP_DEG,
    })

    print(f"  超调  PID {m_pid['overshoot']:.1f}%  → 积分分离 {m_iso['overshoot']:.1f}%"
          f"  → +死区 {m_ded['overshoot']:.1f}%")
    print(f"  稳态峰峰值  {m_pid['ripple']:.4f}° / {m_iso['ripple']:.4f}° / "
          f"{m_ded['ripple']:.4f}°")
    print(f"  启停次数  {m_pid['starts']} → {m_iso['starts']} → {m_ded['starts']}")


if __name__ == "__main__":
    main()
