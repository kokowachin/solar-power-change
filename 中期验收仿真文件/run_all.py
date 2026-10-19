# -*- coding: utf-8 -*-
"""一键复现全部仿真成果：python run_all.py"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODULES = [
    ("fig1_sunpath", "太阳位置算法验证"),
    ("fig2_energy", "年发电量对比"),
    ("fig3_dual_mode", "置信度与双模切换"),
    ("fig4_pid_deadband", "积分分离PID与动态死区"),
    ("fig5_schedule_power", "非均匀调度与功耗"),
    ("fig6_mppt_pso", "MPPT与PSO"),
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    if os.path.exists(os.path.join(here, "out", "metrics.json")):
        os.remove(os.path.join(here, "out", "metrics.json"))

    total = time.time()
    for mod, title in MODULES:
        print(f"\n=== {title} ({mod}.py) ===")
        t0 = time.time()
        __import__(mod).main()
        print(f"    耗时 {time.time() - t0:.1f}s")
    print(f"\n全部完成，总耗时 {time.time() - total:.1f}s")
    print("图表目录:", os.path.join(here, "out", "figures"))
    print("指标汇总:", os.path.join(here, "out", "metrics.json"))


if __name__ == "__main__":
    main()
