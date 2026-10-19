
# 仿真代码说明

本目录包含《基于双轴控制与光强反馈的高精度太阳能追光系统设计》中期检查的全部仿真源码。

## 快速开始

```bash
cd 仿真代码
python run\\\_all.py
```

约 4 秒跑完全部 6 项仿真，自动生成 `out/figures/` 下 6 张图和 `out/metrics.json` 指标汇总。

## 环境要求

```
Windows / Python 3.12
numpy · matplotlib · pandas · scipy
```

## 文件清单

|文件|作用|产出|
|-|-|-|
|`common.py`|公共模块：NOAA 太阳位置算法、晴空辐照模型、倾斜面 POA、绘图样式、指标汇总|—|
|`fig1\\\_sunpath.py`|成果 1：太阳位置算法验证与全年视日轨迹|`fig1\\\_sun\\\_path.png`|
|`fig2\\\_energy.py`|成果 2：固定 / 单轴 / 双轴年发电量对比|`fig2\\\_energy\\\_gain.png`|
|`fig3\\\_dual\\\_mode.py`|成果 3：置信度评估与双模（天文 / 光电 / 保持）切换|`fig3\\\_dual\\\_mode.png`|
|`fig4\\\_pid\\\_deadband.py`|成果 4：积分分离 PID 与动态死区控制性能对比|`fig4\\\_pid\\\_deadband.png`|
|`fig5\\\_schedule\\\_power.py`|成果 5：非均匀间歇调度与系统功耗|`fig5\\\_schedule\\\_power.png`|
|`fig6\\\_mppt\\\_pso.py`|成果 6：变步长 MPPT 与 PSO 局部阴影寻优|`fig6\\\_mppt\\\_pso.png`|
|`run\\\_all.py`|一键按顺序执行上述 6 个脚本|全部图表与指标|
|`md2docx.py`|工具：把 Markdown 报告转成 docx（标题 / 表格 / 图片）|—|

## 输出目录

```
out/
├─ figures/          6 张图（220 dpi PNG）
├─ data/             逐时功率、误差轨迹、调度序列等原始数据 CSV
└─ metrics.json      全部指标数字，报告中的每个数值都可在此溯源
```

`out/` 是脚本生成的产物，删除后重新运行即可完整重建。

## 各脚本可单独运行

```bash
python fig3\\\_dual\\\_mode.py      # 只跑其中一项
```

脚本之间只通过 `common.py` 共享参数，没有执行顺序依赖（`fig5` 会调用 `fig2` 中的辐照函数，但不落盘）。

## 模型口径提醒

* 全部结果为**数值仿真**，非户外实测；
* 辐照采用 Meinel 晴空模型，并按长春典型年逐月总辐照（1340 kWh/m²）标定月尺度晴空指数；
* 未计入云量逐时随机性、组件温升、风载扰动与安装失准，因此**绝对发电量偏乐观，对比类结论相对可靠**；
* 站点参数集中在 `common.py` 顶部（`LAT / LON / PANEL\\\_ETA / FIXED\\\_TILT` 等），改站点只需改这几个常量。

