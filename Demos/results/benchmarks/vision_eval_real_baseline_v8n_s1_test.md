# CA-YOLO 验证集评测

- 权重：`E:\创新创业\Demos\insulator_guardian\..\results\runs\real_baseline_v8n_s1\weights\best.pt`
- 划分：test
- 生成时间：2026-10-29T18:22:17+08:00
- mAP@0.5：**0.9167**
- mAP@0.5:0.95：**0.6086**

| 类别 | mAP@0.5 | mAP@0.5:0.95 | precision | recall |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.9950 | 0.9541 | 0.9950 | 0.9972 |
| flashover | 0.8400 | 0.3620 | 0.8902 | 0.7519 |
| broken | 0.9151 | 0.5097 | 0.8989 | 0.8802 |

> 口径说明：本表是离线验证集的**检测**指标（YOLO 标注真值），不代表整机巡检的缺陷识别精度；集成层 ``sim_detector.py`` 是几何仿真检测器，两者不可直接比较。
