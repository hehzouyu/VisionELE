# CA-YOLO 验证集评测

- 权重：`E:\创新创业\Demos\insulator_guardian\..\results\runs\real_baseline_b96_v8n\weights\best.pt`
- 划分：test
- 生成时间：2026-10-29T23:09:55+08:00
- mAP@0.5：**0.9086**
- mAP@0.5:0.95：**0.6064**

| 类别 | mAP@0.5 | mAP@0.5:0.95 | precision | recall |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.9950 | 0.9529 | 0.9972 | 0.9982 |
| flashover | 0.8213 | 0.3643 | 0.8937 | 0.7377 |
| broken | 0.9095 | 0.5021 | 0.8980 | 0.8711 |

> 口径说明：本表是离线验证集的**检测**指标（YOLO 标注真值），不代表整机巡检的缺陷识别精度；集成层 ``sim_detector.py`` 是几何仿真检测器，两者不可直接比较。
