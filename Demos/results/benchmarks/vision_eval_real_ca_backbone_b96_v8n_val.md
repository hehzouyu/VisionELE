# CA-YOLO 验证集评测

- 权重：`E:\创新创业\Demos\insulator_guardian\..\results\runs\real_ca_backbone_b96_v8n\weights\best.pt`
- 划分：val
- 生成时间：2026-10-29T22:49:55+08:00
- mAP@0.5：**0.9048**
- mAP@0.5:0.95：**0.5912**

| 类别 | mAP@0.5 | mAP@0.5:0.95 | precision | recall |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.9950 | 0.9401 | 0.9907 | 1.0000 |
| flashover | 0.8026 | 0.3346 | 0.8444 | 0.7425 |
| broken | 0.9167 | 0.4988 | 0.8368 | 0.8718 |

> 口径说明：本表是离线验证集的**检测**指标（YOLO 标注真值），不代表整机巡检的缺陷识别精度；集成层 ``sim_detector.py`` 是几何仿真检测器，两者不可直接比较。
