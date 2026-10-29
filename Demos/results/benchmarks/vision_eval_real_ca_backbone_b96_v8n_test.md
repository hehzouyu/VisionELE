# CA-YOLO 验证集评测

- 权重：`E:\创新创业\Demos\insulator_guardian\..\results\runs\real_ca_backbone_b96_v8n\weights\best.pt`
- 划分：test
- 生成时间：2026-10-29T22:50:17+08:00
- mAP@0.5：**0.9063**
- mAP@0.5:0.95：**0.5956**

| 类别 | mAP@0.5 | mAP@0.5:0.95 | precision | recall |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.9950 | 0.9528 | 0.9961 | 0.9972 |
| flashover | 0.8214 | 0.3531 | 0.8711 | 0.7456 |
| broken | 0.9024 | 0.4808 | 0.8915 | 0.8750 |

> 口径说明：本表是离线验证集的**检测**指标（YOLO 标注真值），不代表整机巡检的缺陷识别精度；集成层 ``sim_detector.py`` 是几何仿真检测器，两者不可直接比较。
