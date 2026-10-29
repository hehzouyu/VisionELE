# CA-YOLO 验证集评测

- 权重：`E:\创新创业\Demos\insulator_guardian\..\results\runs\real_ca_backbone_b32_v8n\weights\best.pt`
- 划分：val
- 生成时间：2026-10-29T22:29:36+08:00
- mAP@0.5：**0.9132**
- mAP@0.5:0.95：**0.6112**

| 类别 | mAP@0.5 | mAP@0.5:0.95 | precision | recall |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.9947 | 0.9448 | 0.9894 | 1.0000 |
| flashover | 0.8040 | 0.3369 | 0.8643 | 0.6840 |
| broken | 0.9408 | 0.5519 | 0.8577 | 0.8974 |

> 口径说明：本表是离线验证集的**检测**指标（YOLO 标注真值），不代表整机巡检的缺陷识别精度；集成层 ``sim_detector.py`` 是几何仿真检测器，两者不可直接比较。
