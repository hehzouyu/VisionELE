# CA-YOLO 验证集评测

- 权重：`E:\创新创业\Demos\insulator_guardian\..\results\runs\real_ca_backbone_v8n\weights\best.pt`
- 划分：val
- 生成时间：2026-10-29T20:36:40+08:00
- mAP@0.5：**0.9161**
- mAP@0.5:0.95：**0.5993**

| 类别 | mAP@0.5 | mAP@0.5:0.95 | precision | recall |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.9947 | 0.9354 | 0.9881 | 0.9933 |
| flashover | 0.8194 | 0.3588 | 0.8923 | 0.7035 |
| broken | 0.9343 | 0.5038 | 0.8968 | 0.8974 |

> 口径说明：本表是离线验证集的**检测**指标（YOLO 标注真值），不代表整机巡检的缺陷识别精度；集成层 ``sim_detector.py`` 是几何仿真检测器，两者不可直接比较。
