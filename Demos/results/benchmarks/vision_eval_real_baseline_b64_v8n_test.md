# CA-YOLO 验证集评测

- 权重：`E:\创新创业\Demos\insulator_guardian\..\results\runs\real_baseline_b64_v8n\weights\best.pt`
- 划分：test
- 生成时间：2026-10-29T20:54:18+08:00
- mAP@0.5：**0.9125**
- mAP@0.5:0.95：**0.6082**

| 类别 | mAP@0.5 | mAP@0.5:0.95 | precision | recall |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.9950 | 0.9558 | 0.9962 | 0.9972 |
| flashover | 0.8344 | 0.3642 | 0.9136 | 0.7297 |
| broken | 0.9081 | 0.5048 | 0.9327 | 0.8646 |

> 口径说明：本表是离线验证集的**检测**指标（YOLO 标注真值），不代表整机巡检的缺陷识别精度；集成层 ``sim_detector.py`` 是几何仿真检测器，两者不可直接比较。
