# CA-YOLO 验证集评测

- 权重：`E:\创新创业\Demos\insulator_guardian\..\results\runs\real_baseline_b96_v8n\weights\best.pt`
- 划分：val
- 生成时间：2026-10-29T23:09:21+08:00
- mAP@0.5：**0.9157**
- mAP@0.5:0.95：**0.5988**

| 类别 | mAP@0.5 | mAP@0.5:0.95 | precision | recall |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.9950 | 0.9426 | 0.9918 | 1.0000 |
| flashover | 0.8005 | 0.3306 | 0.8467 | 0.7298 |
| broken | 0.9514 | 0.5233 | 0.8715 | 0.9231 |

> 口径说明：本表是离线验证集的**检测**指标（YOLO 标注真值），不代表整机巡检的缺陷识别精度；集成层 ``sim_detector.py`` 是几何仿真检测器，两者不可直接比较。
