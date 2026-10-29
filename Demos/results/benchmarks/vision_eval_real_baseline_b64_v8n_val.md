# CA-YOLO 验证集评测

- 权重：`E:\创新创业\Demos\insulator_guardian\..\results\runs\real_baseline_b64_v8n\weights\best.pt`
- 划分：val
- 生成时间：2026-10-29T20:53:56+08:00
- mAP@0.5：**0.9234**
- mAP@0.5:0.95：**0.6063**

| 类别 | mAP@0.5 | mAP@0.5:0.95 | precision | recall |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.9950 | 0.9451 | 0.9930 | 1.0000 |
| flashover | 0.8305 | 0.3451 | 0.9071 | 0.6981 |
| broken | 0.9446 | 0.5285 | 0.9281 | 0.8846 |

> 口径说明：本表是离线验证集的**检测**指标（YOLO 标注真值），不代表整机巡检的缺陷识别精度；集成层 ``sim_detector.py`` 是几何仿真检测器，两者不可直接比较。
