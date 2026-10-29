# CA-YOLO 验证集评测

- 权重：`E:\创新创业\Demos\insulator_guardian\..\results\runs\real_merged_flash3x_v8n\weights\best.pt`
- 划分：val
- 生成时间：2026-10-29T20:18:01+08:00
- mAP@0.5：**0.9118**
- mAP@0.5:0.95：**0.5945**

| 类别 | mAP@0.5 | mAP@0.5:0.95 | precision | recall |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.9950 | 0.9359 | 0.9898 | 1.0000 |
| flashover | 0.8534 | 0.3676 | 0.8573 | 0.7932 |
| broken | 0.8871 | 0.4801 | 0.8723 | 0.8077 |

> 口径说明：本表是离线验证集的**检测**指标（YOLO 标注真值），不代表整机巡检的缺陷识别精度；集成层 ``sim_detector.py`` 是几何仿真检测器，两者不可直接比较。
