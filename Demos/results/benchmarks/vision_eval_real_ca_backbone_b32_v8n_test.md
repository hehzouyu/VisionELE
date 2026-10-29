# CA-YOLO 验证集评测

- 权重：`E:\创新创业\Demos\insulator_guardian\..\results\runs\real_ca_backbone_b32_v8n\weights\best.pt`
- 划分：test
- 生成时间：2026-10-29T22:30:01+08:00
- mAP@0.5：**0.9062**
- mAP@0.5:0.95：**0.6094**

| 类别 | mAP@0.5 | mAP@0.5:0.95 | precision | recall |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.9950 | 0.9573 | 0.9952 | 0.9972 |
| flashover | 0.8185 | 0.3601 | 0.8953 | 0.7318 |
| broken | 0.9050 | 0.5108 | 0.9087 | 0.8809 |

> 口径说明：本表是离线验证集的**检测**指标（YOLO 标注真值），不代表整机巡检的缺陷识别精度；集成层 ``sim_detector.py`` 是几何仿真检测器，两者不可直接比较。
