# VisionELE · 绝缘卫士

「基于 CA 注意力机制的绝缘子缺陷检测自主巡检机器人」的工作区。武汉理工大学 电气工程创新创业实践 / 大创项目。

## 目录

| 路径 | 内容 |
| --- | --- |
| `Demos/insulator_guardian/` | 主体代码：`src/{vision,decision,integration,enhanced,utils}`、`configs/`、`scripts/`、`tests/` |
| `Demos/results/` | 评测 JSON/Markdown、轨迹图、延迟图、增强模块报告（权重与训练图像不入库） |
| `Demos/WORKLOG.md` | 全部实验矩阵、踩坑记录、后续待办 |
| `report2Stu/` | 课程报告（LaTeX 源 + 7 页 PDF，含附录 A「创新点与论证缺口」）与创新点问题清单 |
| `.dsh/skills/` | 课程报告 / 文献 / 格式三个项目级 skill |
| `绝缘卫士_*.md` | 技术路线、实现方案、项目方案、技术实现构想 |

## 快速开始

```bash
cd Demos/insulator_guardian

# 决策闭环 demo（仿真检测器 + A* + DWA + 复检调度）
python scripts/demo_pipeline.py --max-steps 3000

# 视觉线：数据准备 → 训练 → 评测（需要 torch + ultralytics，见 requirements.txt）
python scripts/prepare_dataset.py --src ../训练样本/FBID/raw --dst ../训练样本/FBID/yolo
python scripts/train_vision.py --run-name baseline_v8n --no-ca --epochs 60 --batch 96 --cache ram
python scripts/evaluate_vision.py --weights ../results/runs/baseline_v8n/weights/best.pt --split test --device 0

# 测试
python -m pytest tests -p no:cacheprovider
```

## 结论口径（很重要）

- 真实数据（FBID，1600 张）上 YOLOv8n 的 test mAP@0.5 为 **0.906–0.917**，broken 单类 0.90–0.95。
- **CA 坐标注意力在本数据/本规模下测不出稳定增益**（差值 ≤0.12 pt，小于 seed 噪声 0.48 pt）。CA 只能写成「结构上的可选增强」，不能写成「精度提升点」。
- batch 与 seed 的交互大于 CA 的影响，**跨 batch 的数字不可互比**；本仓库所有对照均为同 batch 同 seed 数。
- 增强四模块（频域 / 图 / 小样本 / cDCGAN）只做了模块级实验，未接主干，其中频域融合为负增益（−1.80 pt）。

## 未入库内容

训练数据（FBID / FLASH / MERGED 数百 MB 到 GB）、`.venv`、模型权重、课程素材与申报材料（含参考文献 PDF 与个人签名文件）均通过 `.gitignore` 排除，需要时按下文说明在本地重建。

数据来源：[FBID](https://github.com/xxlree/FBID)（EPL-2.0）。
