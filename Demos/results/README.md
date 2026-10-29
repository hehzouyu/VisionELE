# results/ 产物说明

本目录只放**代码跑出来的**东西，不放手工填写的数字。文件名带日期（`YYYYMMDD`），
同一天重复运行会覆盖同类文件，保证「一次命令 = 一份可追溯结果」。

```
results/
├── plots/                       # 图（PNG）
│   ├── nav_trajectory_<date>.png      # 栅格底图 + 膨胀障碍 + 机器人轨迹 + 任务/缺陷真值点
│   ├── latency_<date>.png             # 检测 / 决策 / 循环延迟分布（标 p95）
│   ├── benchmark_success_<date>.png   # 决策线随机图成功率环图 + 失败原因
│   └── benchmark_steps_<date>.png     # 决策线随机图到达步数分布
├── gifs/
│   └── nav_demo_<date>.gif            # 机器人逐帧位置（≤120 帧）
├── benchmarks/                  # 数值结果 + 一页摘要
│   ├── pipeline_<date>.json        # summary + 配置快照 + 机器信息（可复现实验记录）
│   ├── pipeline_<date>.md          # 中文结果摘要，直接可贴报告
│   ├── decision_<date>.json        # 50 张随机图的逐图记录
│   ├── decision_<date>.md          # 决策线基准摘要
│   ├── vision_eval_<run>_<split>.json/.md  # 真实数据训练后评测：mAP + 逐类 AP/PR
│   ├── training_summary.md            # 扫所有 vision_eval_*.json 生成的对照总表
│   ├── flash_oversample_e2.json       # E2 过采样清单统计（FBID）
│   ├── flash_oversample_merged_e3.json# E3 过采样清单统计（MERGED）
│   └── vision_latency_<date>.json/.md      # 视觉延迟基准（需 torch）
├── logs/
│   └── defect_ledger_<date>.jsonl  # 缺陷台账：检测事件 / 任务创建 / 任务完成
├── enhanced/                    # 增强模块真实样本实验
│   ├── enhanced_metrics.json       # freq / graph / cdmth 指标
│   ├── enhanced_report.md          # 一页中文摘要（含负结果与口径）
│   ├── real_crops.png              # 真实标注框裁剪抽样
│   ├── feature_pca.png             # 空间 vs 频域融合特征的 2D PCA
│   ├── cdcgan_<date>/              # cDCGAN 60 epoch：样本图 / 损失 / generator.pt
│   └── cdcgan_<date>_300e/         # 300 epoch 版本（同样欠拟合，仅作趋势对照）
└── runs/                        # 视觉训练产物（Ultralytics 的 runs/detect/...）
    ├── real_baseline_v8n/          # yolov8n 基线（无 CA，batch 32，seed 0）
    ├── real_ca_v8n/                # yolov8n + CoordAtt（只注入 SPPF 层，seed 0）
    ├── real_baseline_v8n_s1 / real_ca_v8n_s1   # seed=1 的对照
    ├── real_baseline_b64_v8n/      # batch 64 基线（E2 之后的统一对照）
    ├── real_flash3x_v8n/           # E2：flashover 图 ×3 过采样（batch 32）
    ├── real_merged_flash3x_v8n/    # E3：并入 296 张同源新图 + ×3 过采样（batch 32，50 ep 中断）
    ├── real_merged_b64_v8n/        # E3：同上，batch 64 完整 60 ep
    ├── real_ca_backbone_v8n/       # E4：CoordAtt 挂 backbone 9 层（batch 64）
    └── catest_smoke/               # 1 epoch 冒烟（验证 CA 注入是否真的进了权重）
```

## 真实数据结果（FBID，1600 张）

| run | val mAP@0.5 | val mAP@0.5:0.95 | test mAP@0.5 | test mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| baseline s0 | 0.9112 | 0.5999 | 0.9056 | 0.6086 |
| baseline s1 | 0.9161 | 0.6069 | 0.9167 | 0.6086 |
| CA s0 | 0.9172 | 0.6075 | 0.9031 | 0.6013 |
| CA s1 | 0.9124 | 0.6024 | 0.9168 | 0.6089 |

两 seed 均值：CA − 基线 = val **+0.12pt**、test **−0.12pt**；同配置 seed 间极差 0.48–0.49pt。
→ **现配置下测不出 CA 的增益**，不能写成「CA 提升 mAP」。

逐类短板：normal 0.995、broken 0.90–0.95、**flashover 只有 0.80–0.84**。

增强模块（3883 个真实裁剪，模块级实验）：freq 40.78% → 38.98%（负）、
graph 45.71%/46.35%、cdmth 5-shot 42.56%±5.28%、cdcgan 生成样本未收敛不可增广。

## flashover 补充实验（E2 / E3 / E4）

动机：flashover 是三类里最短板（val 0.80 / test 0.82）。公开数据集只有
`zxc060413/Insulator-defect_yolo_format`（InsuFault-Net）可用，且与 FBID
**82.5% 同源**，去重后只多 296 张（其中 156 张含 flashover）。

| split | run | batch | mAP50 | flashover | broken |
| --- | --- | ---: | ---: | ---: | ---: |
| val | baseline | 32 | 0.9112 | 0.800 | 0.938 |
| val | E2 flashover×3 | 32 | 0.9193 | **0.852** | 0.911 |
| val | baseline | 64 | **0.9234** | 0.830 | 0.945 |
| val | E3 MERGED+×3 | 64 | 0.9149 | **0.853** | 0.897 |
| val | E4 CA-backbone | 64 | 0.9161 | 0.819 | 0.934 |
| test | baseline | 32 | 0.9056 | 0.820 | 0.902 |
| test | E2 flashover×3 | 32 | 0.9043 | 0.830 | 0.888 |
| test | baseline | 64 | 0.9125 | 0.834 | 0.908 |
| test | E3 MERGED+×3 | 64 | **0.9199** | **0.856** | 0.909 |
| test | E4 CA-backbone | 64 | 0.9027 | 0.804 | 0.909 |

- 补 flashover 数据：该单类两个划分都 +2.2pt（E3-b64），test 整体 +0.74pt，
  但 val 的 broken 掉 4.8pt —— 采样权重挤占了别的类，整体持平。
- CA 换注入位置（backbone 9 层 C2f/SPPF 全挂，权重里已核实 9 个
  `CoordAttWrapper`、参数 3,172,745）：比同 batch 基线低 val 0.7pt / test 1.0pt。
- **batch 64 比 batch 32 高 1.2pt mAP50**，是本轮最大的非模型因素；
  跨 batch 的数字不能互相比较。

完整多 run 表见 `benchmarks/training_summary.md`（脚本 `summarize_runs.py` 生成）。

## broken 算法验证（2026-10-29 夜：batch 档位 × CA 注入位置）

动机：flashover 与污秽在图像上本就混淆、标注不稳，改用 **broken**（结构破损、标注可信）
作为「算法到底有没有用」的验证目标。数据仍是 FBID 1152 训练 / 128 val / 320 test。

| batch | run | val broken | test broken |
| ---: | --- | ---: | ---: |
| 32 | baseline | 0.9380/0.5223 | 0.9015/0.5018 |
| 32 | CA@SPPF | 0.9525/0.5314 | 0.9013/0.4928 |
| 32 | CA@backbone | **0.9408/0.5519** | 0.9050/0.5108 |
| 64 | baseline | 0.9446/0.5285 | 0.9081/0.5048 |
| 64 | CA@SPPF | 0.9440/0.5044 | 0.9048/0.5014 |
| 64 | CA@backbone | 0.9343/0.5038 | **0.9087/0.5104** |
| 96 | baseline | **0.9514/0.5233** | **0.9095/0.5021** |
| 96 | CA@backbone | 0.9167/0.4988 | 0.9024/0.4808 |

每格为 broken 单类 `mAP50/mAP50-95`，全部 yolov8n、60 epoch、seed 0、`--cache ram`。
单类汇总：`benchmarks/broken_summary.md`（`summarize_runs.py --focus broken`）。

- **CA 注入位置没有可复现增益**：b32 上 backbone 的 mAP50-95 高 3.0pt，b64/b96 反过来低
  2.4–2.5pt；同 batch 三档 mAP50 差 ≤1.5pt，而 seed 间 0.9–2.1pt、batch 间 0.7–1.2pt。
- **batch 不是越大越好**：broken mAP50 随 batch 上升，test 几乎不动，整体 mAP50-95 以 b64 最好；
  batch 96 的收益在吞吐（17–19 s/epoch、显存 11.9 GB、GPU 65%）。
- **broken 水平**：mAP50 0.90–0.95、mAP50-95 0.50–0.55；test 比 val 低 0.5–1.5pt（192 个框，抖动大）。

## 怎么复现

```bash
cd E:/创新创业/Demos/insulator_guardian
python scripts/demo_pipeline.py            # 决策线闭环 + 图 + 摘要
python scripts/benchmark_decision.py       # 50 张随机图 A*+DWA 基准
python scripts/make_figures.py             # 读最新 decision_*.json 出基准图

# 真实数据（FBID）训练线：用 ..\.venv\Scripts\python.exe
..\.venv\Scripts\python.exe scripts/train_vision.py --run-name real_baseline_v8n --epochs 60 --batch 32 --cache ram --no-ca
..\.venv\Scripts\python.exe scripts/train_vision.py --run-name real_ca_v8n       --epochs 60 --batch 32 --cache ram
..\.venv\Scripts\python.exe scripts/evaluate_vision.py --weights ../results/runs/real_ca_v8n/weights/best.pt --split test
..\.venv\Scripts\python.exe scripts/benchmark_vision.py --weights ../results/runs/real_ca_v8n/weights/best.pt

python scripts/run_enhanced.py --max-crops 4000   # 增强模块实验（numpy，conda python 即可）
```

flashover 补充实验（E2/E3/E4）的复现命令：

```bash
cd E:/创新创业/Demos/insulator_guardian
python scripts/make_flash_oversample.py                 # E2：FBID 里含 flashover 的图 ×3
python scripts/merge_flash_subset.py                    # E3：把同源新图并入 训练样本/MERGED
python scripts/make_flash_oversample.py --dataset MERGED --tag merged_e3

# 训练（venv 里的 python，统一 batch 64 便于对照）
..\.venv\Scripts\python.exe scripts/train_vision.py --run-name real_baseline_b64_v8n --epochs 60 --batch 64 --cache ram --no-ca
..\.venv\Scripts\python.exe scripts/train_vision.py --run-name real_merged_b64_v8n  --epochs 60 --batch 64 --cache ram --no-ca --data-yaml ../训练样本/MERGED/data_flash3x.yaml
..\.venv\Scripts\python.exe scripts/train_vision.py --run-name real_ca_backbone_v8n --epochs 60 --batch 64 --cache ram --ca-inject backbone

# 评测 + 汇总
..\.venv\Scripts\python.exe scripts/evaluate_vision.py --weights ../results/runs/real_merged_b64_v8n/weights/best.pt --split test --device 0
python scripts/summarize_runs.py                        # 生成 benchmarks/training_summary.md
```

## 口径

- 闭环 demo 用 `SimulatedDetector`：检测框由**场景预设的地面真值点**按相机几何反推，
  用来验证「定位 → 调度 → 规划 → 避障」链路，**不代表真实检测精度**。
- 延迟统计包含 Python 侧调用开销，数值只用于验证「决策 < 30 ms / 检测 < 200 ms /
  整拍 < 500 ms」这类预算，不等价于部署端（C++/TensorRT）实测。
- `vision_eval_*` 是**离线检测指标**（YOLO 标注真值）；数据来自 FBID 公开数据集，
  与决赛现场工况、与 `sim_detector.py` 的几何仿真都不等价。
- 训练对照是**单 seed / 两 seed**，「CA 与基线的差」必须同时给出 seed 均值与极差，
  不要把单跑差异写成显著增益。增强模块的指标是**模块级手工特征实验**，
  不等于接进 YOLO 主干后的端到端增益。
- 补数据的实验里，E2/E3 的 296 张新图与 FBID 同源（文件名重叠 82.5%），
  属于「同分布扩样本」而不是「换域」，它的增益上限本来就不高；
  另外 **batch 32 与 batch 64 不可直接比**（本轮相差 1.2pt mAP50）。
