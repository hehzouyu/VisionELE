# 绝缘卫士 Insulator Guardian

> 基于 CA 注意力机制的绝缘子缺陷检测自主巡检机器人 —— **可跑通、可复现、口径写清楚**的工程落地骨架。

本目录是《绝缘卫士_技术路线.md》的实现落点。目标不是「跑一个数字」，而是把
**视觉检测 → 缺陷定位 → 巡检任务调度 → 平滑路径规划 → 局部避障** 这条链路
做成可以端到端跑起来、每一步都有日志和指标的东西；同时把构想里的进阶模块
（频域增强、图学习、元迁移、小样本生成）做成**可单测的代码骨架**，等真实数据到位
再逐个接入。

---

## 1. 目录结构

```
insulator_guardian/
├── configs/                 # 所有可调参数（yaml），代码里不写死超参
│   ├── vision.yaml          # 模型/训练/推理/数据/增强配置
│   ├── decision.yaml        # 地图/A*/DWA/规划器/调度器参数
│   └── pipeline.yaml        # 相机内参/仿真闭环/日志路径
├── src/
│   ├── utils/               # 配置加载、IO、日志
│   ├── vision/              # 视觉线
│   │   ├── attention.py     #   CoordAtt（CA 注意力）注入
│   │   ├── rfa.py           #   感受野聚合（RFA）模块骨架
│   │   ├── dataset.py       #   YOLO 格式数据集读取/划分
│   │   ├── detector.py      #   Ultralytics 推理封装（延迟导入）
│   │   ├── train.py         #   训练 / 评估 / 延迟基准
│   │   └── viz.py           #   检测框与掩码可视化
│   ├── decision/            # 决策线
│   │   ├── map.py           #   栅格地图、膨胀、障碍距离场
│   │   ├── astar.py         #   A*（8 邻域+转向惩罚）+ 路径平滑
│   │   ├── dwa.py           #   动态窗口法局部避障
│   │   ├── planner.py       #   全局+局部统一入口（重规划/恢复行为）
│   │   └── scheduler.py     #   巡检任务调度（缺陷复检机位规划、去重）
│   ├── integration/         # 线程接缝：数据结构、坐标、仿真场景、闭环
│   │   ├── schema.py        #   Detection / DetectionEvent / TaskPoint / VelocityCommand
│   │   ├── coords.py        #   像素 ↔ 地面坐标（相机针孔模型）
│   │   ├── sim_detector.py  #   闭环仿真用「真值检测器」（非识别精度）
│   │   ├── scene.py         #   12×8 m 走廊场景 + 7 个真值缺陷点
│   │   └── pipeline.py      #   感知→调度→规划→控制 单步闭环 + 台账
│   └── enhanced/            # 进阶语义增强（M4 骨架，numpy 可单测）
│       ├── freq.py          #   CWT / VMD 频域分支 + SE 通道注意力融合
│       ├── graph.py         #   多尺度图注意力（MSA-GCN）+ 半监督损失
│       ├── cdmth.py         #   原型网络元迁移检测头
│       └── cdcgan.py        #   条件 GAN 小样本生成（torch 延迟导入）
├── scripts/                 # 一键跑的命令行入口
├── tests/                   # pytest 单测/集成测试
└── results/                 # 产物（图/表/日志/台账），见根目录 README 说明
```

---

## 2. 关键口径（先看这里，避免误读）

1. **检测精度不来自仿真。** `src/integration/sim_detector.py` 只把「场景里预设的
   缺陷点」按相机几何投影成检测框，用来验证**定位、调度、规划、避障的链路正确性**。
   它不代表 CA-YOLO 的真实 mAP —— 真实精度要用真实数据集跑 `scripts/train_vision.py`。
2. **视觉线没装 torch 也能测。** `attention.py` / `rfa.py` / `cdcgan.py` 都是延迟导入，
   缺依赖时用到才报错并给出安装命令；`tests/` 里相关用例自动 skip。
3. **构想的「7 章」不逐条硬凹。** 频域增强、图学习、元迁移、小样本生成在
   `src/enhanced/` 里是**可运行、可单测的实现骨架**，参数留在 `vision.yaml::enhanced`，
   默认 `enabled: false`，M4 阶段逐项打开做消融——而不是把没数据的实验伪装成结论。
4. **规划用「膨胀图规划 + 原图急停」。** 全局 A* 在按 `robot_radius + safety_margin`
   膨胀的地图上搜（默认 0.3 + 0.1 = 0.4 m），保证路径留有安全间隙；DWA 急停距离按
   **原始地图**算，否则离墙 0.3 m 就会误判 `obstacle_too_close` 而卡死。这是实测踩过的坑，
   见 `Planner.dwa.raw_map`。
5. **到达前按距离限速（`dwa.approach_radius`）。** 1.0 m/s × 0.1 s 的单步位移比到达阈值
   （0.15 m）大一个量级，不加限速会在目标点附近反复超越、绕圈；现在距终点 < 0.8 m 时
   速度上限按距离线性收敛。
6. **调度去重按「缺陷点」而不是「机位」。** 同一个缺陷在不同帧被看到会生成不同机位，
   若按机位判重会重复派单；现在按缺陷点坐标 claim（`InspectionScheduler._claims`），
   同一处缺陷在 `dedupe_radius` 内只派一次。
7. **卡住有三级恢复。** DWA 报停时：车后有 0.4 m 退路就倒一小步，否则原地转向目标；
   连续 10 拍几乎没挪窝则切纯跟踪（沿 A* 已验证的路径小步走，默认走 40 拍）。
   `scripts/benchmark_decision.py` 的 50 图成功率从 52% 提到 94%，主要就是这一段的功劳。
8. **检测精度来自真实数据训练，不来自仿真。** 视觉线已用公开数据集 FBID（1600 张真实巡检照片，
   见 `../训练样本/FBID`）训练并评测：yolov8n 基线 test mAP@0.5 = 0.906~0.917。**当前以 broken
   为主验证指标**（flashover 与污秽混淆、标注不稳）：broken test mAP50 稳定在 0.90–0.95、
   mAP50-95 0.50–0.55；CA 对照在同 batch 下测不出可复现增益，不要把它当卖点写。
9. **CA 必须注入在 trainer 里。** Ultralytics 的 `Model.train()` 会重建模型并重载权重，
   在 `YOLO(...)` 实例上提前包装的层会被丢掉；`src/vision/train.py::_ca_trainer()`
   在 `get_model()` 里注入，权重里才能看到 `CoordAttWrapper`。

---

## 3. 快速开始

```bash
cd E:/创新创业/Demos/insulator_guardian

# 只需要决策线 + 闭环仿真：numpy / pyyaml / matplotlib / pillow
python scripts/demo_pipeline.py            # 跑完整巡检闭环，产物写到 results/

# 视觉线（torch + ultralytics）用本机已装好的 Demos/.venv：
#   .venv 在 Demos 下，不在本项目里；后面三条命令都把解释器换成它
..\.venv\Scripts\python.exe scripts/train_vision.py --help
..\.venv\Scripts\python.exe scripts/evaluate_vision.py --weights ../results/runs/real_ca_v8n/weights/best.pt
..\.venv\Scripts\python.exe scripts/benchmark_vision.py --weights ../results/runs/real_ca_v8n/weights/best.pt --device 0

# 真实数据（FBID）全链路：数据准备 → 训练 → 评测
#   数据已落在 ../训练样本/FBID/（见该目录 README）
python scripts/prepare_dataset.py --src ../训练样本/FBID/raw --dst ../训练样本/FBID/yolo
..\.venv\Scripts\python.exe scripts/train_vision.py --run-name real_baseline_v8n --epochs 60 --batch 32 --cache ram --no-ca
..\.venv\Scripts\python.exe scripts/train_vision.py --run-name real_ca_v8n       --epochs 60 --batch 32 --cache ram
..\.venv\Scripts\python.exe scripts/evaluate_vision.py --weights ../results/runs/real_ca_v8n/weights/best.pt --split test --device 0
python scripts/run_enhanced.py --max-crops 4000     # 增强模块实验（不依赖 torch）
..\.venv\Scripts\python.exe scripts/train_cdcgan.py --epochs 300 --batch 64

# flashover 补充实验：过采样 / 并入同源新图 / CA 换注入位置
python scripts/make_flash_oversample.py                                   # E2
python scripts/merge_flash_subset.py                                      # E3 数据合并
python scripts/make_flash_oversample.py --dataset MERGED --tag merged_e3  # E3 过采样清单
..\.venv\Scripts\python.exe scripts/train_vision.py --run-name real_ca_backbone_v8n --epochs 60 --batch 64 --cache ram --ca-inject backbone
python scripts/summarize_runs.py                    # 汇总所有 vision_eval_*.json
```

训练前建议先设两个缓存目录（沙箱里 AppData 不可写）：

```powershell
$env:YOLO_CONFIG_DIR='E:\创新创业\Demos\.tmp_ultra'
$env:MPLCONFIGDIR='E:\创新创业\Demos\.tmp_mpl'
```

依赖：

```bash
# 决策线（必需）
pip install numpy pyyaml
# 出图（demo 会用到）
pip install matplotlib pillow
# 视觉线（可选；本机已装在 ../.venv，torch 2.9.1+cu126 / ultralytics 8.4.192）
pip install torch torchvision ultralytics
```

---

## 4. 复现链路

1. **构建场景**：`src/integration/scene.py::build_inspection_field()` 生成 120×80
   @0.1 m 的走廊栅格，四角立柱 + 中段柱，7 个绝缘子真值点（含 3 个缺陷：2 闪络 + 1 破损）。
2. **闭环仿真**：`InspectionPipeline.run()` 每拍执行
   `仿真检测 → 像素转地面坐标 → 任务调度 → 规划器 step → 机器人积分`，
   并写 `results/logs/defect_ledger_<date>.jsonl` 台账（检测事件 / 任务创建 / 任务完成）。
3. **产物**：`results/plots/nav_trajectory_<date>.png`（轨迹与地面真值）、
   `results/plots/latency_<date>.png`（每拍耗时）、`results/gifs/nav_demo_<date>.gif`，
   以及 `results/benchmarks/pipeline_<date>.json` 与同名 `.md`。所有数字都由代码算出来，不手填。

当前基线（`python scripts/demo_pipeline.py`，seed=7）：

| 指标 | 数值 |
| --- | ---: |
| 闭环步数 | 789 / 3000（任务做完提前收工） |
| 完成任务 | 7（常规巡检 4 + 缺陷复检 3），待执行 0 |
| 碰撞 / 最小障碍间距 | 0 次 / 0.40 m |
| 轨迹长度 | 25.50 m |
| 单拍耗时 p95（检测 / 决策 / 循环） | 0.41 / 0.47 / 0.86 ms |

`python scripts/benchmark_decision.py`（50 张 80×80@0.1 m、障碍占比 0.12 的随机图，
A*+DWA 闭环）：**47/50 = 94%** 到达，失败 2 张碰撞、1 张超时；到达样本 188±64 步。
随机图是压力测试，不代表变电站现场成功率。

**真实数据训练（FBID，1600 张真实巡检照片，60 epoch × 2 seed）**：

| run | val mAP@0.5 | test mAP@0.5 | test mAP@0.5:0.95 |
| --- | ---: | ---: | ---: |
| yolov8n 基线（2 seed 均值） | 0.9137 | 0.9112 | 0.6086 |
| + CoordAtt@SPPF（2 seed 均值） | 0.9148 | 0.9100 | 0.6051 |

差值 ≤0.12pt、小于 seed 极差（0.48–0.49pt）→ 该配置下 CA 无稳定增益。
逐类短板是 flashover（0.80–0.84），normal 0.995、broken 0.90–0.95。
推理延迟（4090 / 640 / fp32）：mean 13.1 ms、p50 12.5 ms、p95 16.7 ms。

**flashover 补充实验（seed 0；同 batch 内比，跨 batch 不可比）**：

| run | batch | val mAP50 | val flashover | test mAP50 | test flashover |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 32 | 0.9112 | 0.800 | 0.9056 | 0.820 |
| E2 flashover×3 过采样 | 32 | 0.9193 | **0.852** | 0.9043 | 0.830 |
| baseline | 64 | **0.9234** | 0.830 | 0.9125 | 0.834 |
| E3 并入 296 张同源新图 + ×3 | 64 | 0.9149 | **0.853** | **0.9199** | **0.856** |
| E4 CoordAtt@backbone 9 层 | 64 | 0.9161 | 0.819 | 0.9027 | 0.804 |

- 补 flashover 数据：**单类两个划分都 +2.2pt**（E3-b64），代价是 val broken −4.8pt，
  整体持平 —— 采样权重挤占了别的类，公开同源数据的天花板到了这里。
- CA 换注入位置（backbone 9 层全挂，权重里已核实 9 个 `CoordAttWrapper`、
  参数 3,172,745）：比同 batch 基线低 val 0.7pt / test 1.0pt，注意力不是瓶颈。
- **batch 64 比 batch 32 高 1.2pt mAP50**，是本轮最大的非模型因素。
- 全部 run 的对照表见 `../results/benchmarks/training_summary.md`。

---

## 5. 测试

```bash
python -m pytest tests -q -p no:cacheprovider
```

当前：**82 passed, 5 skipped**（4 条 attention 用例与 1 条 GAN 构建用例因本机无 torch
自动跳过）。分三层：

| 层级 | 覆盖 | 是否依赖 torch |
| --- | --- | --- |
| 纯逻辑单测 | 数据结构往返、坐标变换、栅格/膨胀、A* 可达性与最优性、DWA 采样与碰撞、调度去重 | 否 |
| 集成测试 | 场景 → 闭环若干拍、台账写出、延迟统计结构 | 否 |
| 增强模块 | 频域分支形状与有限性、kNN 图对称性、GCN 前向/损失下降、原型网络、生成样本筛选 | 否（GAN 构建用例 skip） |

---

## 6. 已知限制

- 仿真检测器给出的是**几何投影框**，分辨率/遮挡/光照都不模拟；真实鲁棒性必须用实地数据验证。
- 增强模块（`src/enhanced/`）在**真实裁剪**上跑过模块级实验（`results/enhanced/`），
  但没有接进 YOLO 主干训练，所以不提供「提升多少 mAP」的结论；目前只有 cDCGAN 生成实验，
  且生成样本质量未收敛，不能当增广数据用。
- 训练数据是公开数据集 FBID（1600 张），与决赛现场工况、拍摄角度不完全一致；
  能补到的公开缺陷数据只有 `zxc060413/Insulator-defect_yolo_format`，且与 FBID
  **82.5% 同源**（去重后仅多 296 张）。补数据能把 flashover 推到 0.85–0.86，但会挤占
  broken，整体持平；真正的下一步是现场巡检图，不是继续堆公开集。
- CA 消融（2026-10-29 复测）：**以 broken 为主指标、同 batch 三档对照后，测不出可复现增益**。
  batch 32 上 backbone 9 层的 broken mAP50-95 比基线高 3.0pt（val），到 batch 64 / 96 反过来低
  2.4–2.5pt；同 batch 内三档 mAP50 差距 ≤1.5pt，而 seed 间 0.9–2.1pt、batch 间 0.7–1.2pt，
  差异落在噪声里。所以 CA 只能当「结构选项」，不要写成「提升了精度」。
  明细见 `Demos/WORKLOG.md` 的「broken 算法验证」与 `Demos/results/benchmarks/broken_summary.md`。
- 仿真机器人是差分驱动运动学，不含轮胎打滑、电机延迟。
- 决策线基准仍有 2/50 张图发生碰撞（都是随机图里 < 1 m 的窄缝、直角拐角），
  说明 DWA 在极窄通道下仍可能蹭到墙体；场景图的 1.8 m 走廊不受影响。
- 台账是 JSONL 追加写，没有做滚动清理，长时间跑要留意磁盘。

---

## 7. 与文档的对应

| 文档章节 | 落点 |
| --- | --- |
| 1.2 视觉检测（CA-YOLO） | `src/vision/attention.py`、`detector.py`、`train.py` |
| 1.3.1 频域增强 | `src/enhanced/freq.py` |
| 1.3.2 图学习 | `src/enhanced/graph.py` |
| 1.3.4 小样本生成 | `src/enhanced/cdcgan.py` |
| 1.3.5 元迁移 | `src/enhanced/cdmth.py` |
| 1.4 路径规划与避障 | `src/decision/{astar,dwa,planner}.py` |
| 1.5 巡检调度 | `src/decision/scheduler.py`、`src/integration/pipeline.py` |
| 仿真验证闭环 | `src/integration/{scene,sim_detector,pipeline}.py` |
