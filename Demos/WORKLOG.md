# 绝缘卫士 · 工作计划与进度

> 位置：`E:\创新创业\Demos\WORKLOG.md`。按时间倒序，最新在上；每条写清「做了什么 / 现在的状态 / 下一步怎么接」。

## 2026-10-29（真实样本训练线）

### 目标

用户要求：解决「训练样本 + 训练本身」的问题，使能进阶增强模块；先找真实样本，尽量用真实样本训练，样本统一放 `E:\创新创业\Demos\训练样本`。

### 训练结果（真实样本，两 seed 对照）

| run | 权重 | val mAP@0.5 | val mAP@0.5:0.95 | test mAP@0.5 | test mAP@0.5:0.95 |
| --- | --- | ---: | ---: | ---: | ---: |
| `real_baseline_v8n`（yolov8n，seed 0） | `results/runs/real_baseline_v8n/weights/best.pt` | 0.9112 | 0.5999 | 0.9056 | 0.6086 |
| `real_baseline_v8n_s1`（seed 1） | `results/runs/real_baseline_v8n_s1/weights/best.pt` | 0.9161 | 0.6069 | 0.9167 | 0.6086 |
| `real_ca_v8n`（+CoordAtt@SPPF，seed 0） | `results/runs/real_ca_v8n/weights/best.pt` | 0.9172 | 0.6075 | 0.9031 | 0.6013 |
| `real_ca_v8n_s1`（seed 1） | `results/runs/real_ca_v8n_s1/weights/best.pt` | 0.9124 | 0.6024 | 0.9168 | 0.6089 |

两 seed 均值：基线 val 0.9137 / test 0.9112；CA val 0.9148 / test 0.9100。
差值（CA − 基线）= val **+0.12pt**、test **−0.12pt**，而同配置内的 seed 间极差是 0.48–0.49pt。

逐类 mAP@0.5（val / test，CA s0 / s1）：normal 0.995/0.995、flashover 0.804/0.813、0.818/0.835、
broken 0.953/0.901、0.924/0.921；基线同口径在 0.995、0.800/0.820、0.938/0.902 附近。

> 口径：**在该数据量、该超参、该注入位置（SPPF 之后）下，CA 与基线没有可区分的差异**。
> 差值小于 seed 噪声，不能写成「CA 提升了 mAP」，也不能反过来说「CA 无效」——
> 只能说这组对照没测出增益。要做增益结论，需要：更多缺陷样本（尤其 flashover 只有 ~1700 框）、
> 换注入位置（Backbone 各 C2f 输出，而非只挂 P5 入口）、或换更大骨干。
> flashover 一直是短板（0.80–0.84），报告里值得单独说明。

评测产物：`results/benchmarks/vision_eval_{real_baseline_v8n,real_ca_v8n,real_baseline_v8n_s1,real_ca_v8n_s1}_{val,test}.{json,md}`。


### 已完成

1. **真实数据集落盘**：FBID（github.com/xxlree/FBID，EPL-2.0），1600 张真实巡检照片
   （train 1280 + test 320），落在 `训练样本/FBID/raw`，下载脚本与产物见 `.tmp_ds/download_fbid.py`，
   文件数校验 3200/3200 全部到位。
2. **类别重映射 + 划分**：`scripts/prepare_dataset.py` 把 FBID 的
   `0 pollution_flashover / 1 broken / 2 insulator` 映射到项目
   `0 normal / 1 flashover / 2 broken`，按「是否含缺陷」分层切出 val 128 张，
   产物在 `训练样本/FBID/yolo`（含 `data.yaml`、`dataset_stats.json`、`preview_gt.jpg`）。
   类别统计：train 1331/1729/823，val 149/212/78，test 352/507/192。
3. **数据说明**：`训练样本/FBID/README.md`（来源、许可证、映射表、统计、注意点）。
4. **训练环境已就绪**：`.venv` 里装好 torch 2.9.1+cu126 / torchvision 0.24.1+cu126 /
   ultralytics 8.4.192 / numpy 2.5.4 / opencv 5.0.0.93 / polars 1.30.0 等（CUDA 可用，
   识别到 RTX 4090 Laptop 16GB）。装包全部走 `.tmp_pip/wheel_install.py`（wheel 本质是
   zip，直接解到 site-packages），因为 pip 在受限令牌下连 `--target` 都会 `WinError 5`。
5. **基线 + CA 各 60 epoch × 2 seed 跑完**（batch 32、`--cache ram`，约 20–40 s/epoch；
   RAM 缓存把单 epoch 从 100 s 压到 40 s）。`evaluate()` 的逐类指标已修（按
   `box.ap_class_index` 对齐，之前 per_class 恒为空）；评测文件名带 run 名 + split。
6. **`train_vision.py` 新增 `--seed`**，`train()` 透传 seed。
7. **修掉一个会让 CA 对照失效的 bug**：Ultralytics 的 `Model.train()` 会按 yaml 重建
   `DetectionModel` 并重新载入预训练权重，原先在 `YOLO(...)` 实例上注入的 CoordAtt
   包装层被丢掉了——第一轮的「CA 模型」其实就是基线（参数量 3,138,489，第 9 层是 SPPF）。
   现在在 `DetectionTrainer.get_model()` 里注入（`src/vision/train.py` 的 `_ca_trainer()`），
   权重里能看到 `CoordAttWrapper`，参数量 3,145,161。
8. **`scripts/train_cdcgan.py` 跑通**（真实缺陷裁剪 64×64，hinge + 谱归一化）。
   60 epoch 用时 34 s、300 epoch 用时 552 s，产物 `results/enhanced/cdcgan_20261029{,_300e}/`。
   生成样本仍是低频色块，**不能当增广数据用**，只作为「管线可跑」的证据。
9. **`scripts/run_enhanced.py` 32×32 全量**（3883 个真实标注框裁剪）：
   freq 空间 40.78% → CWT/VMD 融合 38.98%（**−1.80 pt，负结果**）；
   graph（MSA-GCN，20% 标签）acc_all 45.71% / 无标签 46.35%，loss 1.141→0.872；
   cdmth 5-shot 42.56%±5.28%、10-shot 45.91%±2.68%。报告：`results/enhanced/enhanced_report.md`。

### 文档与基准已收尾

- CA 消融与增强模块数字已写进 `results/README.md` 和 `insulator_guardian/README.md`
  （口径第 8、9 条）。
- 视觉延迟已测（`results/benchmarks/vision_latency_20261029.{json,md}`）：
  mean 13.11 ms / p50 12.45 / p95 16.66 / max 23.41（640×640、batch 1、4090）。

### 还能往下做的

1. **CA 要出结论得换实验条件**：现配置下 CA 与基线差 ≤0.12pt、小于 seed 噪声。
   可试：把 CoordAtt 挂到 Backbone 每个 C2f 输出（不止 P5 入口）、增大 epoch/模型、
   或只针对 flashover 类做消融（该类 mAP 只有 0.80–0.84，是短板）。
2. **补 flashover 样本**：FBID 里 flashover 与污秽混标，若决赛现场有真实缺陷图，
   优先补进来；或对 flashover 做小目标/低对比度增广。
3. `scripts/train_cdcgan.py` 若要出「可用于增广」的样本，需要更多 epoch + 更强判别器
   （或换 diffusion），当前 4090 上 300 epoch 仍欠拟合。

### 增强模块结果（真实裁剪，模块级实验）

| 模块     | 指标                       |                          数值 | 结论                |
| ------ | ------------------------ | --------------------------: | ----------------- |
| freq   | 最近类中心 acc（空间 → +CWT/VMD） |             40.78% → 38.98% | 频域融合在本组数据上是负增益    |
| graph  | MSA-GCN acc（全节点 / 无标签）   |             45.71% / 46.35% | 三个模块里判别性最好，依赖同类成簇 |
| cdmth  | 5-shot / 10-shot         | 42.56%±5.28% / 45.91%±2.68% | 少样本基线可测，可直接做对照    |
| cdcgan | 生成样本                     |           1200/2400 通过中心距筛选 | 管线通、质量未收敛，不可直接增广  |

> 3 类随机基线 33%。这些数字来自 64×64 裁剪上的手工块统计特征与 numpy 实现，
> 用来说明「模块在真实数据上跑得通、有可测判别性」，不等于接进 YOLO 主干后的端到端增益。

### flashover 补充实验（E2/E3/E4）

短板是 flashover（CA s0 val/test mAP50 0.804/0.813）。先补这一类数据，再谈换注入位置。

**外部数据调研结论（2026-10-29）**

- 能拿到的公开集几乎都是同一批图的再分发：`zxc060413/Insulator-defect_yolo_format`
  （release v1.0.0，1688 张、4 类 `insulator/damaged/Flashover/hammer`）与 FBID 按文件名
  重叠 82.5%，按内容 md5 去重后**只有 296 张是真新增（156 张含 flashover）**。
- `nextcombo/*`、`QQ767172261/*`、`lonlonago/*` 全是「资源清单/加群赞助获取」，不是数据。
- `phd-benel/BGI` 只有 broken 类（544 张），补不了 flashover；Kaggle 无凭据；
  `sofia-loya/*`、`WhiteLiu1123/*` 只有 README。HF / Zenodo 本机网络不可达。
- GitHub release 资产直连与 `objects.githubusercontent.com` 都超时，必须走
  `gh-proxy.com` / `ghfast.top` 镜像（脚本 `.tmp_ds/dl_mirror.py`）。

**下面三组实验都只用 train 侧改动，val/test 保持 FBID 原样**，所以能和此前 4 个 run 直接比。

| 编号 | 训练数据 | 说明 |
| --- | --- | --- |
| E2 | FBID train + flashover 图 ×3（2290 条） | 只靠过采样，不引入新图 |
| E3 | MERGED train（1152 + 296 新图）+ 同上过采样 | 看同源新图是否还有边际收益 |
| E4 | 仅当 E2/E3 都无改进时 | CoordAtt 换注入位置（Backbone 各 C2f 输出）重跑消融 |

**结果（seed 0，60 epoch；E3-b32 只跑到 50 epoch 被中断，已标出）**

每格 `mAP50 / mAP50-95`；`b64` 表示 batch=64（后面为了提速补的对照）。

| 划分 | 模型 | mAP50 | mAP50-95 | normal | flashover | broken |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| val | baseline (b32) | 0.9112 | 0.5999 | 0.995/0.946 | 0.800/0.331 | 0.938/0.522 |
| val | E2 过采样 (b32) | 0.9193 | 0.5937 | 0.995/0.945 | **0.852/0.349** | 0.911/0.487 |
| val | E3 合并新图 (b32, 50ep) | 0.9118 | 0.5945 | 0.995/0.936 | 0.853/0.368 | 0.887/0.480 |
| val | baseline (b64) | **0.9234** | **0.6063** | 0.995/0.945 | 0.830/0.345 | 0.945/0.529 |
| val | E3 合并新图 (b64) | 0.9149 | 0.5911 | 0.995/0.948 | **0.853/0.354** | 0.897/0.472 |
| val | E4 CA-backbone (b64) | 0.9161 | 0.5993 | 0.995/0.935 | 0.819/0.359 | 0.934/0.504 |
| test | baseline (b32) | 0.9056 | 0.6086 | 0.995/0.961 | 0.820/0.363 | 0.902/0.502 |
| test | E2 过采样 (b32) | 0.9043 | 0.6027 | 0.995/0.959 | 0.830/0.365 | 0.888/0.483 |
| test | baseline (b64) | 0.9125 | 0.6082 | 0.995/0.956 | 0.834/0.364 | 0.908/0.505 |
| test | E3 合并新图 (b64) | **0.9199** | **0.6096** | 0.995/0.957 | **0.856/0.370** | 0.909/0.501 |
| test | E4 CA-backbone (b64) | 0.9027 | 0.6073 | 0.995/0.949 | 0.804/0.362 | 0.909/0.510 |

结论（都按同一 batch 内的对照说，不跨 batch 比较）：

1. **过采样（E2，b32）**：flashover val +5.2pt、test 只 +1.0pt，broken 掉 1–3pt，
   整体 mAP50-95 降 0.6pt。
2. **并同源新图 + 过采样（E3，b64）**：flashover **两个划分都涨**
   （val 0.830→0.853、test 0.834→0.856），test 整体 mAP50 也 +0.74pt；
   代价是 val 的 broken 掉 4.8pt（0.945→0.897），整体 val 持平。
   方向可复现：补 flashover 数据能让这个短板类稳定 +2pt 左右，但会挤占 broken。
3. **CA 换注入位置（E4，9 层 C2f/SPPF 全挂，b64）**：比同 batch baseline 低
   val 0.7pt / test 1.0pt。注意力不是这一轮的瓶颈。
4. **batch 64 比 batch 32 高 1.2pt mAP50**（0.9234 vs 0.9112），这是本轮最大的
   非模型因素；以后对照必须统一 batch，否则会把 batch 效果误读成模型改进。

**下一步建议（按性价比）**：① 若要保住 broken，把新图只用于 copy-paste 到
「无 broken 图」上、或对 broken 也做过采样，别让 flashover 独占采样权重；
② 换更大 backbone（yolov8s/m）看瓶颈是否在容量；③ 现场真实巡检图才是这
两类混淆的根治手段，公开集同源、天花板已经摸到。

**数据脚本备注**

- `scripts/make_flash_oversample.py` → `训练样本/FBID/yolo/{train_flash3x.txt,data_flash3x.yaml}`；
  txt 每行必须带 `./` 前缀（ultralytics 相对 txt 所在目录解析，否则按 cwd 找、报 corrupt image）。
- `scripts/merge_flash_subset.py` → `训练样本/MERGED/{images,labels}/{train,val,test}` + `data.yaml`；
  类别重映射 `insulator→normal、damaged→broken、Flashover→flashover、hammer→丢弃`；
  FLASH 压缩包里标签在 `labels/labels/<split>/`（双层目录）。
- 全部 run 的对照总表：`results/benchmarks/training_summary.md`（`scripts/summarize_runs.py` 生成）。


### 本轮踩到的坑（换机器/换环境要留意）

- **CA 注入丢失**（见上面第 7 条）：任何「在 YOLO 实例上改模型再 train」的写法都会失效。
- `evaluate()` 的 `model.val(workers=0)` 必须显式传：沙箱下 DataLoader 多进程命名管道被拒
  （`PermissionError: [WinError 5]` 于 `multiprocessing/connection.py::CreateFile`）。
- `train_cdcgan.py` 出图前的描述子曾按 `row.reshape(24, 8, 3)` 算，对 12288 维平铺图必崩；
  已改为 NHWC 下 8×8 块均值。


### 沙箱内对第三方库做的三处补丁（换机器要重打）
1. `ultralytics/data/dataset.py::cache_labels`：`ThreadPool(NUM_THREADS)` → 单线程 `map`。
   沙箱拒绝 multiprocessing 命名管道（`PermissionError: [WinError 5]` 于
   `multiprocessing/connection.py::CreateFile`），不改这里数据集扫描直接崩。
2. `polars==2.1.0` 是 PyPI 上的假包（纯 Python、无 `LazyFrame`），训练结束画图会报
   `name 'PyLazyFrame' is not defined`；改装 `polars 1.30.0 cp39-abi3-win_amd64`。
3. `YOLO_CONFIG_DIR=E:\创新创业\Demos\.tmp_ultra`，避开 `C:\Users\...\AppData` 写权限。

### 环境与坑（重要）

- DSH 沙箱下 `ops_bash` 不可用，命令一律走 `pwsh`。
- `pip` 在受限令牌下完全不可用（`--python` 子进程、`--target` 都会 `WinError 5`）；
  统一用 `.tmp_pip/wheel_install.py` 解 wheel（含 `cp37-abi3`/`cp39-abi3` 回退匹配）。
- **含中文的文件只用 write/edit 工具改**：PowerShell 5.1 的 `Get-Content -Raw` 按 cp936
  解码无 BOM 的 UTF-8，再 `Set-Content` 会把中文写成乱码（本次 `configs/vision.yaml`
  就被这样写坏过，`data.root` 变成 `../璁粌鏍锋湰/...`，训练报 missing path）。
- 装包时 `TEMP/TMP` 指到 `E:\创新创业\Demos\.tmp_pip`，避免写到系统临时目录被沙箱拒绝。
- `Demos` 下仍有 53 个 ACL 受限、删不掉的目录（受限令牌生成的 0700 DACL），
  普通权限删不动，需管理员终端清理：
  `.tmp_pip\` 下 39 个 `pip-*` / `tmp*`（装包残留），
  `insulator_guardian\` 下 `.pytest_basetmp`、`.pytest_tmp`、8 个 `pytest-cache-files-*`、5 个 `tmp*`。
  管理员 PowerShell 里：
  `Remove-Item "E:\创新创业\Demos\.tmp_pip","E:\创新创业\Demos\insulator_guardian\.pytest_basetmp","E:\创新创业\Demos\insulator_guardian\.pytest_tmp" -Recurse -Force -ErrorAction SilentlyContinue`
  其余零散目录按通配删：
  `Get-ChildItem "E:\创新创业\Demos\insulator_guardian" -Filter "pytest-cache-files-*" -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue`、
  `Get-ChildItem "E:\创新创业\Demos\insulator_guardian" -Filter "tmp*" -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue`
- `.venv` 是训练环境，不要删；`data/` 仍为空（FBID 在 `训练样本/`，不进 git）。

## broken 算法验证（batch 档位 × CA 注入位置）

**为什么换 broken**：flashover 与污秽在图像上本就混淆、标注也不稳；broken 是结构破损，
标注可信，适合当「算法有没有用」的验证目标。数据仍是 FBID 1152 张训练 / 128 val / 320 test。

**实验矩阵**（全部 yolov8n、60 epoch、seed 0、`--cache ram`、`--no-ca` 为基线）：

| batch | baseline | CA@SPPF | CA@backbone（9 层） |
| ---: | --- | --- | --- |
| 32 | 0.9380/0.5223 ｜ 0.9015/0.5018 | 0.9525/0.5314 ｜ 0.9013/0.4928 | 0.9408/0.5519 ｜ 0.9050/0.5108 |
| 64 | 0.9446/0.5285 ｜ 0.9081/0.5048 | 0.9440/0.5044 ｜ 0.9048/0.5014 | 0.9343/0.5038 ｜ 0.9087/0.5104 |
| 96 | 0.9514/0.5233 ｜ 0.9095/0.5021 | —（未跑） | 0.9167/0.4988 ｜ 0.9024/0.4808 |

每格 `val ｜ test`，数值是 broken 单类的 `mAP50/mAP50-95`。原始 JSON：
`results/benchmarks/vision_eval_*_{val,test}.json`，汇总表 `results/benchmarks/broken_summary.md`（`summarize_runs.py --focus broken`）。

**结论（按证据强度排）**：

1. **CA 注入位置在 broken 上没有可复现增益。** b32 上 backbone 的 mAP50-95 比 baseline 高
   3.0pt（val），到 b64 变 −2.5pt、b96 变 −2.4pt；同 batch 内三档 mAP50 差距 ≤1.5pt，
   而 seed 间波动 0.9–2.1pt、batch 间 0.7–1.2pt —— 这点差距落在噪声里。
2. **batch 不是越大越好。** baseline 的 broken mAP50 随 batch 上升（0.9380→0.9446→0.9514），
   但 test 几乎不动（0.9015→0.9081→0.9095），整体 mAP50-95 以 b64 最好；batch 96 的价值
   在吞吐（17–19 s/epoch、显存 11.9 GB、GPU 利用率 65%），不在精度。
3. **broken 的实际水平**：mAP50 稳定在 0.90–0.95、mAP50-95 0.50–0.55；test 比 val 低
   0.5–1.5pt（test 只有 192 个 broken 框，抖动大）。

**口径**：CA 只能当「结构选项」，不能写成「提升 mAP 的模块」；本数据上决定精度的是
数据质量与 seed/batch 噪声，不是注意力注入位置。

**要往下走还需要**：① 每档 ≥3 个 seed，用均值±极差才能谈 ±1pt 的差别；② imgsz 960
（broken 目标小，分辨率是最有物理依据的杠杆，单轮 40–50 min）；③ conf 阈值扫描
（`scripts/sweep_broken_conf.py` 已写好未跑）；④ 换 yolov8s/m 探容量边界。

## 困难场景增广（S1/S2/S3）与 RFA 对照（2026-10-30）

**动机**：真实巡检会遇到起雾、抖动、离焦、暗光与雨痕。先不改结构，只在训练集上做
退化模拟，看数据侧还能挤出多少精度。

**构造**（`scripts/make_degraded.py`，产物 `训练样本/DEG/`）：
- 退化算子：雾（大气散射模型）、运动模糊、离焦模糊、低照度+噪声、雨痕；每张随机 1–2 种，
  severity ∈ {1,2,3}，最长边压到 1280。
- 训练集三套：**S1** 原始 FBID 1152；**S2** 仅退化 1152；**S3** 原始+退化混合 2304。
- 测试两套：**干净** = FBID test 320；**困难** = 同 320 张的退化版 `test_deg`。
- 全部 60 epoch、batch 64、`--cache ram`、seed 0，唯一变量是训练集；RFA 用原始数据。

| 训练集 | 测试 | normal P/R | flashover P/R | broken P/R | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S1 原始 | 干净 | .996/.997 | .913/.730 | .941/.859 | 0.911 | 0.603 |
| S1 原始 | 困难 | .992/.903 | .697/.621 | .862/.620 | 0.751 | 0.482 |
| S2 仅困难 | 干净 | .998/.997 | .870/.686 | .862/.880 | 0.881 | 0.577 |
| S2 仅困难 | 困难 | .992/.994 | .829/.650 | .840/.819 | 0.846 | 0.550 |
| **S3 混合** | **干净** | .997/.999 | .896/.787 | .929/.880 | **0.914** | **0.605** |
| **S3 混合** | **困难** | .989/.991 | .861/.707 | .927/.849 | **0.887** | **0.577** |
| RFA（原始） | 干净 | .995/.997 | .880/.734 | .946/.901 | 0.913 | 0.604 |
| RFA（原始） | 困难 | .993/.920 | .840/.581 | .932/.625 | 0.769 | 0.492 |

**结论**：
1. **混合增广（S3）是唯一「干净不掉、困难大涨」的方案**：干净 +0.3pt，困难 +13.6pt
   （broken 0.690→0.883、flashover 0.631→0.784），困难集召回提升（broken .620→.849）
   比精度更明显。
2. **只用退化图（S2）会牺牲干净集**：困难 +9.5pt，但干净 flashover −6.3pt；适合「现场必然
   恶劣」的场景，不适合通用部署。
3. **RFA 单独用几乎不动**：干净 +0.2pt、困难 +1.8pt，都在噪声量级；只有干净 broken
   mAP50 0.904→0.926 稍像样。
4. 排序：干净集 **S3 ≈ RFA ≈ S1 > S2**；困难集 **S3 > S2 > RFA ≈ S1**。数据侧收益（+13.6pt）
   比结构侧（<2pt）高一个数量级，这是本轮最明确的结论。
5. 口径：单 seed；困难测试集是生成的，只代表一种退化分布，不代表真实雾天/雨天。

**下一轮最值得做**：S3 训练集 + RFA（数据+结构两条线同时上），再 3 seed 确认；
以及 `sweep_broken_conf.py` 看困难集 F1 拐点。

**产物**：`results/benchmarks/pr_table_2026-10-30.md`；权重 `results/runs/{real_deg_v8n,
real_degmix_v8n,real_rfa_v8n}/weights/best.pt`。

## 2026-10-31 01:50 · 实验矩阵中断记录（未完成，明日续跑）

用户 01:47 主动停掉训练链，前 7 个 run 已全部 exit=0，第 8 个（real_baseline_b32_960）
跑到约 21 epoch 被终止，半成品目录已删除。

| # | run | 数据 | batch | imgsz | seed | 状态 | 用时 |
|---|-----|------|-------|-------|------|------|------|
| 1 | real_degmix_rfa_v8n | S3 + RFA | 64 | 640 | 0 | 完成 | 28.2 min |
| 2 | real_baseline_b64_s1 | S1 | 64 | 640 | 1 | 完成 | 15.3 min |
| 3 | real_baseline_b64_s2 | S1 | 64 | 640 | 2 | 完成 | 14.5 min |
| 4 | real_deg_b64_s1 | S2 | 64 | 640 | 1 | 完成 | 15.8 min |
| 5 | real_deg_b64_s2 | S2 | 64 | 640 | 2 | 完成 | 17.4 min |
| 6 | real_degmix_b64_s1 | S3 | 64 | 640 | 1 | 完成 | 31.7 min |
| 7 | real_degmix_b64_s2 | S3 | 64 | 640 | 2 | 完成 | 31.8 min |
| 8 | real_baseline_b32_960 | S1 | 32 | 960 | 0 | **未完成** | — |
| 9 | real_degmix_b32_960 | S3 | 32 | 960 | 0 | **未跑** | — |

**续跑命令**（矩阵驱动 `E:/创新创业/Demos/.tmp_run/run_matrix.py`，把 RUNS 里前 7 项删掉即可）：
```powershell
cd E:\创新创业\Demos
.\.venv\Scripts\python.exe .tmp_run\run_matrix.py
```

**评测命令**（7 个已完成 run 的干净/困难双测试集，约 15 分钟）：
```powershell
cd E:\创新创业\Demos\insulator_guardian
..\.venv\Scripts\python.exe scripts\eval_pr_table.py --device 0 `
  --weights "S1 原始=..\results\runs\real_baseline_b64_v8n\weights\best.pt" `
  --weights "S1 seed1=..\results\runs\real_baseline_b64_s1\weights\best.pt" `
  --weights "S1 seed2=..\results\runs\real_baseline_b64_s2\weights\best.pt" `
  --weights "S2 seed1=..\results\runs\real_deg_b64_s1\weights\best.pt" `
  --weights "S2 seed2=..\results\runs\real_deg_b64_s2\weights\best.pt" `
  --weights "S3 seed1=..\results\runs\real_degmix_b64_s1\weights\best.pt" `
  --weights "S3 seed2=..\results\runs\real_degmix_b64_s2\weights\best.pt" `
  --weights "S3+RFA=..\results\runs\real_degmix_rfa_v8n\weights\best.pt" `
  --test "干净=..\训练样本\FBID\yolo\data.yaml" `
  --test "困难=..\训练样本\DEG\data_s1_testdeg.yaml"
```
产物写 `results/benchmarks/pr_table_<date>.md`，之后据此更新 WORKLOG 与报告第 4 节。

## 2026-10-31 14:30 · 七组权重 × 双测试集逐类 P/R（seed 汇总）

数据源：`results/benchmarks/pr_table_2026-10-31.md`（8 组权重 × 2 测试集 × 3 类）。
S1 = 原始 FBID 1152；S2 = 仅人工退化 1152；S3 = 原始 + 退化 2304。
"全体" 为 mAP50 / mAP50-95；逐类为 mAP50。S1 为 3 seed（s0/s1/s2），S2/S3 为 2 seed，S3+RFA 单 seed。

| 训练集 | 干净·全体 | 困难·全体 | 干净·broken | 困难·broken | 干净·flashover | 困难·flashover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1（3 seed 均值） | 0.910 / 0.603 | 0.746 / 0.480 | 0.902 | 0.674 | 0.832 | 0.625 |
| S2（2 seed 均值） | 0.890 / 0.581 | 0.865 / 0.557 | 0.899 | 0.868 | 0.775 | 0.732 |
| S3（2 seed 均值） | **0.922 / 0.606** | **0.887 / 0.579** | **0.927** | **0.897** | **0.844** | **0.770** |
| S3 + RFA（单 seed） | 0.913 / 0.602 | 0.890 / 0.582 | 0.922 | 0.904 | 0.821 | 0.770 |

**结论（seed 化之后，与单 seed 时的判断有修正）**

1. **S3 混合增广是两个测试集同时变好**：干净集 mAP50 +1.2pt（0.910→0.922）、困难集 +14.1pt
   （0.746→0.887）；broken 干净 +2.5pt、困难 +22.3pt。上一轮单 seed 看到的「S3 干净集 −3pt」
   是 seed 噪声（S1 三个 seed 自己就跨 0.904~0.915）。
2. **S2 是「拿干净集换困难集」**：干净集 −2.0pt、flashover −5.7pt，困难集 +11.9pt。
   如果部署场景确定是恶劣天气，S2 可用；否则 S3 更稳。
3. **RFA 叠在 S3 上没有增益**：相对 S3 均值，干净 −0.9pt、困难 +0.4pt、broken −0.5 / +0.8pt，
   全部在 seed 极差（0.9~2.2pt）以内。RFA 无增益这个结论在 3 种数据配方上都成立。
4. **flashover 依然是最弱类**：干净 0.83~0.84、困难仅 0.63~0.77；broken 反而在增广后接近饱和
   （干净 0.93、困难 0.90）。报告里把 broken 当主指标，flashover 降为附录是对的。

**这一轮真正的结论句**：在这个数据与预算下，能稳定拿到的增益来自**数据侧混合增广**（干净 +1.2 /
困难 +14.1 / broken-困难 +22.3），结构侧（CA、RFA）在 seed 噪声内测不出；这条比之前"数据侧远大于
结构侧"的说法更精确——混合增广不仅不掉干净集，还能小幅提升。

## 2026-10-31 16:10 · imgsz 960 两档补齐 + 完整矩阵

| 配方 | 训练分辨率 | 干净·全体 | 困难·全体 | 干净 broken | 困难 broken | 用时 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| S1（3 seed 均值） | 640 | 0.910 / 0.603 | 0.746 / 0.480 | 0.902 | 0.674 | 15 min |
| S2（2 seed 均值） | 640 | 0.890 / 0.581 | 0.865 / 0.557 | 0.899 | 0.868 | 16 min |
| S3（2 seed 均值） | 640 | 0.922 / 0.606 | 0.887 / 0.579 | 0.927 | 0.897 | 32 min |
| S3 + RFA | 640 | 0.913 / 0.602 | 0.890 / 0.582 | 0.922 | 0.904 | 28 min |
| S1 960 | 960 | **0.938 / 0.630** | 0.706 / 0.452 | **0.932** | 0.632 | 35.5 min |
| S3 960 | 960 | 0.936 / 0.619 | **0.892 / 0.575** | 0.916 | 0.871 | 52 min |

（640 的行在 640 下评测，960 的行在 960 下评测；均为 test 320 张，`pr_table_2026-10-31.md`
与 `pr960/pr_table_2026-10-31.md`。）

**结论**

1. **分辨率只在清晰图上换精度**：S1 从 640→960，干净集 +2.8pt（0.910→0.938）、
   broken +3.0pt；但困难集 −4.0pt（0.746→0.706）、broken −4.2pt。S3 也是同样趋势
   （干净 +1.4、困难 +0.5，基本持平）。原因是人工退化（雾/模糊）本身压掉了高频细节，
   高分辨率的优势在输入侧就被抹平，还额外放大了训练/测试分布差。
2. **两个条件都稳的只有 S3 混合增广**：640 下干净 +1.2 / 困难 +14.1；960 下与 S1-960
   打平（0.936 vs 0.938 干净、0.892 vs 0.706 困难）。要一个"全天候"配置，选 S3@640 或 S3@960。
3. **单目标最优**：干净/晴天用 S1@960（0.938，broken 0.932）；恶劣天气用 S3@640
   （困难 0.887、broken 0.897）；两者差 19pt 的困难集差距说明部署时必须按场景选型。
4. RFA 仍然无增益（三种配方都验证过）；flashover 依旧最弱。

**这一轮的完整结论句**：算法侧（CA / RFA / 分辨率）在本数据上都不构成稳定增益，
唯一在两个测试条件下都提升的是**数据侧混合增广**；分辨率是"晴天专用"的开关。

---

## 2026-11-02 21:50 参考资料与跨数据集对照

**参考项目（GitHub 检索，2026-11-02）**：`Deepleen/CACS-YOLO`(21★, AGPL-3.0)、`zhangzhengde0225/FINet`(144★, 无 LICENSE)、`LuYang-2023/Insulator-Defect-Detection-YOLO`(34★, MIT)，均无 release 权重。笔记 `Demos/docs/参考项目对照_CACS-YOLO_FINet.md`。
- CACS-YOLO：backbone 每个 stage 的 C2f 换成 `CACS_C2f`，`CACS_Bottleneck` 把通道对半切后对分支加 `CoordAttlayer`，cat 后 `channel_shuffle`——与我们「整层包 CoordAttWrapper」是不同挂法。
- FINet：径向雾公式（见 `--fog radial`），另主张 13,700 张雾天数据。

**SFID 不可达**：`ihepbox.ihep.ac.cn` https 握手 `SSL: UNEXPECTED_EOF_WHILE_READING`、http 502，40 次重试全败；百度网盘/Google Drive 在本环境不可用。

**改用 HF 镜像（hf-mirror.com）**：
- `KeenForgeAI/CPLID_Insulators-corrected`（72 MB，VOC）→ `训练样本/CPLID/yolo`：143 图 / 286 框（normal 239、broken 47，1152×864）。仓库只上传了 143 张图，845 个 xml 里多数找不到图。
- `introvoyz043/InsulatorDataSet`（213 MB，CPLID 原版）→ `训练样本/IDD/yolo`：248 图 / 248 框，框是**缺陷局部**。

**跨数据集（不微调）**：S3@640 / S1@960 在 CPLID 上 mAP50 0.147 / 0.086（broken 均为 0），在 IDD 上均为 0.000。原因不是单纯过拟合：标注层级不同（整串 vs 缺陷局部）+ 拍摄域差异，跨集数字必须在统一标注口径后才可比。产物 `results/benchmarks/pr_table_2026-11-02.md`。

**代码**：`scripts/make_degraded.py` 新增 `--fog {depth,radial,both}`，默认 depth（旧数据不变）。
