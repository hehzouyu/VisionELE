# 参考项目对照笔记：CACS-YOLO 与 FINet

> 用途：判断这两个开源项目里**哪些能搬到 `Demos/insulator_guardian`**，以及搬的成本。
> 参考时间：2026-11-02；源码已解压在 `Demos/.tmp_refs/`（`CACS-YOLO-main/`、`FINet-master/`）。
> 当前基线：ultralytics 8.4.192 + YOLOv8n，类目 normal/flashover/broken，数据 FBID + DEG。

## 0. 速览

| | CACS-YOLO | FINet |
| --- | --- | --- |
| 许可 | AGPL-3.0（代码不能并进闭源/自研仓库） | 无 LICENSE（代码不可复用） |
| 底座 | ultralytics 分支（≈8.x），`YOLO(model='...yaml')` 直接跑 | YOLOv5 分支 + `hai` 框架，`pytorch<=1.10` |
| 结构改动 | 把 backbone 的 `C2f` 换成 `CACS_C2f`（注意力在 bottleneck 内部） | `SE-YOLOv5`：SE 通道注意力插在 C3/C3TR 里 |
| 数据 | SFRID 20,567 张（原图 + 合成雾 + 合成雨），两类 | SFID 13,700 张雾天图（5.92 GB，有直链） |
| 可搬运程度 | 结构可搬（重写），数据可下 | 只搬算法与数据，代码别抄 |
| 部署难度 | 低（与我们同栈，加一个模块 + 一份 yaml） | 中（旧 torch 锁定 + hai 依赖） |

结论先说：**结构学 CACS-YOLO，造雾学 FINet，两边都不直接抄代码。**

---

## 1. CACS-YOLO：注意力挂在 `C2f` 内部，而不是挂整层

### 1.1 它到底改了什么

`ultralytics/cfg/models/CACS-YOLO.yaml` 的 backbone：

```
- [-1, 3, CACS_C2f, [128, True]]   # 3
- [-1, 6, CACS_C2f, [256, True]]   # 6
- [-1, 6, CACS_C2f, [512, True]]   # 9
- [-1, 3, CACS_C2f, [1024, True]]  # 12
- [-1, 1, SPPF, [1024, 5]]         # 13
```

即**每个 stage 的 C2f 都被换掉**，head 仍是普通 `C2f`。`nc: 2`（insulator / defect），`depth 0.67 / width 0.75`（≈YOLOv8m 的缩放）。

### 1.2 `CACS_Bottleneck` 的核心（`ultralytics/nn/modules/block.py:37`）

```python
class CACS_Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        self.add = shortcut and c1 == c2
        c1 >>= 1          # 通道对半切
        c2 >>= 1
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)                 # 1x1 降维
        self.conv = nn.Conv2d(c_, c2, k[1], 1, autopad(3, None), groups=g, bias=False)  # 深度可分离
        self.bn = nn.BatchNorm2d(c2)
        self.ca = CoordAttlayer(c2, c2, reduction=32)    # ★ 注意力加在分支上
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.cv1(x2); x2 = self.conv(x2); x2 = self.bn(x2)
        x2 = self.ca(x2)                                  # ★ 只对分支做注意力
        out = torch.cat((x2, x1), dim=1)
        out = self.act(out)
        return channel_shuffle(out, 2)                    # ★ 通道重排
```

`CoordAttlayer`（`block.py:136`）就是标准 CoordAttention：H/W 双向池化 → 1×1 + h_swish → 拆回两个方向的注意力图 → `x * a_h * a_w`。和我们 `src/vision/attention.py:39` 的 `CoordAtt` 是同一个东西（我们用的是 `mean(dim=3)`/`mean(dim=2)` 版本，数学等价）。

### 1.3 与我们现在做法的差别（这是最值得记的一条）

| | 我们（`inject_coordatt`，mode=last/backbone） | CACS-YOLO |
| --- | --- | --- |
| 挂载粒度 | **整层包装**：`CoordAttWrapper(整层)`，注意力作用在该层输出特征图上 | **bottleneck 内部**：只作用于 C2f 里被切出来的一半分支 |
| 覆盖范围 | 选 9 个层名的层（C2f/SPPF…），每层一个 CA | backbone 每个 stage 的每个 bottleneck 里都有 |
| 附带改动 | 无 | 通道对半切 + 深度可分离卷积 + `LeakyReLU` + `channel_shuffle`（轻量化组合拳） |
| 参数量 | 我们 CA@backbone 3,172,745 | 官图给的是 v8m 级别，不能直接比 |

这解释了一件事：**我们测「CA 无增益」，测的是「整层输出后置一个 CA」这种用法；CACS-YOLO 用的是「CSP 分支内的注意力 + 通道重排」**。两者不是同一件事，不能拿我们的负结果去否定它，反过来也不能拿它的正结果去证明我们该改。

### 1.4 搬到我们仓库要动的地方

1. 新建 `src/vision/cacs.py`，把 `CoordAttlayer`、`CACS_Bottleneck`、`CACS_C2f` 抄成独立模块（`channel_shuffle` 用 `torch.nn.functional.channel_shuffle`，torch≥1.9 自带，不用手写）。
2. 因为 ultralytics 的 `parse_model` 是在 `ultralytics/nn/tasks.py` 的模块 globals 里按名字找类，所以不用改它的源码，**运行时把类塞进那个命名空间**即可：
   ```python
   import ultralytics.nn.tasks as _tasks
   from src.vision.cacs import CACS_C2f
   _tasks.CACS_C2f = CACS_C2f    # parse_model 里的 eval(module_name) 就能找到
   ```
   然后 `YOLO("configs/cacs_yolov8n.yaml")` 就能建模。
3. 抄一份我们的 `configs/vision.yaml` 对应的架构 yaml，把 backbone 的 `C2f` 换成 `CACS_C2f`，`nc` 改 3。
4. 训练入口不用改（`scripts/train_vision.py` 已经支持 `--data-yaml / --epochs / --batch`），但 CA 注入要关掉（`--no-ca`），否则两套注意力叠加会污染对照。
5. 评测仍走 `scripts/eval_pr_table.py`，加一个权重即可，和现有 8 组权重并排比。

### 1.5 风险

- **AGPL-3.0**：拿来跑实验、在报告里引用没问题；但如果你要把 `CACS_C2f` 的**代码原样**并进 VisionELE 仓库，整个仓库就受 AGPL 约束。稳妥做法是**按论文/CVPR 级描述自己实现一遍**（上面的结构描述足够重写），并在报告里标注思路来源。
- 它的 `nc: 2` 与我们三类不同，改 `nc: 3` 后 `Detect` 头会自动重建，权重不能直接迁移，只能从头训。

---

## 2. FINet：造雾公式值得抄，代码不值得抄

### 2.1 它的雾模型（`scripts/synthetic_fog.py:76`）

```python
d  = -0.04 * np.linalg.norm(mask, axis=2) + size     # mask = 像素到图像中心的距离
td = np.exp(-thickness * d)                          # 透射率
fogged = img * td + brightness * (1 - td)            # 大气散射公式
```

- `size = sqrt(max(h, w))`，`brightness ∈ [0,1]`（默认 0.7），`thickness ∈ [0,0.1]`（默认 0.05）
- **雾心在图像正中间，向外线性变淡**（径向雾），不是按深度
- 只有高效率分支做了 `np.clip(0,1)`，循环分支没做——抄的时候记得补 clip

### 2.2 我们现在的雾（`scripts/make_degraded.py:42`）

```python
ys = np.linspace(1.0, 0.0, h)              # 深度代理：上远下近
d  = clip(ys + noise + 0.15*sin(...))      # 加抖动，避免出现一条水平线
t  = exp(-beta * d)
return arr * t + A * (1 - t)               # A = 空气光亮度
```

区别：

| | FINet | 我们 |
| --- | --- | --- |
| 深度来源 | 到图像中心的像素距离（径向） | 上下渐变 + 正弦抖动（伪深度） |
| 雾的分布 | 中间浓、四周淡（像镜头前有雾） | 远处浓、近处淡（像看远处） |
| 空气光 A | 单值 `brightness` | 随 severity 变化的 `air` |
| 速度 | 矩阵版 ~80 ms（864×1152） | numpy 向量化，同量级 |

两者都**不是物理正确的深度雾**，但方向互补：我们的更像"相机看远处的雾"，FINet 的更像"镜头起雾"。

### 2.3 搬运建议（成本很低）

在 `make_degraded.py` 里加一个 `add_radial_fog()`，作为第五种退化：

```python
def add_radial_fog(arr, rng, severity):
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    thickness = {1: 0.02, 2: 0.04, 3: 0.06}[severity] * rng.uniform(0.8, 1.2)
    brightness = rng.uniform(0.68, 0.85)
    d = -0.04 * dist + np.sqrt(max(h, w))
    td = np.exp(-thickness * d)[..., None]
    return np.clip(arr * td + brightness * (1 - td), 0, 1)
```

之后 S2/S3 的退化组合从 5 选 1-2 变成 6 选 1-2，重新生成 DEG 只多花几分钟。**这是本轮性价比最高的一次数据改动**，因为它把"径向起雾"这个真实场景补进了训练分布。

### 2.4 数据侧：SFID 值不值得下

- 13,700 张、5.92 GB、直链可下（`ihepbox`），含雾天标注。
- 用途不是合并训练（类别体系和 FBID 不同、标注质量未知），而是**做域外测试集**：拿我们最好的一组权重直接在 SFID 上跑，看跨数据集掉多少。这个数字比再刷 0.5pt mAP 有说服力。
- 注意：文档里没写数据许可，只用做学术对照，报告里注明来源。

### 2.5 风险

- 仓库无 LICENSE → 代码不能抄进仓库。上面的雾公式是数学描述 + 我们自己的 numpy 实现，属于算法复现，风险可控；但最好在报告里写明"参考 FINet 的径向雾模型"。
- `requirements.txt` 锁 `pytorch<=1.10`，且依赖 `hai`（`hepai`）框架，想复现它的训练得另建环境——**不建议**，只取公式。

---

## 3. 落地清单（按性价比）

| # | 动作 | 改动位置 | 成本 | 收益 |
| --- | --- | --- | --- | --- |
| 1 | 加径向雾 `add_radial_fog`，重生成 DEG | `scripts/make_degraded.py` | 30 min | 数据分布更接近"镜头起雾"，S2/S3 各补一组 |
| 2 | 实现 `CACS_C2f` 并注册进 ultralytics 命名空间 | 新增 `src/vision/cacs.py` + `configs/cacs_yolov8n.yaml` | 1.5 h | 得到一组"分支内注意力"对照，能回答 CA 到底该挂哪 |
| 3 | 训练 CACS-YOLOv8n（同数据同 batch） | `scripts/train_vision.py --no-ca` | 训练 15 min | 与现有 8 组权重并排 |
| 4 | SFID 域外测试 | `scripts/eval_pr_table.py --test SFID=data_sfid.yaml` | 20 min + 下载 | 跨域泛化数字，比同分布 mAP 更硬 |
| 5 | 报告加"外部对照"小节 | 10-31 版 tex 的 §4 | 30 min | 回答"为什么不跟 SOTA 比"的质疑 |

合计约 4 小时（下载另算），其中 #1 和 #4 最划算。

---

## 4. 一句话总结

- **CACS-YOLO 卖的是「注意力往 CSP 分支里塞 + 通道重排」，不是「末端加一个 CA」**——这正好补上我们之前没试过的那一种挂法。
- **FINet 卖的是「径向雾 + 13,700 张雾天数据」**，公式三行就能复现，数据能当域外测试集。
- 两个都**不要直接抄代码**（AGPL / 无许可），按论文描述自己实现，报告里标注思路来源。

---

## 5. 实测补充（2026-11-02 晚）

### 5.1 SFID 下载：不可达
- `https://ihepbox.ihep.ac.cn/ihepbox/index.php/s/adTHe1UPu0Vc7vI/download` 的 https 握手报
  `SSL: UNEXPECTED_EOF_WHILE_READING`，http 返回 502，40 次重试（含 UA/legacy SSL）全部失败；
  百度网盘与 Google Drive 的备用通道在本环境也不可用。**结论：SFID 走不通，跨数据集对照改用 HF 镜像。**

### 5.2 实测补充：换用 CPLID / IDD（HF 镜像，hf-mirror.com）
- `KeenForgeAI/CPLID_Insulators-corrected`：989 个文件 / 72.1 MB，VOC 标注。
  仓库只带了 143 张图（845 个 xml，其余图未上传），转 YOLO 后 **143 图 / 286 框**
  （normal 239、broken 47），统一 1152×864。
- `introvoyz043/InsulatorDataSet`：991 个文件 / 213.7 MB，CPLID 原版布局
  （`data/Defective_Insulators/{images,labels/defect}` + `data/Normal_Insulators/images`），
  xml 的 `<filename>` 不带扩展名。转出 **248 图 / 248 框**，框标的是**缺陷区域**而不是整串绝缘子。

### 5.3 跨数据集结果（本项目权重直接推理，不做微调）
| 权重 | 测试集 | 全体 mAP50 / mAP50-95 | 逐类 mAP50 |
| --- | --- | --- | --- |
| S3@640 | CPLID | **0.147 / 0.057** | normal 0.293、broken **0.000** |
| S1@960 | CPLID | **0.086 / 0.039** | normal 0.172、broken **0.000** |
| S3@640 | IDD | **0.000 / 0.000** | broken 0.000（P 0.002 / R 0.004） |
| S1@960 | IDD | **0.000 / 0.000** | broken 0.000（P 0.003 / R 0.004） |

产物：`Demos/results/benchmarks/pr_table_2026-11-02.md`。

### 5.4 这两个数字怎么读（别写错）
- 它**不是**「模型在别的数据集上只有 0.15 mAP」这么简单：两边的标注口径根本不同。
  我们训的是「整串绝缘子 + 缺陷类型」，CPLID/IDD 的 defect 框是**裂纹/破损的小区域**，
  同一个图里我们的模型会把整串判成 normal，与 defect 小框的 IoU 接近 0 → AP 归零。
- 因此这两个结果真正说明的是两件事：
  1. **域差距确实存在且很大**：干净集 0.91 mAP50 不能外推到别的巡检数据；
  2. **标注口径不兼容**：要拿公开缺陷集做交叉验证，必须先把标签统一到「同一层级」
     （整串级 vs 缺陷局部级），否则任何跨集数字都不可比。
- 能直接引用的一句话：*FBID 上训练的三个配方，在 CPLID/IDD 上 cross-dataset 全部低于 0.15 mAP50，
  其中 broken 为 0；主要原因是标注层级不同（整串 vs 缺陷局部）与拍摄域差异，不是单纯过拟合。*

### 5.5 顺带落地的一处改进
- `scripts/make_degraded.py` 新增 `--fog {depth,radial,both}`：
  `radial` 用 FINet 的径向雾公式（`d = -0.04·r + sqrt(max(h,w))`、`t = exp(-thickness·d)`，中心雾最浓），
  默认仍是本报告数据用的 `depth`（上远下近），旧数据不受影响。
