#!/usr/bin/env python
"""把 FBID 训练/测试图渲染成「困难场景」副本：起雾、运动模糊、离焦、暗光、雨痕。

只改像素，不动标注——三种退化都不改变目标位置，标框可以原样复制。
输出目录结构（供 ultralytics 直接吃 txt 清单）：

    <dst>/images/train_deg/   退化后的训练图（与原始一一对应）
    <dst>/labels/train_deg/   原标注的硬链接/复制
    <dst>/images/test_deg/    退化后的测试图
    <dst>/labels/test_deg/
    <dst>/lists/train_orig.txt, train_deg.txt, train_mix.txt, val.txt, test.txt, test_deg.txt
    <dst>/data_s1.yaml … data_s3.yaml, data_s1_testdeg.yaml … data_s3_testdeg.yaml

用法：
    python scripts/make_degraded.py --src ../训练样本/FBID/yolo --dst ../训练样本/DEG
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------- 退化算子
def _to_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), "RGB")


def add_fog(arr: np.ndarray, rng: random.Random, severity: int) -> np.ndarray:
    """大气散射模型 I = J·t + A·(1−t)，t = exp(−β·d)。"""
    h, w = arr.shape[:2]
    beta = {1: 0.8, 2: 1.6, 3: 2.8}[severity]
    air = rng.uniform(0.72, 0.92)
    # 深度代理：上远下近，再叠一点平滑扰动，避免"一条水平线"
    ys = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]
    noise = rng.uniform(-0.12, 0.12)
    d = np.clip(ys + noise + 0.15 * np.sin(np.linspace(0, 3.1, w, dtype=np.float32))[None, :], 0.0, 1.0)
    t = np.exp(-beta * d)[..., None]
    A = np.full_like(arr, air)
    return arr * t + A * (1.0 - t)


def add_radial_fog(arr: np.ndarray, rng: random.Random, severity: int) -> np.ndarray:
    """FINet（zhangzhengde0225/FINet）式径向雾：雾浓度随「到画面中心的距离」衰减。

    原实现（scripts/synthetic_fog.py::fogging_img）：
        size = sqrt(max(h, w))
        d  = -0.04 * ||pixel - center|| + size
        td = exp(-thickness * d)
        fogged = img * td + brightness * (1 - td)
    与 add_fog 的「上远下近」深度代理不同，这里雾以画面中心为圆心：
    d 随到中心的距离减小 -> t 变大 -> **中心雾最浓、边角相对通透**（像镜头前的一团雾）。
    thickness 原实现限制 <=0.1，这里 severity=3 会到 0.18，属于有意加强。
    """
    h, w = arr.shape[:2]
    thickness = {1: 0.05, 2: 0.10, 3: 0.18}[severity]
    brightness = rng.uniform(0.70, 0.85)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    d = -0.04 * dist + np.sqrt(float(max(h, w)))
    td = np.exp(-thickness * d)[..., None]
    return arr * td + brightness * (1.0 - td)


def add_motion_blur(arr: np.ndarray, rng: random.Random, severity: int) -> np.ndarray:
    """沿随机方向做线性运动模糊：用可分离的均匀滤波近似（比逐次平移快两个数量级）。"""
    from scipy.ndimage import uniform_filter1d

    reach = {1: 3, 2: 7, 3: 12}[severity]
    angle = rng.uniform(0, np.pi)
    wx = abs(np.cos(angle)) * reach
    wy = abs(np.sin(angle)) * reach
    out = arr
    if wx >= 0.5:
        out = uniform_filter1d(out, size=int(2 * wx) + 1, axis=1, mode="nearest")
    if wy >= 0.5:
        out = uniform_filter1d(out, size=int(2 * wy) + 1, axis=0, mode="nearest")
    return out


def add_defocus(arr: np.ndarray, rng: random.Random, severity: int) -> np.ndarray:
    radius = {1: 1.2, 2: 2.4, 3: 4.0}[severity]
    return np.asarray(
        Image.fromarray((arr * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius)),
        dtype=np.float32,
    ) / 255.0


def add_lowlight(arr: np.ndarray, rng: random.Random, severity: int) -> np.ndarray:
    gain = {1: 0.55, 2: 0.38, 3: 0.25}[severity]
    noise_sigma = {1: 0.02, 2: 0.04, 3: 0.07}[severity]
    out = arr * gain
    out = out + np.random.default_rng(rng.randrange(1 << 30)).normal(0, noise_sigma, out.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def add_rain(arr: np.ndarray, rng: random.Random, severity: int) -> np.ndarray:
    """雨痕：叠一层半透明细线，再整体轻微模糊。"""
    h, w = arr.shape[:2]
    n = {1: 120, 2: 260, 3: 420}[severity]
    layer = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(layer)
    for _ in range(n):
        x, y = rng.randrange(w), rng.randrange(h)
        ln = rng.randint(8, 26)
        draw.line([(x, y), (x - ln // 4, y + ln)], fill=rng.randint(70, 140), width=rng.choice([1, 1, 2]))
    mask = np.asarray(layer.filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float32) / 255.0
    out = arr * (1.0 - 0.35 * mask[..., None]) + 0.55 * mask[..., None]
    return out


DEGRADATIONS = {
    "fog": add_fog,
    "motion": add_motion_blur,
    "defocus": add_defocus,
    "lowlight": add_lowlight,
    "rain": add_rain,
}


def degrade(img: Image.Image, rng: random.Random, severity: int, ops: list[str] | None = None) -> Image.Image:
    arr = _to_float(img)
    chosen = ops or ["fog", "motion", "defocus", "lowlight", "rain"]
    # 每次取 1~2 种退化叠加，模拟真实巡检里"又雾又抖"的复合退化
    k = rng.choice([1, 1, 2])
    picks = rng.sample(chosen, k)
    for name in picks:
        arr = DEGRADATIONS[name](arr, rng, severity)
        arr = np.clip(arr, 0.0, 1.0)
    return _to_image(arr)


# ---------------------------------------------------------------- 数据集装配
def list_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXT)


def label_path_for(img: Path, src: Path) -> Path | None:
    split = img.parent.name
    cand = src / "labels" / split / f"{img.stem}.txt"
    return cand if cand.exists() else None


def write_list(path: Path, items: list[Path], base: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for p in items:
        p = p.resolve()
        if p.is_relative_to(base.resolve()):
            lines.append("./" + p.relative_to(base.resolve()).as_posix())
        else:
            # 绝对路径直接写，不能再加 "./"，否则 ultralytics 会当成相对路径
            lines.append(p.as_posix())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成困难场景（雾/模糊/暗光/雨）退化数据集")
    ap.add_argument("--src", default="../训练样本/FBID/yolo", help="源 YOLO 数据集根目录")
    ap.add_argument("--dst", default="../训练样本/DEG", help="退化数据集输出目录")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--severity", type=int, default=0, choices=[0, 1, 2, 3], help="0=每张随机 1~3 级")
    ap.add_argument("--lists-only", action="store_true", help="只重建 txt 清单与 data yaml，不重新渲染图像")
    ap.add_argument(
        "--max-size",
        type=int,
        default=1280,
        help="退化前把长边压到该值（YOLO 标注是归一化的，不影响框；0=保持原尺寸）",
    )
    ap.add_argument("--max-step-mb", type=int, default=0, help="占位参数，保持与其他脚本一致的风格")
    ap.add_argument(
        "--fog",
        default="depth",
        choices=["depth", "radial", "both"],
        help="雾模型：depth=上远下近的大气散射（默认，本报告数据用这个）；"
        "radial=FINet 式径向雾；both=每张随机二选一",
    )
    args = ap.parse_args(argv)

    if args.fog == "radial":
        DEGRADATIONS["fog"] = add_radial_fog
    elif args.fog == "both":
        DEGRADATIONS["fog"] = lambda a, r, s: (add_fog if r.random() < 0.5 else add_radial_fog)(a, r, s)

    src = (PROJECT_ROOT / args.src).resolve() if not Path(args.src).is_absolute() else Path(args.src)
    dst = (PROJECT_ROOT / args.dst).resolve() if not Path(args.dst).is_absolute() else Path(args.dst)
    rng = random.Random(args.seed)

    splits = {"train": [], "test": []}
    for split in splits:
        imgs = list_images(src / "images" / split)
        splits[split] = imgs
        print(f"[src] {split}: {len(imgs)} 张")
    if not splits["train"]:
        print(f"[error] 源目录没有训练图：{src / 'images' / 'train'}")
        return 1

    for split, imgs in splits.items():
        out_img = dst / "images" / f"{split}_deg"
        out_lab = dst / "labels" / f"{split}_deg"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lab.mkdir(parents=True, exist_ok=True)
        if args.lists_only:
            print(f"[skip] 复用已有退化图 {out_img}")
            continue
        for i, p in enumerate(imgs, 1):
            sev = args.severity or rng.choice([1, 2, 2, 3])
            with Image.open(p) as im:
                im = im.convert("RGB")
                if args.max_size and max(im.size) > args.max_size:
                    scale = args.max_size / max(im.size)
                    im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR)
                out = degrade(im, rng, sev)
            out.save(out_img / p.name, quality=92)
            lab = label_path_for(p, src)
            if lab is not None:
                (out_lab / lab.name).write_text(lab.read_text(encoding="utf-8"), encoding="utf-8")
            if i % 200 == 0:
                print(f"  [{split}] {i}/{len(imgs)}", flush=True)
        print(f"[out] {out_img} ({len(imgs)})")

    # ultralytics 把 txt 里的相对路径按「txt 所在目录」解析，所以清单必须放在数据集根目录
    lists = dst

    def link_originals(split: str, imgs: list[Path]) -> list[Path]:
        """把原图/原标注硬链接进 DEG，好处是所有清单都能写成相对该根目录的路径。

        Ultralytics 对 txt 里的绝对路径处理不一致（会当成相对当前 yaml 目录），
        统一用 "./images/..." 相对写法最稳。
        """
        out_i = dst / "images" / f"{split}_orig"
        out_l = dst / "labels" / f"{split}_orig"
        out_i.mkdir(parents=True, exist_ok=True)
        out_l.mkdir(parents=True, exist_ok=True)
        linked = []
        for p in imgs:
            target = out_i / p.name
            if not target.exists():
                try:
                    os.link(p, target)
                except OSError:
                    shutil.copy2(p, target)
            lab = label_path_for(p, src)
            if lab is not None:
                lt = out_l / lab.name
                if not lt.exists():
                    try:
                        os.link(lab, lt)
                    except OSError:
                        shutil.copy2(lab, lt)
            linked.append(target)
        return linked

    train_orig = link_originals("train", splits["train"])
    val_orig = link_originals("val", list_images(src / "images" / "val"))
    test_orig = link_originals("test", splits["test"])
    train_deg = list_images(dst / "images" / "train_deg")
    test_deg = list_images(dst / "images" / "test_deg")

    write_list(lists / "train_orig.txt", train_orig, dst)
    write_list(lists / "train_deg.txt", train_deg, dst)
    write_list(lists / "train_mix.txt", train_orig + train_deg, dst)
    write_list(lists / "val.txt", val_orig, dst)
    write_list(lists / "test.txt", test_orig, dst)
    write_list(lists / "test_deg.txt", test_deg, dst)

    names = "{ 0: normal, 1: flashover, 2: broken }"
    sets = {
        "s1": ("原始训练集", "train_orig.txt"),
        "s2": ("困难训练集", "train_deg.txt"),
        "s3": ("原始+困难混合", "train_mix.txt"),
    }
    for key, (title, train_list) in sets.items():
        (dst / f"data_{key}.yaml").write_text(
            f"# {title}\npath: {dst.as_posix()}\ntrain: {train_list}\n"
            f"val: val.txt\ntest: test.txt\nnames: {names}\n",
            encoding="utf-8",
        )
        (dst / f"data_{key}_testdeg.yaml").write_text(
            f"# {title}（评测用困难测试集）\npath: {dst.as_posix()}\ntrain: {train_list}\n"
            f"val: val.txt\ntest: test_deg.txt\nnames: {names}\n",
            encoding="utf-8",
        )
    print(f"[done] data_s{{1,2,3}}.yaml 与 *_testdeg.yaml 已写入 {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
