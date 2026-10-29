"""在真实缺陷裁剪上训练 cDCGAN（构想 1.3.4 的可执行版本）。

做什么
------
1. 用 ``run_enhanced.load_crops`` 从 FBID 标注框裁出 64×64 图块（真实巡检照片）；
2. 用 ``src/enhanced/cdcgan.py`` 的条件 GAN 训练（hinge 损失 + 谱归一化判别器）；
3. 按「生成特征到真实特征中心距离」筛掉偏远的生成样本（``screen_generated``）；
4. 出图：真实 vs 生成对照网格、损失曲线；落 json/csv 指标。

口径
----
- 这是**模块级实验**：生成质量用「中心距」这一半代理，FID 需要 Inception 特征，
  留到接入真实模型后再算。
- 默认 64×64、batch 64；60 epoch 在 4090 上约几分钟，够看趋势不代表收敛最优。

用法
----
    python scripts/train_cdcgan.py                       # 默认 60 epoch
    python scripts/train_cdcgan.py --epochs 120 --batch 128 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.enhanced.cdcgan import (  # noqa: E402
    CdcganConfig,
    TORCH_AVAILABLE,
    build_discriminator,
    build_generator,
    hinge_losses,
    screen_generated,
)
from scripts.run_enhanced import load_crops  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.io import date_tag, ensure_dir, save_json  # noqa: E402

RESULTS_ROOT = PROJECT_ROOT.parent / "results"
CLASS_NAMES = ["normal", "flashover", "broken"]


def _import_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="真实裁剪上的 cDCGAN 小样本生成实验")
    p.add_argument("--data-root", type=str, default=None,
                   help="YOLO 数据集根目录（默认取 configs/vision.yaml data.root）")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--latent", type=int, default=100)
    p.add_argument("--base-channels", type=int, default=128,
                   help="生成器基础通道（默认 128，显存不够可再降）")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=str, default=None, help="输出目录（默认 results/enhanced/cdcgan_<日期>）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not TORCH_AVAILABLE:  # pragma: no cover - 无 torch 环境
        print("[error] cDCGAN 需要 torch：请用 Demos/.venv 里的 python 运行本脚本")
        return 2

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    vision_cfg = load_config("vision")
    data_root = Path(args.data_root) if args.data_root else resolve_path(
        vision_cfg["data"]["root"], base=PROJECT_ROOT
    )
    images, labels, _ = load_crops(data_root, split=args.split, max_crops=2400, seed=args.seed)
    print(f"[data] {data_root} / {args.split}: {len(images)} crops，类别计数 "
          f"{ {n: int((labels == i).sum()) for i, n in enumerate(CLASS_NAMES)} }")

    x = torch.from_numpy((images - 0.5) / 0.5).float()          # [-1, 1]
    y = torch.from_numpy(labels.astype(np.int64))
    device = "cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    if args.device == "cpu":
        device = "cpu"

    cfg = CdcganConfig(latent_dim=args.latent, num_classes=3, image_size=64,
                       base_channels=args.base_channels, lr_g=args.lr, lr_d=args.lr)
    G = build_generator(cfg).to(device)
    D = build_discriminator(cfg).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=cfg.betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=cfg.betas)

    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch, shuffle=True, drop_last=True)
    out_dir = ensure_dir(Path(args.out) if args.out else
                         RESULTS_ROOT / "enhanced" / f"cdcgan_{date_tag()}")
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        g_sum = d_sum = 0.0
        n_batches = 0
        for real, lab in loader:
            real, lab = real.to(device), lab.to(device)
            bsz = real.size(0)
            z = torch.randn(bsz, cfg.latent_dim, device=device)

            # --- 判别器 ---
            fake = G(z, lab).detach()
            d_loss, _ = hinge_losses(D(real, lab), D(fake, lab), cfg.hinge_margin)
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            # --- 生成器 ---
            z = torch.randn(bsz, cfg.latent_dim, device=device)
            fake = G(z, lab)
            _, g_loss = hinge_losses(D(real, lab).detach(), D(fake, lab), cfg.hinge_margin)
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            d_sum += float(d_loss)
            g_sum += float(g_loss)
            n_batches += 1

        rec = {"epoch": epoch, "d_loss": d_sum / max(1, n_batches),
               "g_loss": g_sum / max(1, n_batches), "elapsed_s": round(time.time() - t0, 1)}
        history.append(rec)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[train] epoch {epoch:3d}/{args.epochs}  d={rec['d_loss']:.3f}  "
                  f"g={rec['g_loss']:.3f}  {rec['elapsed_s']:.0f}s")

    # --- 生成 + 筛选（中心距代理） ---
    G.eval()
    with torch.no_grad():
        z = torch.randn(len(x), cfg.latent_dim, device=device)
        lab_all = y.to(device)
        gen = G(z, lab_all).cpu().numpy()
    gen_images = (gen + 1.0) / 2.0
    # 8×8 块均值下采样当描述子（64 能被 8 整除），避免像素噪声主导中心距
    def _desc(nhwc: np.ndarray) -> list[list[float]]:
        blocks = nhwc.reshape(len(nhwc), 8, 8, 8, 8, 3).mean(axis=(2, 4))
        return blocks.reshape(len(nhwc), -1).tolist()

    keep = screen_generated(
        _desc(np.transpose(gen_images, (0, 2, 3, 1))),
        _desc(np.transpose(images, (0, 2, 3, 1))),
        keep_ratio=float(vision_cfg.get("enhanced", {}).get("cdcgan", {}).get("keep_ratio", 0.5)),
    )

    # --- 出图 ---
    plt = _import_plt()
    n_show = min(8, len(images))
    fig, axes = plt.subplots(2, n_show, figsize=(1.6 * n_show, 3.6))
    picks = rng.choice(len(images), size=n_show, replace=False)
    for j, idx in enumerate(picks):
        axes[0, j].imshow(np.transpose(images[idx], (1, 2, 0)))
        axes[0, j].set_title(CLASS_NAMES[int(labels[idx])], fontsize=8)
        axes[1, j].imshow(np.transpose(gen_images[idx], (1, 2, 0)))
        axes[1, j].set_title("gen", fontsize=8)
        for row in (0, 1):
            axes[row, j].axis("off")
    fig.suptitle("real crops (top) vs cDCGAN generations (bottom)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "cdcgan_samples.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=140)
    ax.plot([h["epoch"] for h in history], [h["d_loss"] for h in history], label="D loss")
    ax.plot([h["epoch"] for h in history], [h["g_loss"] for h in history], label="G loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("hinge loss")
    ax.legend()
    ax.set_title("cDCGAN training")
    fig.tight_layout()
    fig.savefig(out_dir / "cdcgan_loss.png", dpi=140)
    plt.close(fig)

    with (out_dir / "loss_history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    torch.save({"model": G.state_dict(), "config": cfg.__dict__}, out_dir / "generator.pt")
    metrics = {
        "data_root": str(data_root),
        "split": args.split,
        "crops": int(len(images)),
        "class_counts": {n: int((labels == i).sum()) for i, n in enumerate(CLASS_NAMES)},
        "device": device,
        "epochs": args.epochs,
        "batch": args.batch,
        "latent": args.latent,
        "base_channels": args.base_channels,
        "final_d_loss": history[-1]["d_loss"],
        "final_g_loss": history[-1]["g_loss"],
        "kept_ratio": round(len(keep) / max(1, len(gen_images)), 3),
        "screening": "中心距代理（FID 待接 Inception 特征）",
        "elapsed_s": history[-1]["elapsed_s"],
    }
    save_json(metrics, out_dir / "cdcgan_metrics.json")

    print(f"[done] {out_dir}")
    print(f"  cdcgan_samples.png / cdcgan_loss.png / generator.pt / cdcgan_metrics.json")
    print(f"  筛选保留 {len(keep)}/{len(gen_images)}（中心距代理，非 FID）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
