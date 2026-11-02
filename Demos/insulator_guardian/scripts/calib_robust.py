"""标定鲁棒性：相机参数扰动对世界坐标的影响。

相机装了但没标定准（或车体俯仰随负载变化），像素落点会被算偏多少？
这个脚本把整幅图像的射线批量投到地面，和标称参数下的落点比，按距离分箱统计
误差分位数。全部是 numpy 向量化，几秒钟跑完。

用法::

    python scripts/calib_robust.py
    python scripts/calib_robust.py --conf configs/pipeline.yaml --out-dir ../results/calibration
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.coords import CameraModel  # noqa: E402
from src.utils.config import cfg_get, load_config, resolve_path  # noqa: E402
from src.utils.io import date_tag, ensure_dir, save_json  # noqa: E402

POSES = (0.0, 0.0, 0.0)


def _rot_z(a: np.ndarray | float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def ground_points(cam: CameraModel, us: np.ndarray, vs: np.ndarray,
                  pitch_deg: float | None = None,
                  height_m: float | None = None,
                  roll_deg: float | None = None,
                  fx_scale: float = 1.0,
                  yaw_deg: float = 0.0) -> np.ndarray:
    """批量像素 → 地面点，返回 (N,3)，打不到地面的行填 nan。"""
    pitch = math.radians(cam.pitch_deg if pitch_deg is None else pitch_deg)
    roll = math.radians(cam.roll_deg if roll_deg is None else roll_deg)
    height = cam.height_m if height_m is None else height_m
    fx, fy = cam.fx * fx_scale, cam.fy * fx_scale

    x = (us - cam.cx) / fx
    y = (vs - cam.cy) / fy
    rays = np.stack([x, y, np.ones_like(x)], axis=1).astype(np.float32)
    rays = rays @ _rot_x(roll).T
    robot = np.stack([rays[:, 2], -rays[:, 0], -rays[:, 1]], axis=1).astype(np.float32)
    rays = robot @ _rot_y(pitch).T
    rays = rays @ _rot_z(math.radians(yaw_deg)).T

    mount = _rot_z(math.radians(yaw_deg)) @ np.array([cam.mount_x, cam.mount_y, 0.0],
                                                    dtype=np.float32)
    origin = np.array([mount[0], mount[1], height], dtype=np.float32)
    dz = rays[:, 2]
    t = np.where(dz < -1e-6, -origin[2] / dz, np.nan)
    pts = origin[None, :] + t[:, None] * rays
    pts[t < 0] = np.nan
    return pts


def summarize(nominal: np.ndarray, perturbed: np.ndarray,
              bins: list[tuple[float, float]]) -> dict[str, Any]:
    ok = np.isfinite(nominal[:, 0]) & np.isfinite(perturbed[:, 0])
    err = np.linalg.norm(perturbed[ok, :2] - nominal[ok, :2], axis=1)
    dist = np.linalg.norm(nominal[ok, :2], axis=1)
    out: dict[str, Any] = {"n": int(err.size), "all": _stats(err)}
    for lo, hi in bins:
        mask = (dist >= lo) & (dist < hi)
        if mask.any():
            out[f"{lo:g}-{hi:g}"] = _stats(err[mask])
    return out


def _stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="相机标定误差对地面坐标的影响")
    ap.add_argument("--conf", default="pipeline.yaml", help="配置文件名或路径")
    ap.add_argument("--grid", type=int, nargs=2, default=[161, 121], metavar=("NU", "NV"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    conf = load_config("pipeline") if args.conf == "pipeline.yaml" else load_config(args.conf)
    cam_cfg = cfg_get(conf, "camera", {}) or {}
    cam = CameraModel.from_dict(cam_cfg)

    us = np.linspace(0, cam.width - 1, args.grid[0], dtype=np.float32)
    vs = np.linspace(cam.cy, cam.height - 1, args.grid[1], dtype=np.float32)
    UU, VV = np.meshgrid(us, vs)
    us, vs = UU.ravel(), VV.ravel()

    base = ground_points(cam, us, vs)
    visible = base[np.isfinite(base[:, 0])]
    dists = np.linalg.norm(visible[:, :2], axis=1)
    near, far = float(dists.min()), float(dists.max())
    edges = np.linspace(near, far, 5)
    bins = [(float(edges[i]), float(edges[i + 1])) for i in range(4)]
    band = {
        "near_m": near, "far_m": far,
        "span_m": far - near,
        # 光轴（v=cy）落点：这是几何上能看到的"最远处"
        "axis_hit_m": cam.height_m / math.tan(cam.pitch_rad),
    }

    cases: dict[str, dict[str, Any]] = {}
    specs = [
        ("pitch +1°", dict(pitch_deg=cam.pitch_deg + 1)),
        ("pitch +2°", dict(pitch_deg=cam.pitch_deg + 2)),
        ("pitch +3°", dict(pitch_deg=cam.pitch_deg + 3)),
        ("pitch -2°", dict(pitch_deg=cam.pitch_deg - 2)),
        ("height +0.05m", dict(height_m=cam.height_m + 0.05)),
        ("height +0.10m", dict(height_m=cam.height_m + 0.10)),
        ("height +0.20m", dict(height_m=cam.height_m + 0.20)),
        ("roll +1°", dict(roll_deg=cam.roll_deg + 1)),
        ("roll +3°", dict(roll_deg=cam.roll_deg + 3)),
        ("yaw +1°", dict(yaw_deg=1.0)),
        ("fx +2%", dict(fx_scale=1.02)),
        ("组合(俯+2°,高+0.1,滚+1°)",
         dict(pitch_deg=cam.pitch_deg + 2, height_m=cam.height_m + 0.10,
              roll_deg=cam.roll_deg + 1)),
    ]
    for name, kwargs in specs:
        cases[name] = summarize(base, ground_points(cam, us, vs, **kwargs), bins)

    # 检测框中心 5 像素抖动等价于整幅图里 ±5px 的偏移，单独量一次
    shift = ground_points(cam, us + 5.0, vs)
    cases["bbox 中心 +5px (u)"] = summarize(base, shift, bins)

    out_root = (Path(args.out_dir).resolve() if args.out_dir
                else resolve_path(cfg_get(conf, "paths.results_root", "../results")) / "calibration")
    out_root = ensure_dir(out_root)
    tag = date_tag()

    payload = {"camera": cam.__dict__, "cases": cases,
               "bins_m": [[lo, hi] for lo, hi in bins], "visible_band": band}
    json_path = save_json(payload, out_root / f"calib_robust_{tag}.json")

    md = ["# 相机标定鲁棒性：参数扰动 → 地面坐标误差", "",
          f"相机：{cam.width}×{cam.height}、fx={cam.fx:g}、height={cam.height_m:g} m、"
          f"pitch={cam.pitch_deg:g}°、roll={cam.roll_deg:g}°；"
          f"采样 {us.size} 条射线（v ≥ cy，只取地平线以下）。", "",
          "## 几何本身的可视带", "",
          f"- 地板可见区间：**{near:.2f} – {far:.2f} m**（跨度 {far - near:.2f} m）",
          f"- 光轴（画面中心）落点：{band['axis_hit_m']:.2f} m —— 挂高 "
          f"{cam.height_m:g} m / 俯仰 {cam.pitch_deg:g}° 下，再远的东西根本不在画面里。",
          "", "单位：米。表格里是落点误差的中位数 / p95。", ""]
    header = " | ".join(f"{lo:.1f}–{hi:.1f} m" for lo, hi in bins)
    md += [f"| 扰动 | {header} | 全体 p95 |",
           "| --- | " + " | ".join(["---:"] * (len(bins) + 1)) + " |"]
    for name, item in cases.items():
        cells = []
        for lo, hi in bins:
            key = f"{lo:g}-{hi:g}"
            st = item.get(key)
            cells.append(f"{st['median']:.3f} / {st['p95']:.3f}" if st else "—")
        md.append(f"| {name} | " + " | ".join(cells) + f" | {item['all']['p95']:.3f} |")
    md += ["", "读法：误差随距离单调放大，所以真正决定「缺陷点被放到哪」的是可视带远端那一档。", ""]
    worst = max(cases.items(), key=lambda kv: kv[1]["all"]["p95"])
    md.append(f"最差情形：{worst[0]}，全体 p95 = {worst[1]['all']['p95']:.3f} m。")
    md_path = out_root / f"calib_robust_{tag}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    if not args.no_figure:
        _figure(cases, bins, cam, out_root / f"calib_robust_{tag}.png")

    print(f"[done] {json_path}")
    print(f"[done] {md_path}")
    for name, item in cases.items():
        print(f"  {name:<26} p95={item['all']['p95']:.3f} m  max={item['all']['max']:.3f} m")
    return 0


def _figure(cases: dict[str, dict[str, Any]], bins: list[tuple[float, float]],
            cam: CameraModel, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from plot_utils import setup_cjk_font

        setup_cjk_font()
    except Exception:  # 字体探测失败也不影响出图
        pass

    labels = [f"{lo:.1f}–{hi:.1f}" for lo, hi in bins]
    picks = ["pitch +1°", "pitch +2°", "pitch +3°", "height +0.10m",
             "roll +3°", "组合(俯+2°,高+0.1,滚+1°)"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=150)
    for name in picks:
        item = cases.get(name)
        if not item:
            continue
        ys = [item.get(f"{lo:g}-{hi:g}", {}).get("p95", 0.0) for lo, hi in bins]
        ax.plot(labels, ys, marker="o", markersize=3.5, linewidth=1.4, label=name)
    ax.set_xlabel("地面距离分箱 (m)")
    ax.set_ylabel("落点误差 p95 (m)")
    ax.set_title(f"相机参数扰动 → 地面坐标误差 p95（挂高 {cam.height_m:g} m，"
                 f"俯仰 {cam.pitch_deg:g}°）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
