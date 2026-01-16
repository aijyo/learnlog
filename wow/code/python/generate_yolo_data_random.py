# -*- coding: utf-8 -*-
"""
YOLOv8 dataset generator for WoW skill icon box (icon=64 default)

Key behavior (v2):
- Auto-select monitor by click position (multi-monitor safe).
- Click point is ANY point inside the target icon (not necessarily center).
- Crop is NOT centered at click; click can lie anywhere inside the crop.
- Allow partial bbox truncation: label uses intersection (bbox ∩ crop).
- Optional: avoid capturing neighboring box centers (assume neighbor centers at +/- neighbor_step on x/y).
- Guarantee ok == N by retrying.

Output structure:
  out_dir/
    images/train/*.jpg|png
    images/val/*.jpg|png
    labels/train/*.txt
    labels/val/*.txt
    data.yaml

Notes:
- This script still uses a synthetic geometry model (grid-like neighbor centers). If your UI layout
  differs (e.g. non-uniform spacing), consider disabling --exclude-neighbors and simply keep crop
  size small.
"""

import os
import time
import random
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import mss
from PIL import Image
from pynput import mouse
import yaml


@dataclass
class Config:
    out_dir: str
    n_samples: int
    icon_size: int                  # bbox size (square), default 64
    crop_min: int                   # desired min crop size
    crop_max: int                   # desired max crop size
    neighbor_step: int              # assumed neighbor center distance (pixels)
    neighbor_margin: int            # safety margin (pixels)
    exclude_neighbors: bool         # whether to reject crops that include neighbor centers
    click_pad: int                  # keep click at least this many pixels away from bbox edges
    min_visible_ratio: float        # min intersection area ratio to keep label
    val_ratio: float
    seed: int
    delay_sec: float
    img_ext: str
    jpg_quality: int
    class_name: str
    max_tries_per_sample: int


def ensure_dirs(root: str):
    # Create YOLO directory structure
    paths = [
        os.path.join(root, "images", "train"),
        os.path.join(root, "images", "val"),
        os.path.join(root, "labels", "train"),
        os.path.join(root, "labels", "val"),
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)


def wait_for_left_click() -> Tuple[int, int]:
    """Wait for the first left click and return absolute screen coordinates."""
    clicked = {"pos": None}

    def on_click(x, y, button, pressed):
        if pressed and button == mouse.Button.left:
            clicked["pos"] = (int(x), int(y))
            return False  # stop listener

    print("\n[1/3] 请在目标技能图标上随便点一下（点在图标内部任意位置即可），左键单击一次...")
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()

    pos = clicked["pos"]
    if pos is None:
        raise RuntimeError("No click captured.")
    print(f"[OK] 捕获点击坐标: {pos}")
    return pos


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def rect_intersection(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1) -> Optional[Tuple[float, float, float, float]]:
    """Return intersection rect (x0,y0,x1,y1), or None if no overlap."""
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def write_yolo_label(label_path: str, class_id: int, x_center_n: float, y_center_n: float, w_n: float, h_n: float):
    """Write a single bbox in YOLO format (normalized)."""
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(f"{class_id} {x_center_n:.6f} {y_center_n:.6f} {w_n:.6f} {h_n:.6f}\n")


def save_data_yaml(root: str, class_name: str):
    """Create YOLO data.yaml."""
    data = {
        "path": os.path.abspath(root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_name},
    }
    yaml_path = os.path.join(root, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"[OK] 写入: {yaml_path}")


def pick_monitor_by_point(sct: mss.mss, x: int, y: int) -> Tuple[int, Dict]:
    """Pick the monitor that contains the point (x,y)."""
    for mi in range(1, len(sct.monitors)):
        m = sct.monitors[mi]
        if (m["left"] <= x < m["left"] + m["width"] and
                m["top"] <= y < m["top"] + m["height"]):
            return mi, m
    return 1, sct.monitors[1]


def point_in_crop(px: float, py: float, crop_left: float, crop_top: float, crop_w: float, crop_h: float, margin: float = 0.0) -> bool:
    """Check whether a point is inside crop rectangle with optional inner margin."""
    return (crop_left + margin) <= px <= (crop_left + crop_w - margin) and (crop_top + margin) <= py <= (crop_top + crop_h - margin)


def sample_bbox_center_from_click(click_x: float, click_y: float, icon_size: int, click_pad: int) -> Tuple[float, float]:
    """Sample bbox center so that click lies inside the bbox (with a pad away from bbox edges)."""
    half = icon_size / 2.0
    pad = float(max(0, min(click_pad, int(half) - 1)))
    max_off = max(1.0, half - pad)

    # Click = bbox_center + offset; so bbox_center = click - offset
    ox = random.uniform(-max_off, max_off)
    oy = random.uniform(-max_off, max_off)
    bbox_cx = click_x - ox
    bbox_cy = click_y - oy
    return bbox_cx, bbox_cy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="dataset_wow_skillbox_64", help="Output dataset directory")
    ap.add_argument("--n", type=int, default=2000, help="Number of valid samples to generate")
    ap.add_argument("--icon", type=int, default=64, help="Icon bbox size (square)")
    ap.add_argument("--crop-min", type=int, default=72, help="Desired min crop size")
    ap.add_argument("--crop-max", type=int, default=96, help="Desired max crop size")
    ap.add_argument("--neighbor-step", type=int, default=74,
                    help="Assumed distance from target center to adjacent box center. 0 means use --icon.")
    ap.add_argument("--neighbor-margin", type=int, default=5, help="Safety margin to keep neighbor centers outside crop")
    ap.add_argument("--exclude-neighbors", action="store_true", help="Reject crops that include neighbor centers (default: on)")
    ap.add_argument("--no-exclude-neighbors", action="store_true", help="Disable neighbor center rejection")
    ap.add_argument("--click-pad", type=int, default=-1,
                    help="Keep click at least this many pixels away from bbox edges. Default: icon//10")
    ap.add_argument("--min-visible-ratio", type=float, default=0.30,
                    help="Minimum (intersection_area / full_bbox_area) to keep as positive label. Otherwise retry.")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed")
    ap.add_argument("--delay", type=float, default=5.0, help="Seconds to wait after click before capture")
    ap.add_argument("--ext", type=str, default="jpg", choices=["jpg", "png"], help="Image extension")
    ap.add_argument("--quality", type=int, default=92, help="JPG quality (ignored for PNG)")
    ap.add_argument("--class-name", type=str, default="skill_icon_box", help="Class name in data.yaml")
    ap.add_argument("--max-tries-per-sample", type=int, default=300, help="Retry limit per sample")
    args = ap.parse_args()

    neighbor_step = args.neighbor_step if args.neighbor_step > 0 else args.icon
    exclude_neighbors = True
    if args.no_exclude_neighbors:
        exclude_neighbors = False
    if args.exclude_neighbors:
        exclude_neighbors = True

    click_pad = args.click_pad if args.click_pad >= 0 else max(1, args.icon // 10)

    cfg = Config(
        out_dir=args.out,
        n_samples=args.n,
        icon_size=args.icon,
        crop_min=args.crop_min,
        crop_max=args.crop_max,
        neighbor_step=neighbor_step,
        neighbor_margin=args.neighbor_margin,
        exclude_neighbors=exclude_neighbors,
        click_pad=click_pad,
        min_visible_ratio=args.min_visible_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        delay_sec=args.delay,
        img_ext=args.ext,
        jpg_quality=args.quality,
        class_name=args.class_name,
        max_tries_per_sample=args.max_tries_per_sample,
    )

    random.seed(cfg.seed)
    ensure_dirs(cfg.out_dir)

    click_x, click_y = wait_for_left_click()
    print(f"[2/3] 将在 {cfg.delay_sec:.2f}s 后开始截图，请切回 WoW 并保持技能栏不动...")
    time.sleep(cfg.delay_sec)

    train_count = int(round(cfg.n_samples * (1.0 - cfg.val_ratio)))
    indices = list(range(cfg.n_samples))
    random.shuffle(indices)
    train_set = set(indices[:train_count])

    with mss.mss() as sct:
        mi, mon = pick_monitor_by_point(sct, click_x, click_y)
        screen_left = mon["left"]
        screen_top = mon["top"]
        screen_w = mon["width"]
        screen_h = mon["height"]
        print(f"[INFO] Auto-selected monitor #{mi}: left={screen_left}, top={screen_top}, w={screen_w}, h={screen_h}")

        ok = 0
        tries = 0

        while ok < cfg.n_samples:
            tries += 1
            if tries > cfg.n_samples * cfg.max_tries_per_sample:
                raise RuntimeError("Too many retries. Relax constraints (e.g. --no-exclude-neighbors or smaller crop).")

            split = "train" if ok in train_set else "val"
            img_dir = os.path.join(cfg.out_dir, "images", split)
            lab_dir = os.path.join(cfg.out_dir, "labels", split)

            # Sample crop size
            crop_w = random.randint(cfg.crop_min, cfg.crop_max)
            crop_h = random.randint(cfg.crop_min, cfg.crop_max)

            # Sample bbox center so click lies within bbox (not necessarily center)
            bbox_cx, bbox_cy = sample_bbox_center_from_click(click_x, click_y, cfg.icon_size, cfg.click_pad)
            half_icon = cfg.icon_size / 2.0

            # Bbox in absolute coords
            bbox_x0 = bbox_cx - half_icon
            bbox_y0 = bbox_cy - half_icon
            bbox_x1 = bbox_cx + half_icon
            bbox_y1 = bbox_cy + half_icon

            # Sample crop position so click lies ANYWHERE inside crop, but crop stays on screen
            min_left = max(screen_left, click_x - crop_w + 1)
            max_left = min(screen_left + screen_w - crop_w, click_x)
            min_top = max(screen_top, click_y - crop_h + 1)
            max_top = min(screen_top + screen_h - crop_h, click_y)

            if min_left > max_left or min_top > max_top:
                continue

            crop_left = random.randint(int(min_left), int(max_left))
            crop_top = random.randint(int(min_top), int(max_top))

            # Optional: reject crops that include neighbor centers
            if cfg.exclude_neighbors:
                neighbors = [
                    (bbox_cx + cfg.neighbor_step, bbox_cy),
                    (bbox_cx - cfg.neighbor_step, bbox_cy),
                    (bbox_cx, bbox_cy + cfg.neighbor_step),
                    (bbox_cx, bbox_cy - cfg.neighbor_step),
                ]
                bad = False
                for nx, ny in neighbors:
                    if point_in_crop(nx, ny, crop_left, crop_top, crop_w, crop_h, margin=cfg.neighbor_margin):
                        bad = True
                        break
                if bad:
                    continue

            # Capture crop region
            monitor = {
                "left": int(crop_left),
                "top": int(crop_top),
                "width": int(crop_w),
                "height": int(crop_h),
            }
            shot = sct.grab(monitor)
            img = Image.frombytes("RGB", shot.size, shot.rgb)

            # Compute bbox ∩ crop intersection in absolute coords
            crop_x0 = crop_left
            crop_y0 = crop_top
            crop_x1 = crop_left + crop_w
            crop_y1 = crop_top + crop_h

            inter = rect_intersection(bbox_x0, bbox_y0, bbox_x1, bbox_y1, crop_x0, crop_y0, crop_x1, crop_y1)
            if inter is None:
                continue

            ix0, iy0, ix1, iy1 = inter
            inter_area = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            full_area = float(cfg.icon_size * cfg.icon_size)
            if inter_area / full_area < cfg.min_visible_ratio:
                continue

            # Convert intersection rect to crop-local coords
            lx0 = ix0 - crop_left
            ly0 = iy0 - crop_top
            lx1 = ix1 - crop_left
            ly1 = iy1 - crop_top

            # Normalize YOLO coords
            bw = (lx1 - lx0)
            bh = (ly1 - ly0)
            cx = (lx0 + lx1) / 2.0
            cy = (ly0 + ly1) / 2.0

            x_center_n = cx / crop_w
            y_center_n = cy / crop_h
            w_n = bw / crop_w
            h_n = bh / crop_h

            # Save image + label
            fname = f"{ok:06d}.{cfg.img_ext}"
            img_path = os.path.join(img_dir, fname)
            lab_path = os.path.join(lab_dir, f"{ok:06d}.txt")

            if cfg.img_ext.lower() == "jpg":
                img.save(img_path, quality=cfg.jpg_quality)
            else:
                img.save(img_path)

            write_yolo_label(lab_path, 0, x_center_n, y_center_n, w_n, h_n)

            ok += 1
            if ok % 100 == 0:
                print(f"[OK] generated: {ok}/{cfg.n_samples}")

    save_data_yaml(cfg.out_dir, cfg.class_name)
    print(f"[DONE] Dataset saved to: {os.path.abspath(cfg.out_dir)}")


if __name__ == "__main__":
    main()
