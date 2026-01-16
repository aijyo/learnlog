# -*- coding: utf-8 -*-
"""
YOLOv8 dataset generator for WoW skill icon box (icon=64 default)

Features:
- Auto-select monitor by click position (multi-monitor safe).
- Crop is ALWAYS centered at click point.
- Allow partial bbox truncation: label uses intersection (bbox ∩ crop).
- Avoid capturing neighboring box centers (assume neighbor centers at +/- neighbor_step on x/y).
- Auto-clamp crop size by:
  (1) neighbor constraint  (2) screen edge constraint around click
- Guarantee ok == N by retrying.

Output structure:
  out_dir/
    images/train/*.jpg|png
    images/val/*.jpg|png
    labels/train/*.txt
    labels/val/*.txt
    data.yaml
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
    bbox_center_jitter: int         # bbox center jitter around click (pixels)
    crop_min: int                   # desired min crop size
    crop_max: int                   # desired max crop size
    neighbor_step: int              # assumed neighbor center distance (pixels)
    neighbor_margin: int            # safety margin (pixels)
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

    print("\n[1/3] 请把鼠标移到目标技能格附近（尽量靠近中心），然后左键点击一次...")
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
    """
    Pick the monitor that contains the point (x,y).
    Returns (monitor_index, monitor_dict). If none found, fallback to primary.
    """
    for mi in range(1, len(sct.monitors)):
        m = sct.monitors[mi]
        if (m["left"] <= x < m["left"] + m["width"] and
                m["top"] <= y < m["top"] + m["height"]):
            return mi, m
    return 1, sct.monitors[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="dataset_wow_skillbox_64", help="Output dataset directory")
    ap.add_argument("--n", type=int, default=2000, help="Number of valid samples to generate")
    ap.add_argument("--icon", type=int, default=64, help="Icon bbox size (square)")
    ap.add_argument("--bbox-jitter", type=int, default=6,
                    help="Jitter of bbox center around click (pixels). Crop center stays at click.")
    ap.add_argument("--crop-min", type=int, default=72, help="Desired min crop size (will be auto-clamped)")
    ap.add_argument("--crop-max", type=int, default=100, help="Desired max crop size (will be auto-clamped)")
    ap.add_argument("--neighbor-step", type=int, default=74,
                    help="Assumed distance from target center to adjacent box center. 0 means use --icon.")
    ap.add_argument("--neighbor-margin", type=int, default=5, help="Safety margin to keep neighbor centers outside crop")
    ap.add_argument("--min-visible-ratio", type=float, default=0.30,
                    help="Minimum (intersection_area / full_bbox_area) to keep as positive label. Otherwise retry.")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed")
    ap.add_argument("--delay", type=float, default=5.0, help="Seconds to wait after click before capture")
    ap.add_argument("--ext", type=str, default="jpg", choices=["jpg", "png"], help="Image extension")
    ap.add_argument("--quality", type=int, default=92, help="JPG quality (ignored for PNG)")
    ap.add_argument("--class-name", type=str, default="skill_icon_box", help="Class name in data.yaml")
    ap.add_argument("--max-tries-per-sample", type=int, default=200, help="Retry limit per sample")
    args = ap.parse_args()

    neighbor_step = args.neighbor_step if args.neighbor_step > 0 else args.icon

    cfg = Config(
        out_dir=args.out,
        n_samples=args.n,
        icon_size=args.icon,
        bbox_center_jitter=args.bbox_jitter,
        crop_min=args.crop_min,
        crop_max=args.crop_max,
        neighbor_step=neighbor_step,
        neighbor_margin=args.neighbor_margin,
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

    # Step 1: click point (crop center)
    click_x, click_y = wait_for_left_click()
    print(f"[2/3] 将在 {cfg.delay_sec:.2f}s 后开始截图，请切回 WoW 并保持技能栏不动...")
    time.sleep(cfg.delay_sec)

    # Split train/val
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

        # Neighbor constraint: crop_size < 2*(neighbor_step - margin)
        max_by_neighbor = max(1, 2 * (cfg.neighbor_step - cfg.neighbor_margin) - 1)

        # Edge constraint around click (centered crop must fit)
        dist_left = click_x - screen_left
        dist_right = (screen_left + screen_w) - click_x
        dist_top = click_y - screen_top
        dist_bottom = (screen_top + screen_h) - click_y
        max_w_by_edge = int(2 * min(dist_left, dist_right))
        max_h_by_edge = int(2 * min(dist_top, dist_bottom))

        # Final maximum crop sizes
        hard_max_w = min(cfg.crop_max, max_by_neighbor, max_w_by_edge)
        hard_max_h = min(cfg.crop_max, max_by_neighbor, max_h_by_edge)

        if hard_max_w < 2 or hard_max_h < 2:
            raise RuntimeError(
                "Click point is too close to screen edge (cannot make centered crop). "
                "Please click more towards the screen center."
            )

        # Minimum crop sizes cannot exceed hard maximum
        hard_min_w = min(max(2, cfg.crop_min), hard_max_w)
        hard_min_h = min(max(2, cfg.crop_min), hard_max_h)

        print(f"[INFO] crop_w range: {hard_min_w}..{hard_max_w}")
        print(f"[INFO] crop_h range: {hard_min_h}..{hard_max_h}")
        print(f"[INFO] neighbor_step={cfg.neighbor_step}, margin={cfg.neighbor_margin}, max_by_neighbor={max_by_neighbor}")
        print(f"[INFO] bbox(icon)={cfg.icon_size}, bbox_center_jitter={cfg.bbox_center_jitter}, min_visible_ratio={cfg.min_visible_ratio}")

        ok = 0
        total_tries = 0

        print("[3/3] 开始生成数据集（centered crop + partial bbox + avoid neighbor centers）...")
        for i in range(cfg.n_samples):
            split = "train" if i in train_set else "val"
            img_dir = os.path.join(cfg.out_dir, "images", split)
            lbl_dir = os.path.join(cfg.out_dir, "labels", split)
            base = f"{split}_{i:06d}"
            img_path = os.path.join(img_dir, f"{base}.{cfg.img_ext}")
            lbl_path = os.path.join(lbl_dir, f"{base}.txt")

            got = False
            for _ in range(cfg.max_tries_per_sample):
                total_tries += 1

                # Sample crop size (center is fixed at click)
                crop_w = random.randint(hard_min_w, hard_max_w)
                crop_h = random.randint(hard_min_h, hard_max_h)
                half_w = crop_w / 2.0
                half_h = crop_h / 2.0

                crop_left = click_x - half_w
                crop_top = click_y - half_h
                crop_right = crop_left + crop_w
                crop_bottom = crop_top + crop_h

                # (Optional) extra sanity check; should always pass due to edge constraint
                if crop_left < screen_left or crop_top < screen_top:
                    continue
                if crop_right > screen_left + screen_w or crop_bottom > screen_top + screen_h:
                    continue

                # BBox center jitter around click (simulate not perfectly centered click)
                jx = random.randint(-cfg.bbox_center_jitter, cfg.bbox_center_jitter)
                jy = random.randint(-cfg.bbox_center_jitter, cfg.bbox_center_jitter)
                bbox_cx = click_x + jx
                bbox_cy = click_y + jy

                # Full bbox rect (absolute)
                bw = float(cfg.icon_size)
                bh = float(cfg.icon_size)
                bbox_x0 = bbox_cx - bw / 2.0
                bbox_y0 = bbox_cy - bh / 2.0
                bbox_x1 = bbox_x0 + bw
                bbox_y1 = bbox_y0 + bh

                # Crop rect (absolute)
                cx0, cy0, cx1, cy1 = crop_left, crop_top, crop_right, crop_bottom

                # Intersection (visible bbox part)
                inter = rect_intersection(bbox_x0, bbox_y0, bbox_x1, bbox_y1, cx0, cy0, cx1, cy1)
                if inter is None:
                    continue

                ix0, iy0, ix1, iy1 = inter
                inter_area = (ix1 - ix0) * (iy1 - iy0)
                full_area = bw * bh
                if inter_area / full_area < cfg.min_visible_ratio:
                    continue

                # Grab the crop
                grab_rect = {
                    "left": int(round(crop_left)),
                    "top": int(round(crop_top)),
                    "width": int(crop_w),
                    "height": int(crop_h),
                }
                shot = sct.grab(grab_rect)
                img = Image.frombytes("RGB", shot.size, shot.rgb)

                # Convert inter bbox to crop-local coords
                lx0 = ix0 - crop_left
                ly0 = iy0 - crop_top
                lx1 = ix1 - crop_left
                ly1 = iy1 - crop_top

                # Clamp to crop bounds
                lx0 = clamp(lx0, 0.0, crop_w)
                ly0 = clamp(ly0, 0.0, crop_h)
                lx1 = clamp(lx1, 0.0, crop_w)
                ly1 = clamp(ly1, 0.0, crop_h)

                lw = max(0.0, lx1 - lx0)
                lh = max(0.0, ly1 - ly0)
                lcx = lx0 + lw / 2.0
                lcy = ly0 + lh / 2.0

                x_center_n = lcx / float(crop_w)
                y_center_n = lcy / float(crop_h)
                w_n = lw / float(crop_w)
                h_n = lh / float(crop_h)

                # Save image
                if cfg.img_ext.lower() == "jpg":
                    img.save(img_path, quality=cfg.jpg_quality)
                else:
                    img.save(img_path)

                # Save label
                write_yolo_label(lbl_path, 0, x_center_n, y_center_n, w_n, h_n)

                got = True
                ok += 1
                break

            if not got:
                raise RuntimeError(
                    f"Failed to generate a valid sample at index {i}. "
                    f"Try lowering --min-visible-ratio, reducing --bbox-jitter, "
                    f"or increasing --max-tries-per-sample."
                )

            if (i + 1) % 200 == 0:
                print(f"  ok: {i+1}/{cfg.n_samples} (total_tries={total_tries}, avg={total_tries / ok:.2f} tries/sample)")

    save_data_yaml(cfg.out_dir, cfg.class_name)
    print(f"\n[DONE] 输出目录: {os.path.abspath(cfg.out_dir)}")
    print(f"       有效样本: {ok}/{cfg.n_samples}")
    print(f"       总尝试次数: {total_tries} (平均每张 {total_tries / max(1, ok):.2f} 次尝试)")

    print("\n训练命令示例：")
    print(f"  yolo detect train data={os.path.join(cfg.out_dir, 'data.yaml')} model=yolov8n.pt imgsz=320 epochs=50")


if __name__ == "__main__":
    main()
