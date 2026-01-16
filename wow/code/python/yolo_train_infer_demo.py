# -*- coding: utf-8 -*-
"""
YOLOv8 training + inference + live screenshot demo (single file)

Usage examples:
  1) Train:
     python yolo8_wow_icon_train_infer_demo.py train --data dataset_wow_skillbox/data.yaml --epochs 50 --imgsz 320

  2) Infer an image:
     python yolo8_wow_icon_train_infer_demo.py infer --weights runs/detect/train/weights/best.pt --source test.png --save out.png

  3) Demo (click center, capture 200x200, detect, save annotated):
     python yolo8_wow_icon_train_infer_demo.py demo --weights runs/detect/train/weights/best.pt --crop 200 --out demo_annotated.png
"""

import os
import sys
import time
import argparse
from typing import Tuple, Optional, List

import cv2
import numpy as np
import mss
from pynput import mouse

# Ultralytics YOLOv8
from ultralytics import YOLO

print("SCRIPT PATH:", os.path.abspath(__file__))
DEFAULT_DATA_YAML = "dataset_wow_skillbox_64/data.yaml"
# ------------------------------
# Utils: mouse click capture
# ------------------------------
def wait_for_left_click() -> Tuple[int, int]:
    """Wait for a single left click and return absolute screen coordinates (x, y)."""
    clicked = {"pos": None}

    def on_click(x, y, button, pressed):
        if pressed and button == mouse.Button.left:
            clicked["pos"] = (int(x), int(y))
            return False  # stop listener

    print("[DEMO] 请把鼠标移到目标区域中心附近，左键点击一次...")
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()

    if clicked["pos"] is None:
        raise RuntimeError("No click captured.")
    print(f"[DEMO] 捕获点击坐标: {clicked['pos']}")
    return clicked["pos"]


# ------------------------------
# Utils: screen capture
# ------------------------------
def capture_center_crop(center_xy: Tuple[int, int], crop_size: int, monitor_index: int = 1) -> np.ndarray:
    """
    Capture a square crop_size x crop_size region centered at center_xy.
    Returns BGR image (OpenCV format).
    """
    cx, cy = center_xy
    half = crop_size // 2

    with mss.mss() as sct:
        mon = sct.monitors[monitor_index]  # primary monitor by default
        left0 = mon["left"]
        top0 = mon["top"]
        sw = mon["width"]
        sh = mon["height"]

        # Convert to absolute rect and clamp to screen
        left = cx - half
        top = cy - half
        left = max(left0, min(left, left0 + sw - crop_size))
        top = max(top0, min(top, top0 + sh - crop_size))

        rect = {"left": int(left), "top": int(top), "width": int(crop_size), "height": int(crop_size)}
        shot = sct.grab(rect)

        # shot.rgb is RGB; convert to BGR for OpenCV
        img = np.array(shot)[:, :, :3]  # BGRA -> BGR (mss returns BGRA)
        # mss numpy array is BGRA by default, so [:,:,:3] is BGR already
        return img


# ------------------------------
# Utils: draw boxes
# ------------------------------
def draw_detections(
    img_bgr: np.ndarray,
    dets: List[Tuple[float, float, float, float, float, int]],
    class_names: Optional[dict] = None,
    conf_thres: float = 0.25
) -> np.ndarray:
    """
    Draw detections on image.
    det tuple: (x1, y1, x2, y2, conf, cls_id)
    """
    out = img_bgr.copy()

    for (x1, y1, x2, y2, conf, cls_id) in dets:
        if conf < conf_thres:
            continue
        x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)

        name = str(cls_id)
        if class_names and cls_id in class_names:
            name = str(class_names[cls_id])

        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1i, y1i - th - 6), (x1i + tw + 6, y1i), (0, 255, 0), -1)
        cv2.putText(out, label, (x1i + 3, y1i - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return out


def yolo_predict(model: YOLO, img_bgr: np.ndarray) -> List[Tuple[float, float, float, float, float, int]]:
    """
    Run YOLO prediction on a BGR image and return detections.
    """
    # Ultralytics accepts numpy arrays; it assumes BGR or RGB both work in most cases,
    # but we keep BGR and let it handle.
    results = model.predict(source=img_bgr, verbose=False)
    r = results[0]

    dets = []
    if r.boxes is None or len(r.boxes) == 0:
        return dets

    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        dets.append((x1, y1, x2, y2, float(conf[i]), int(cls[i])))
    return dets


# ------------------------------
# Commands: train / infer / demo
# ------------------------------
# def make_stop_on_map_callback(map50_thres=0.99, map_thres=0.65):
#     """
#     Stop training early once reaching target metrics.
#     Compatible with Ultralytics YOLOv8 stable versions.
#     """
#     def on_fit_epoch_end(trainer):
#         # metrics access (robust)
#         metrics = trainer.metrics
#         if not hasattr(metrics, "box"):
#             return

#         map50 = float(getattr(metrics.box, "map50", 0.0))
#         map5095 = float(getattr(metrics.box, "map", 0.0))

#         if map50 >= map50_thres and map5095 >= map_thres:
#             print(
#                 f"\n[EARLY STOP] Target reached: "
#                 f"mAP50={map50:.4f}, mAP50-95={map5095:.4f}"
#             )
#             trainer.stop = True

#     return {
#         "on_fit_epoch_end": on_fit_epoch_end
#     }

def cmd_train(args):
    os.makedirs(args.project, exist_ok=True)
    model = YOLO(args.model)

    # callbacks = make_stop_on_map_callback(
    #     map50_thres=args.stop_map50,
    #     map_thres=args.stop_map,
    # )

    results =model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        # callbacks=callbacks,
        hsv_h=0.02,   # 色相轻微变化（UI不要太大）
        hsv_s=0.4,    # 饱和度
        hsv_v=0.4,    # 亮度
        degrees=2.0,  # 轻微旋转
        translate=0.05,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,   # 技能栏一般不需要左右翻转
        flipud=0.0,   # 更不需要上下翻转
        mosaic=0.2,   # UI检测一般 mosaic 不用太大
        mixup=0.0,    # UI类一般不建议 mixup
        erasing=0.2,  # 随机擦除，模拟遮挡/高亮/蒙层
    )

    print("\n[TRAIN] 完成。")
    save_dir = results.save_dir  # 真实保存目录（Path 对象）
    best_pt = os.path.join(save_dir, "weights", "best.pt")

    print("[TRAIN] best.pt =", best_pt)
    print("\n[TRAIN] 完成，开始导出 ONNX…")
    
    model = YOLO(best_pt)

    model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=12,
        simplify=True,
        dynamic=False,  # 技能栏强烈建议 False
    )

    print("[EXPORT] ONNX 导出完成")


def cmd_infer(args):
    """
    Inference on a single image and save annotated image.
    """
    model = YOLO(args.weights)
    img = cv2.imread(args.source, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {args.source}")

    dets = yolo_predict(model, img)
    class_names = getattr(model, "names", None)
    if isinstance(class_names, list):
        class_names = {i: n for i, n in enumerate(class_names)}

    annotated = draw_detections(img, dets, class_names=class_names, conf_thres=args.conf)

    out_path = args.save if args.save else "annotated.png"
    cv2.imwrite(out_path, annotated)
    print(f"[INFER] 保存标注图: {os.path.abspath(out_path)}")
    print(f"[INFER] 检测到 {sum(1 for d in dets if d[4] >= args.conf)} 个框 (conf>={args.conf})")


def cmd_demo(args):
    """
    Demo: click a point, capture center crop, run detection, save annotated.
    """
    model = YOLO(args.weights)

    # Capture click
    cx, cy = wait_for_left_click()
    print(f"[DEMO] {args.delay:.2f}s 后开始截图（请切回 WoW 窗口）...")
    time.sleep(args.delay)

    # Capture crop
    img = capture_center_crop((cx, cy), args.crop, monitor_index=args.monitor)

    # Predict
    dets = yolo_predict(model, img)
    class_names = getattr(model, "names", None)
    if isinstance(class_names, list):
        class_names = {i: n for i, n in enumerate(class_names)}

    annotated = draw_detections(img, dets, class_names=class_names, conf_thres=args.conf)

    # Save
    cv2.imwrite(args.out, annotated)
    print(f"[DEMO] 保存标注图: {os.path.abspath(args.out)}")
    print(f"[DEMO] 检测到 {sum(1 for d in dets if d[4] >= args.conf)} 个框 (conf>={args.conf})")


def build_argparser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    # train
    pt = sub.add_parser("train")
    pt.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="Path to data.yaml from generated dataset")
    pt.add_argument("--model", type=str, default="yolov8n.pt", help="Base model (yolov8n.pt/yolov8s.pt...)")
    pt.add_argument("--epochs", type=int, default=50)
    pt.add_argument("--imgsz", type=int, default=110)
    pt.add_argument("--batch", type=int, default=64)
    pt.add_argument("--device", type=str, default="", help="'' for auto, or '0', 'cpu'")
    pt.add_argument("--project", type=str, default="skillbox",help="Ultralytics project directory")
    pt.add_argument("--name", type=str, default="train_wow_icon_box")
    pt.add_argument("--workers", type=int, default=8)
    pt.add_argument("--patience", type=int, default=5)
    pt.add_argument("--close_mosaic", type=int, default=10, help="Disable mosaic in last N epochs")
    pt.add_argument(
        "--stop-map50",
        dest="stop_map50",
        type=float,
        default=0.99,
        help="Early stop when mAP50 >= this value"
    )
    pt.add_argument(
        "--stop-map",
        dest="stop_map",
        type=float,
        default=0.62,
        help="Early stop when mAP50-95 >= this value"
    )

    # infer
    pi = sub.add_parser("infer")
    pi.add_argument("--weights", type=str, default=r"skillbox\train_wow_icon_box\weights\best.pt", help="best.pt path")
    pi.add_argument("--source", type=str, required=True, help="Input image path")
    pi.add_argument("--save", type=str, default="annotated.png", help="Output annotated image path")
    pi.add_argument("--conf", type=float, default=0.25)

    # demo
    pd = sub.add_parser("demo")
    pd.add_argument("--weights", type=str, default=r"skillbox\train_wow_icon_box\weights\best.pt", help="best.pt path")
    pd.add_argument("--crop", type=int, default=100, help="Crop size (square)")
    pd.add_argument("--out", type=str, default="demo_annotated.png")
    pd.add_argument("--conf", type=float, default=0.25)
    pd.add_argument("--delay", type=float, default=5, help="Delay after click before capture")
    pd.add_argument("--monitor", type=int, default=1, help="mss monitor index (1=primary)")

    return p

def ensure_default_subcommand(argv, subcommands, default="train"):
    """
    If argv has no subcommand, insert default 'train'.
    """
    for a in argv[1:]:
        if a in subcommands:
            return
    argv.insert(1, default)
    
def main():

    parser = build_argparser()
    
    # If no subcommand is provided, default to "train"
    ensure_default_subcommand(sys.argv, {"train", "infer", "demo"}, "train")

    args = parser.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "infer":
        cmd_infer(args)
    elif args.cmd == "demo":
        cmd_demo(args)
    else:
        raise RuntimeError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
