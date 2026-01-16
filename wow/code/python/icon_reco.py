# -*- coding: utf-8 -*-
"""
MobileNetV3-based WoW spell icon recognizer.

This script provides:
1) Training a MobileNetV3 classifier from your icon library (data/classes, data/race).
2) Inference on a full screenshot (not necessarily 64x64):
   - Multi-scale sliding window proposals
   - Classify each window
   - Keep high-confidence results
   - Apply NMS to remove duplicates
   - Output SpellID with bbox and score

Notes:
- This is a generic solution when UI coordinates are unknown.
- For best speed/accuracy, later you can restrict scan regions to the action bars.

Code comments are in English by your preference.
"""

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_spellid_from_filename(name: str) -> int:
    """
    Extract SpellID from filenames like:
    12345_SpellName_IconFileDataID_ClassName.png
    """
    m = re.match(r"^\s*(\d+)_", name)
    if not m:
        return -1
    return int(m.group(1))


def list_icon_images(data_root: Path) -> List[Path]:
    """
    Collect png images from data_root/classes and data_root/race.
    """
    paths: List[Path] = []
    for sub in ["classes", "class", "race"]:
        p = data_root / sub
        if p.exists() and p.is_dir():
            paths.extend(sorted(p.rglob("*.png")))
    return paths


@dataclass
class Sample:
    path: Path
    spell_id: int


def build_label_map(samples: List[Sample]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Map SpellID -> class_index, and reverse map class_index -> SpellID.
    """
    spell_ids = sorted({s.spell_id for s in samples if s.spell_id > 0})
    sid2idx = {sid: i for i, sid in enumerate(spell_ids)}
    idx2sid = {i: sid for sid, i in sid2idx.items()}
    return sid2idx, idx2sid


# ----------------------------
# Dataset
# ----------------------------

class IconDataset(Dataset):
    def __init__(self, samples: List[Sample], sid2idx: Dict[int, int], train: bool):
        self.samples = samples
        self.sid2idx = sid2idx

        # Data augmentation is important to generalize from clean icons to screenshot crops.
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(64, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02),
                transforms.RandomGrayscale(p=0.15),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = Image.open(s.path).convert("RGB")
        x = self.tf(img)
        y = self.sid2idx[s.spell_id]
        return x, y


def split_samples(samples: List[Sample], val_ratio: float = 0.08) -> Tuple[List[Sample], List[Sample]]:
    """
    Split by SpellID to avoid leakage if you have multiple variants per SpellID.
    """
    sid_to_items: Dict[int, List[Sample]] = {}
    for s in samples:
        sid_to_items.setdefault(s.spell_id, []).append(s)

    spell_ids = list(sid_to_items.keys())
    random.shuffle(spell_ids)

    val_count = max(1, int(len(spell_ids) * val_ratio))
    val_sids = set(spell_ids[:val_count])

    train_list: List[Sample] = []
    val_list: List[Sample] = []
    for sid, items in sid_to_items.items():
        if sid in val_sids:
            val_list.extend(items)
        else:
            train_list.extend(items)

    return train_list, val_list


# ----------------------------
# Model
# ----------------------------

def create_model(num_classes: int) -> nn.Module:
    """
    Create a MobileNetV3-small classifier.
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    # Replace classifier head
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


# ----------------------------
# Training
# ----------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        total += y.size(0)

        # Top-1
        pred1 = torch.argmax(logits, dim=1)
        correct1 += (pred1 == y).sum().item()

        # Top-5
        top5 = torch.topk(logits, k=min(5, logits.size(1)), dim=1).indices
        correct5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()

    return correct1 / max(1, total), correct5 / max(1, total)


def train_main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect samples
    imgs = list_icon_images(data_root)
    samples: List[Sample] = []
    for p in imgs:
        sid = parse_spellid_from_filename(p.name)
        if sid > 0:
            samples.append(Sample(path=p, spell_id=sid))

    if len(samples) < 50:
        raise RuntimeError(f"Not enough samples found under {data_root}. Found {len(samples)} pngs.")

    train_samples, val_samples = split_samples(samples, val_ratio=args.val_ratio)
    sid2idx, idx2sid = build_label_map(samples)

    # Save label maps
    (out_dir / "sid2idx.json").write_text(json.dumps(sid2idx, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "idx2sid.json").write_text(json.dumps(idx2sid, ensure_ascii=False, indent=2), encoding="utf-8")

    num_classes = len(sid2idx)
    print(f"[INFO] Samples: train={len(train_samples)} val={len(val_samples)} classes={num_classes}")

    ds_train = IconDataset(train_samples, sid2idx, train=True)
    ds_val = IconDataset(val_samples, sid2idx, train=False)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = create_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_top1 = 0.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in dl_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        scheduler.step()

        top1, top5 = evaluate(model, dl_val, device)
        avg_loss = running_loss / max(1, len(ds_train))

        print(f"[EPOCH {epoch:03d}] loss={avg_loss:.4f} val_top1={top1:.4f} val_top5={top5:.4f}")

        if top1 > best_top1:
            best_top1 = top1
            torch.save({
                "model": model.state_dict(),
                "num_classes": num_classes,
                "sid2idx": sid2idx,
                "idx2sid": idx2sid,
            }, best_path)
            print(f"[OK] Saved best checkpoint: {best_path} (top1={best_top1:.4f})")

    print("[DONE] Training completed.")
    print(f"Best checkpoint: {best_path}")


# ----------------------------
# Screenshot inference: sliding window + NMS
# ----------------------------

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """
    Standard NMS for axis-aligned boxes.
    boxes: [N, 4] -> x1,y1,x2,y2
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0].astype(np.float32)
    y1 = boxes[:, 1].astype(np.float32)
    x2 = boxes[:, 2].astype(np.float32)
    y2 = boxes[:, 3].astype(np.float32)

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]

    return keep


def load_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[int, int], Dict[int, int]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = ckpt["num_classes"]
    sid2idx = ckpt["sid2idx"]
    idx2sid = ckpt["idx2sid"]

    model = create_model(num_classes).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, sid2idx, idx2sid


def infer_main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt_path = Path(args.ckpt)

    model, _, idx2sid = load_checkpoint(ckpt_path, device)

    img = Image.open(args.image).convert("RGB")
    w, h = img.size

    # Preprocess for each crop
    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Multi-scale sliding window settings
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    stride = args.stride

    # Optional: restrict scanning region (x1,y1,x2,y2) to speed up
    roi = None
    if args.roi:
        parts = [int(x) for x in args.roi.split(",")]
        if len(parts) == 4:
            roi = (max(0, parts[0]), max(0, parts[1]), min(w, parts[2]), min(h, parts[3]))

    candidates: List[Tuple[int, int, int, int, float, int]] = []
    batch_imgs: List[torch.Tensor] = []
    batch_meta: List[Tuple[int, int, int, int]] = []

    def flush_batch():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        conf_np = conf.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()

        for (x1, y1, x2, y2), c, p in zip(batch_meta, conf_np, pred_np):
            if c >= args.conf:
                spell_id = int(idx2sid[int(p)])
                candidates.append((x1, y1, x2, y2, float(c), spell_id))

        batch_imgs.clear()
        batch_meta.clear()

    # Scan
    x_start, y_start, x_end, y_end = (0, 0, w, h) if roi is None else roi
    img_np = np.array(img)

    for sz in sizes:
        for yy in range(y_start, max(y_start, y_end - sz + 1), stride):
            for xx in range(x_start, max(x_start, x_end - sz + 1), stride):
                crop = img_np[yy:yy + sz, xx:xx + sz, :]
                if crop.shape[0] != sz or crop.shape[1] != sz:
                    continue

                crop_pil = Image.fromarray(crop)
                x_t = tf(crop_pil)

                batch_imgs.append(x_t)
                batch_meta.append((xx, yy, xx + sz, yy + sz))

                if len(batch_imgs) >= args.batch:
                    flush_batch()

    flush_batch()

    if not candidates:
        print("[INFO] No icon candidates found above confidence threshold.")
        return

    # Apply NMS per SpellID (or global). Here we do global NMS for simplicity.
    boxes = np.array([[c[0], c[1], c[2], c[3]] for c in candidates], dtype=np.float32)
    scores = np.array([c[4] for c in candidates], dtype=np.float32)
    keep = nms(boxes, scores, args.iou)

    kept = [candidates[i] for i in keep]
    kept.sort(key=lambda x: -x[4])

    # Print results
    print(f"[OK] Found {len(kept)} icons after NMS (from {len(candidates)} raw candidates).")
    for x1, y1, x2, y2, conf, sid in kept[: args.topk]:
        print(f"SpellID={sid} conf={conf:.3f} bbox=({x1},{y1},{x2},{y2})")

    # Save JSON
    if args.out_json:
        out = []
        for x1, y1, x2, y2, conf, sid in kept:
            out.append({
                "spell_id": sid,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            })
        Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Wrote: {args.out_json}")


# ----------------------------
# CLI
# ----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    t = sub.add_parser("train")
    t.add_argument("--data_root", type=str, default="data")
    t.add_argument("--out_dir", type=str, default="runs/mnv3")
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch_size", type=int, default=128)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--val_ratio", type=float, default=0.08)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--cpu", action="store_true")

    # infer
    inf = sub.add_parser("infer")
    inf.add_argument("--ckpt", type=str, required=True)
    inf.add_argument("--image", type=str, required=True)
    inf.add_argument("--sizes", type=str, default="32,36,40,48,56,64")
    inf.add_argument("--stride", type=int, default=8)
    inf.add_argument("--conf", type=float, default=0.92)
    inf.add_argument("--iou", type=float, default=0.35)
    inf.add_argument("--batch", type=int, default=256)
    inf.add_argument("--topk", type=int, default=50)
    inf.add_argument("--roi", type=str, default="", help="Optional ROI: x1,y1,x2,y2")
    inf.add_argument("--out_json", type=str, default="")
    inf.add_argument("--cpu", action="store_true")

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.cmd == "train":
        train_main(args)
    elif args.cmd == "infer":
        infer_main(args)
    else:
        raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
