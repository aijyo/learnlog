# train_icon_mobilenetv3_multihead.py
# English comments only.

import os
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image, ImageEnhance, ImageDraw

STATE_NAMES = ["available", "cooldown", "highlight", "unusable"]
STATE_AVAILABLE = 0
STATE_COOLDOWN = 1
STATE_HIGHLIGHT = 2
STATE_UNUSABLE = 3


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv_rows(csv_path: Path) -> List[Tuple[int, str]]:
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            spell_id = int(r["spell_id"])
            path = r["path"]
            rows.append((spell_id, path))
    return rows


def build_label_map(rows: List[Tuple[int, str]]) -> Dict[int, int]:
    # Map SpellID to contiguous class index
    spell_ids = sorted({sid for sid, _ in rows})
    return {sid: i for i, sid in enumerate(spell_ids)}


def apply_synthetic_state(img: Image.Image) -> Tuple[Image.Image, int]:
    # Synthesize different UI states from an "available" icon.
    # Returns (augmented_image, state_label).
    state = random.randint(0, 3)

    if state == STATE_AVAILABLE:
        # Slight jitter to prevent overfitting to pristine icons
        if random.random() < 0.6:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
        if random.random() < 0.6:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
        return img, state

    if state == STATE_COOLDOWN:
        # Darken + desaturate + add radial/pie overlay
        img = ImageEnhance.Color(img).enhance(random.uniform(0.2, 0.6))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.4, 0.75))

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size
        # Pie mask ratio (0~1)
        ratio = random.uniform(0.05, 0.95)
        angle = int(360 * ratio)
        bbox = [0, 0, w, h]
        # Semi-transparent black pie sector
        draw.pieslice(bbox, start=270, end=270 + angle, fill=(0, 0, 0, random.randint(80, 140)))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        return img, state

    if state == STATE_HIGHLIGHT:
        # Brighten + add subtle glow border
        img = ImageEnhance.Brightness(img).enhance(random.uniform(1.1, 1.5))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(1.0, 1.3))

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size
        # Draw a soft border (yellow-ish) without hardcoding exact UI colors
        # We use a light warm tone; you can replace with exact later.
        for t in range(1, 4):
            alpha = 40 - t * 8
            draw.rounded_rectangle(
                [t, t, w - t - 1, h - t - 1],
                radius=6,
                outline=(255, 230, 120, max(0, alpha)),
                width=2
            )
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        return img, state

    # STATE_UNUSABLE
    # Make it look "disabled": desaturate and darken
    img = ImageEnhance.Color(img).enhance(random.uniform(0.0, 0.25))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.35, 0.65))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.0))
    return img, state


class IconSpellDataset(Dataset):
    def __init__(self, rows: List[Tuple[int, str]], label_map: Dict[int, int], root: Path, train: bool):
        self.rows = rows
        self.label_map = label_map
        self.root = root
        self.train = train

        self.base_tf = transforms.Compose([
            transforms.Resize((96, 96)),  # Use 96 for better detail; 64 also works
        ])

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        spell_id, rel_path = self.rows[idx]
        class_idx = self.label_map[spell_id]

        img_path = (self.root / rel_path).resolve()
        img = Image.open(img_path).convert("RGB")

        # Optional random crop/pad to mimic imperfect crops from detector
        if self.train:
            # Randomly pad then crop back
            if random.random() < 0.6:
                pad = random.randint(0, 8)
                padded = Image.new("RGB", (img.size[0] + pad * 2, img.size[1] + pad * 2), (0, 0, 0))
                padded.paste(img, (pad, pad))
                img = padded
            # Random crop
            if random.random() < 0.7:
                w, h = img.size
                crop_margin = random.randint(0, 6)
                left = random.randint(0, crop_margin)
                top = random.randint(0, crop_margin)
                right = w - random.randint(0, crop_margin)
                bottom = h - random.randint(0, crop_margin)
                img = img.crop((left, top, right, bottom))

        img = self.base_tf(img)

        # Multi-head: synthesize state labels from available icons
        if self.train:
            img, state = apply_synthetic_state(img)
        else:
            # Validation: keep it available; you can also evaluate synthesized states if you want
            state = STATE_AVAILABLE

        x = self.to_tensor(img)
        y_spell = torch.tensor(class_idx, dtype=torch.long)
        y_state = torch.tensor(state, dtype=torch.long)
        return x, y_spell, y_state


class MobileNetV3MultiHead(nn.Module):
    def __init__(self, num_spells: int, num_states: int = 4, pretrained: bool = True):
        super().__init__()
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            backbone = mobilenet_v3_small(weights=weights)
        else:
            backbone = mobilenet_v3_small(weights=None)

        # Use backbone feature extractor
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Determine embedding dim from classifier input
        # MobileNetV3-small classifier starts with Linear(in_features, ...)
        emb_dim = backbone.classifier[0].in_features

        # Replace classifier with identity; we will use our own heads
        self.backbone_out = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(emb_dim, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
        )

        # Head A: SpellID classification
        self.head_spell = nn.Linear(512, num_spells)

        # Head B: State classification
        self.head_state = nn.Linear(512, num_states)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.backbone_out(x)
        logits_spell = self.head_spell(x)
        logits_state = self.head_state(x)
        return logits_spell, logits_state


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    spell_loss_w: float = 1.0
    state_loss_w: float = 0.3  # Keep state loss smaller to avoid harming spell accuracy early


def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    with torch.no_grad():
        topk = logits.topk(k, dim=1).indices
        correct = (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    return correct


def main():
    seed_everything(42)
    project_root = Path(__file__).resolve().parent
    data_root = project_root

    train_csv = project_root / "data" / "classifier" / "train.csv"
    val_csv = project_root / "data" / "classifier" / "val.csv"

    train_rows = read_csv_rows(train_csv)
    val_rows = read_csv_rows(val_csv)

    # Build spell-id mapping from train+val to keep consistent indices
    label_map = build_label_map(train_rows + val_rows)
    num_spells = len(label_map)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_ds = IconSpellDataset(train_rows, label_map, data_root, train=True)
    val_ds = IconSpellDataset(val_rows, label_map, data_root, train=False)

    cfg = TrainConfig()

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = MobileNetV3MultiHead(num_spells=num_spells, num_states=4, pretrained=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    out_dir = project_root / "outputs" / "classifier"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save label map for inference
    label_map_path = out_dir / "spell_label_map.csv"
    with label_map_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["spell_id", "class_index"])
        for sid, idx in sorted(label_map.items(), key=lambda x: x[1]):
            w.writerow([sid, idx])

    best_val_top1 = 0.0

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0

        for x, y_spell, y_state in train_loader:
            x = x.to(device, non_blocking=True)
            y_spell = y_spell.to(device, non_blocking=True)
            y_state = y_state.to(device, non_blocking=True)

            logits_spell, logits_state = model(x)

            loss_spell = F.cross_entropy(logits_spell, y_spell)
            loss_state = F.cross_entropy(logits_state, y_state)

            loss = cfg.spell_loss_w * loss_spell + cfg.state_loss_w * loss_state

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_top1 = 0.0
        val_top5 = 0.0
        val_state_acc = 0.0
        n = 0

        with torch.no_grad():
            for x, y_spell, y_state in val_loader:
                x = x.to(device, non_blocking=True)
                y_spell = y_spell.to(device, non_blocking=True)
                y_state = y_state.to(device, non_blocking=True)

                logits_spell, logits_state = model(x)

                loss_spell = F.cross_entropy(logits_spell, y_spell)
                loss_state = F.cross_entropy(logits_state, y_state)
                loss = cfg.spell_loss_w * loss_spell + cfg.state_loss_w * loss_state

                bs = x.size(0)
                val_loss += loss.item() * bs
                val_top1 += accuracy_topk(logits_spell, y_spell, k=1) * bs
                val_top5 += accuracy_topk(logits_spell, y_spell, k=5) * bs
                val_state_acc += accuracy_topk(logits_state, y_state, k=1) * bs
                n += bs

        val_loss /= max(1, n)
        val_top1 /= max(1, n)
        val_top5 /= max(1, n)
        val_state_acc /= max(1, n)

        print(f"[Epoch {epoch+1}/{cfg.epochs}] "
              f"train_loss={total_loss/max(1,len(train_ds)):.4f} "
              f"val_loss={val_loss:.4f} "
              f"spell_top1={val_top1:.4f} spell_top5={val_top5:.4f} "
              f"state_acc={val_state_acc:.4f}")

        # Save best
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            ckpt = {
                "model": model.state_dict(),
                "label_map": label_map,
                "num_spells": num_spells,
            }
            torch.save(ckpt, out_dir / "mobilenetv3_multihead_best.pt")
            print(f"Saved best checkpoint: {best_val_top1:.4f}")

    # Export torchscript for easier deployment
    best_path = out_dir / "mobilenetv3_multihead_best.pt"
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 3, 96, 96, device=device)
    traced = torch.jit.trace(model, dummy)
    traced.save(str(out_dir / "mobilenetv3_multihead_best.torchscript.pt"))
    print("Exported TorchScript model.")
    onnx_path = out_dir / "mobilenetv3_multihead_best.onnx"

    dummy_input = torch.randn(1, 3, 96, 96, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["spell_logits", "state_logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "spell_logits": {0: "batch"},
            "state_logits": {0: "batch"},
        },
    )


if __name__ == "__main__":
    main()
