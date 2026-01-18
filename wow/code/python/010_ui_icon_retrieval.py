import os
import sys
import re
import time
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    # English comment:
    # Make randomness reproducible for dataset sampling and augmentation.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pil_open_rgb(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    # English comment:
    # Convert to RGB while preserving alpha compositing on black background.
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    return img


_FNAME_RE = re.compile(r"^(\d+)_(\d+)_(.+)$")


def parse_fdid_index_state_from_stem(stem: str) -> Optional[Tuple[int, int, str]]:
    """
    English comment:
    Parse "{fdid}_{index}_{state}" from filename stem.
    state may be numeric or word, keep as string.
    """
    m = _FNAME_RE.match(stem)
    if not m:
        return None
    fdid = int(m.group(1))
    idx = int(m.group(2))
    state = str(m.group(3))
    return fdid, idx, state


def make_label_key(fdid: int, state: str) -> str:
    # English comment:
    # Stable composite label key used for SupCon labels and retrieval ids.
    return f"{fdid}|{state}"


def split_label_key(key: str) -> Tuple[int, str]:
    # English comment:
    # Convert "fdid|state" back to (fdid, state).
    a, b = key.split("|", 1)
    return int(a), b


# -----------------------------
# Synthetic UI transform
# -----------------------------
def synth_ui_like(img_rgb_64: Image.Image) -> Image.Image:
    """
    English comment:
    Create a UI-like 64x64 patch from a raw icon 64x64 by simulating:
      - In-game downscale (to ~38x38) and pad back to 64x64
      - slight blur / jpeg artifacts
      - optional overlay masks (cooldown/casting style)
      - noise / gamma / desaturation shifts
    This function is intentionally simple but effective.
    """
    assert img_rgb_64.size == (64, 64)

    # 1) Downscale to inner icon size (simulate UI shrink)
    inner = random.choice([36, 38, 40])
    resample = random.choice([Image.BILINEAR, Image.BICUBIC])
    small = img_rgb_64.resize((inner, inner), resample=resample)

    # 2) Pad to 64x64 centered
    canvas = Image.new("RGB", (64, 64), (0, 0, 0))
    off_x = (64 - inner) // 2 + random.randint(-1, 1)
    off_y = (64 - inner) // 2 + random.randint(-1, 1)
    off_x = max(0, min(64 - inner, off_x))
    off_y = max(0, min(64 - inner, off_y))
    canvas.paste(small, (off_x, off_y))
    img = canvas

    # 3) Color / gamma / contrast jitter
    if random.random() < 0.9:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.75, 1.15))
    if random.random() < 0.9:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.20))
    if random.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))

    # 4) Blur (UI scaling / capture blur)
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

    # 5) Optional overlay (simple geometric mask)
    if random.random() < 0.5:
        state_kind = random.choice(["none", "cooldown", "casting", "disabled"])
        if state_kind == "cooldown":
            arr = np.asarray(img).astype(np.float32)
            h, w = 64, 64
            yy, xx = np.mgrid[0:h, 0:w]
            a = random.uniform(-1.2, 1.2)
            b = random.uniform(-1.2, 1.2)
            t = random.uniform(40, 80)
            mask = (a * xx + b * yy) > t
            dark = random.uniform(0.35, 0.65)
            arr[mask] *= dark
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        elif state_kind == "casting":
            arr = np.asarray(img).astype(np.float32)
            y0 = random.randint(36, 52)
            y1 = min(63, y0 + random.randint(4, 10))
            bright = random.uniform(1.15, 1.45)
            arr[y0:y1, :, :] *= bright
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        elif state_kind == "disabled":
            img = ImageEnhance.Color(img).enhance(random.uniform(0.1, 0.35))
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.55, 0.85))

    # 6) Noise
    if random.random() < 0.7:
        arr = np.asarray(img).astype(np.float32)
        noise = np.random.normal(0.0, random.uniform(1.0, 4.0), size=arr.shape).astype(np.float32)
        arr = arr + noise
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # 7) Minor compression by down/up
    if random.random() < 0.3:
        q = random.choice([56, 60, 64])
        img = img.resize((q, q), resample=Image.BILINEAR).resize((64, 64), resample=Image.BILINEAR)

    return img


def icon_aug(img_rgb_64: Image.Image) -> Image.Image:
    """
    English comment:
    Lighter augmentation for raw icons (keeps semantics).
    """
    img = img_rgb_64
    if random.random() < 0.9:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.85, 1.20))
    if random.random() < 0.9:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.20))
    if random.random() < 0.6:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.90, 1.15))
    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))
    return img


# -----------------------------
# Dataset: SupCon views
# -----------------------------
@dataclass
class SampleViews:
    label_id: int
    label_key: str
    view_a: torch.Tensor
    view_b: torch.Tensor
    view_c: torch.Tensor


class RetrievalSupConDataset(Dataset):
    """
    English comment:
    Each item returns 3 views for the same (fdid,state):
      A: icon_aug(icon)
      B: synth_ui_like(icon)
      C: real_ui if available else synth_ui_like(icon) again

    Labels are composite keys: "fdid|state".
    """

    def __init__(self, icons_dir: Path, ui_real_dir: Optional[Path], tfm: T.Compose, sample_per_epoch: int = 50000):
        self.icons_dir = icons_dir
        self.ui_real_dir = ui_real_dir
        self.tfm = tfm
        self.sample_per_epoch = sample_per_epoch

        # (fdid,state) -> list[Path] for origin icons
        self.icon_map: Dict[str, List[Path]] = {}

        for p in sorted(icons_dir.glob("*.png")):
            parsed = parse_fdid_index_state_from_stem(p.stem)
            if not parsed:
                continue
            fdid, _idx, state = parsed
            key = make_label_key(fdid, state)
            self.icon_map.setdefault(key, []).append(p)

        if not self.icon_map:
            raise RuntimeError(f"No icons found under: {icons_dir}")

        # (fdid,state) -> list[Path] for real UI patches
        self.ui_map: Dict[str, List[Path]] = {}
        if ui_real_dir and ui_real_dir.exists():
            # Expect structure: used_spells/{fdid}/{state}/...png
            for p in sorted(ui_real_dir.rglob("*.png")):
                # Try parse from filename first
                parsed = parse_fdid_index_state_from_stem(p.stem)
                if parsed:
                    fdid, _idx, state = parsed
                    key = make_label_key(fdid, state)
                    self.ui_map.setdefault(key, []).append(p)
                    continue

                # Fallback: parse from parent folders {fdid}/{state}
                try:
                    state = p.parent.name
                    fdid = int(p.parent.parent.name)
                    key = make_label_key(fdid, state)
                    self.ui_map.setdefault(key, []).append(p)
                except Exception:
                    continue

        # Build label-id mapping
        self.keys: List[str] = sorted(self.icon_map.keys())
        self.key_to_id: Dict[str, int] = {k: i for i, k in enumerate(self.keys)}

    def __len__(self):
        # English comment:
        # We sample with replacement; an "epoch" is a fixed number of samples.
        return self.sample_per_epoch

    def __getitem__(self, idx: int) -> SampleViews:
        key = random.choice(self.keys)
        label_id = self.key_to_id[key]

        icon_path = random.choice(self.icon_map[key])
        icon = pil_open_rgb(icon_path).resize((64, 64), resample=Image.BILINEAR)

        a = icon_aug(icon)
        b = synth_ui_like(icon)

        if key in self.ui_map and random.random() < 0.7:
            real_path = random.choice(self.ui_map[key])
            c_img = pil_open_rgb(real_path).resize((64, 64), resample=Image.BILINEAR)
        else:
            c_img = synth_ui_like(icon)

        return SampleViews(
            label_id=label_id,
            label_key=key,
            view_a=self.tfm(a),
            view_b=self.tfm(b),
            view_c=self.tfm(c_img),
        )


def collate_views(batch: List[SampleViews]):
    # English comment:
    # Collate into tensors: [B,3,C,H,W] and labels [B]
    labels = torch.tensor([x.label_id for x in batch], dtype=torch.long)
    views = torch.stack([torch.stack([x.view_a, x.view_b, x.view_c], dim=0) for x in batch], dim=0)
    keys = [x.label_key for x in batch]
    return views, labels, keys


# -----------------------------
# Model
# -----------------------------
class EmbedNet(nn.Module):
    """
    English comment:
    Use a small backbone (MobileNetV3) and output L2-normalized embeddings.
    """

    def __init__(self, emb_dim: int = 128):
        super().__init__()
        backbone = torchvision.models.mobilenet_v3_small(weights=None)
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        x = F.normalize(x, dim=1)
        return x


# -----------------------------
# SupCon loss
# -----------------------------
def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    English comment:
    Supervised contrastive loss:
      features: [B, V, D] normalized
      labels:   [B] (class ids)
    """
    device = features.device
    B, V, D = features.shape
    feats = features.view(B * V, D)  # [B*V, D]
    lbls = labels.view(B, 1).repeat(1, V).view(B * V)  # [B*V]

    logits = torch.matmul(feats, feats.t()) / temperature  # [N,N]
    N = B * V

    lbls_eq = lbls.unsqueeze(0) == lbls.unsqueeze(1)  # [N,N]
    self_mask = torch.eye(N, device=device, dtype=torch.bool)
    pos_mask = lbls_eq & (~self_mask)

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    exp_logits = torch.exp(logits) * (~self_mask)
    denom = exp_logits.sum(dim=1, keepdim=True) + 1e-12

    log_prob = logits - torch.log(denom)

    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    loss = -(log_prob * pos_mask).sum(dim=1) / pos_count
    return loss.mean()


# -----------------------------
# Index building & retrieval
# -----------------------------
@torch.no_grad()
def build_icon_index(
    model: nn.Module,
    icons_dir: Path,
    device: torch.device,
    tfm: T.Compose,
    per_icon_synth: int = 2,
) -> Tuple[np.ndarray, List[str]]:
    """
    English comment:
    Build retrieval index for each (fdid,state):
      - For each origin icon file, compute embedding of raw icon + N synthetic UI variants.
      - ids are composite string keys: "fdid|state"
    """
    model.eval()
    embs: List[np.ndarray] = []
    keys: List[str] = []

    for p in sorted(icons_dir.glob("*.png")):
        parsed = parse_fdid_index_state_from_stem(p.stem)
        if not parsed:
            continue
        fdid, _idx, state = parsed
        key = make_label_key(fdid, state)

        icon = pil_open_rgb(p).resize((64, 64), resample=Image.BILINEAR)

        x = tfm(icon).unsqueeze(0).to(device)
        e = model(x).cpu().numpy()[0]
        embs.append(e)
        keys.append(key)

        for _ in range(per_icon_synth):
            ui = synth_ui_like(icon)
            x2 = tfm(ui).unsqueeze(0).to(device)
            e2 = model(x2).cpu().numpy()[0]
            embs.append(e2)
            keys.append(key)

    emb_mat = np.stack(embs, axis=0).astype(np.float32)
    emb_mat /= (np.linalg.norm(emb_mat, axis=1, keepdims=True) + 1e-12)
    return emb_mat, keys


def cosine_topk(query_emb: np.ndarray, emb_mat: np.ndarray, keys: List[str], k: int = 5):
    # English comment:
    # Simple cosine retrieval with numpy.
    sims = emb_mat @ query_emb.reshape(-1, 1)
    sims = sims.reshape(-1)
    top_idx = np.argpartition(-sims, kth=min(k, len(sims) - 1))[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(keys[i], float(sims[i])) for i in top_idx]


def export_embednet_onnx(model: nn.Module, onnx_path: Path, input_size: int = 64, opset: int = 17):
    # English comment:
    # Export on CPU to avoid CUDA-dependent export issues.
    model_cpu = model.cpu().eval()
    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    torch.onnx.export(
        model_cpu,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["emb"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "N"}, "emb": {0: "N"}},
    )

@torch.no_grad()
def infer_one(
    model: nn.Module,
    img_path: Path,
    emb_mat: np.ndarray,
    keys: List[str],
    device: torch.device,
    tfm: T.Compose,
    topk: int = 5,
    thresh: float = 0.55,
):
    # English comment:
    # Return (ok, best_key, best_score, topk_list)
    model.eval()
    img = pil_open_rgb(img_path).resize((64, 64), resample=Image.BILINEAR)
    x = tfm(img).unsqueeze(0).to(device)
    q = model(x).cpu().numpy()[0].astype(np.float32)
    q /= (np.linalg.norm(q) + 1e-12)

    results = cosine_topk(q, emb_mat, keys, k=topk)
    best_key, best_score = results[0]
    ok = best_score >= thresh
    return ok, best_key, best_score, results


# -----------------------------
# Train
# -----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    icons_dir = Path(args.icons_dir)
    ui_real_dir = Path(args.ui_real_dir) if args.ui_real_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    ds = RetrievalSupConDataset(
        icons_dir=icons_dir,
        ui_real_dir=ui_real_dir,
        tfm=tfm,
        sample_per_epoch=args.samples_per_epoch,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_views,
        drop_last=True,
    )

    model = EmbedNet(emb_dim=args.emb_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    last_ckpt = out_dir / "embednet_last.pt"
    best_loss = 1e9
    start_epoch = 1
    no_improve_epochs = 0

    if last_ckpt.exists():
        print(f"Resuming from checkpoint {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location="cpu")

        # English comment:
        # model.load_state_dict supports strict, optimizer does NOT.
        model.load_state_dict(ckpt["model"], strict=True)

        try:
            opt.load_state_dict(ckpt["opt"])
        except Exception as e:
            # English comment:
            # Optimizer state may be incompatible if hyper-params changed.
            print(f"WARNING: failed to load optimizer state, will re-init optimizer state. err={e}")

        best_loss = float(ckpt.get("best_loss", best_loss))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from epoch {start_epoch} with best_loss {best_loss:.4f}")

    # for epoch in range(start_epoch, args.epochs + 1):
    for epoch in range(0, args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        n = 0

        for views, labels, _keys in dl:
            views = views.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            B, V, C, H, W = views.shape
            flat = views.view(B * V, C, H, W)
            emb = model(flat).view(B, V, -1)

            loss = supcon_loss(emb, labels, temperature=args.temp)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            n += 1

        avg = running / max(1, n)
        dt = time.time() - t0
        print(f"[Epoch {epoch}+1/{args.epochs}] supcon_loss={avg:.4f} time={dt:.1f}s")


        improved = avg < (best_loss - args.min_delta)

        if improved:
            best_loss = avg
            best_path = out_dir / "embednet_best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": epoch + start_epoch,
                    "best_loss": best_loss,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  Saved BEST: {best_path} (loss={best_loss:.4f})")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"  No improve: {no_improve_epochs}/{args.patience} (min_delta={args.min_delta})")

            if no_improve_epochs >= args.patience:
                print("Early stopping triggered.")
                break
        
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch + start_epoch,
                "best_loss": best_loss,
                "args": vars(args),
            },
            last_ckpt,
        )

    # Prefer BEST weights for index + onnx export.
    best_path = out_dir / "embednet_best.pt"
    if best_path.exists():
        ckpt_best = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt_best["model"], strict=True)
        model.to(device)
        model.eval()

    print("Building icon index...")
    emb_mat, key_list = build_icon_index(
        model=model,
        icons_dir=icons_dir,
        device=device,
        tfm=tfm,
        per_icon_synth=args.per_icon_synth,
    )
    np.save(out_dir / "icon_emb.npy", emb_mat)
    np.save(out_dir / "icon_keys.npy", np.array(key_list, dtype=np.str_))
    print(f"Saved index: {out_dir/'icon_emb.npy'} and {out_dir/'icon_keys.npy'}")

    onnx_path = out_dir / "embednet_best.onnx"
    export_embednet_onnx(model, onnx_path, input_size=64, opset=17)
    print(f"Saved ONNX: {onnx_path}")


def load_model_and_index(model_path: Path, index_dir: Path, cpu: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    ckpt = torch.load(model_path, map_location="cpu")
    emb_dim = int(ckpt.get("args", {}).get("emb_dim", 128))

    model = EmbedNet(emb_dim=emb_dim).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    emb_mat = np.load(index_dir / "icon_emb.npy").astype(np.float32)
    keys_np = np.load(index_dir / "icon_keys.npy")
    keys = [str(x) for x in keys_np.tolist()]

    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return model, emb_mat, keys, device, tfm


def infer(args):
    model, emb_mat, keys, device, tfm = load_model_and_index(
        model_path=Path(args.model),
        index_dir=Path(args.index_dir),
        cpu=args.cpu,
    )

    img_path = Path(args.image)
    ok, best_key, best_score, results = infer_one(
        model=model,
        img_path=img_path,
        emb_mat=emb_mat,
        keys=keys,
        device=device,
        tfm=tfm,
        topk=args.topk,
        thresh=args.thresh,
    )

    best_fdid, best_state = split_label_key(best_key)

    print(f"Image: {img_path}")
    print(f"OK: {ok} | best_fdid={best_fdid} | best_state={best_state} | score={best_score:.4f} | thresh={args.thresh:.2f}")
    print("TopK:")
    for k, s in results:
        fdid, st = split_label_key(k)
        print(f"  fdid={fdid}  state={st}  score={s:.4f}")


def infer_dir(args):
    model, emb_mat, keys, device, tfm = load_model_and_index(
        model_path=Path(args.model),
        index_dir=Path(args.index_dir),
        cpu=args.cpu,
    )

    qdir = Path(args.dir)
    imgs: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
        imgs += list(qdir.rglob(ext))
    imgs = sorted(imgs)

    if not imgs:
        print(f"No images found in: {qdir}")
        return

    for p in imgs:
        ok, best_key, best_score, _ = infer_one(
            model=model,
            img_path=p,
            emb_mat=emb_mat,
            keys=keys,
            device=device,
            tfm=tfm,
            topk=args.topk,
            thresh=args.thresh,
        )
        fdid, st = split_label_key(best_key)
        print(f"{p}\tOK={ok}\tfdid={fdid}\tstate={st}\tscore={best_score:.4f}")


# -----------------------------
# CLI
# -----------------------------
def ensure_default_subcommand(argv, subcommands, default="train"):
    # English comment:
    # If argv has no subcommand, insert default.
    for a in argv[1:]:
        if a in subcommands:
            return
    argv.insert(1, default)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_train = sub.add_parser("train")
    p_train.add_argument("--icons_dir", type=str, default="data/origin_spells", help="Origin icons dir: data/origin_spells/*.png")
    p_train.add_argument("--ui_real_dir", type=str, default="data/used_spells", help="Real UI root: data/used_spells/{fdid}/{state}/")
    p_train.add_argument("--out_dir", type=str, default="output/index", help="Output index dir")
    p_train.add_argument("--epochs", type=int, default=8)
    p_train.add_argument("--batch_size", type=int, default=128)
    p_train.add_argument("--num_workers", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--emb_dim", type=int, default=128)
    p_train.add_argument("--temp", type=float, default=0.07)
    p_train.add_argument("--samples_per_epoch", type=int, default=60000)
    p_train.add_argument("--per_icon_synth", type=int, default=2)
    p_train.add_argument("--seed", type=int, default=123)
    p_train.add_argument("--cpu", action="store_true")
    p_train.add_argument("--patience", type=int, default=3, help="Early stop if no improve for N epochs")
    p_train.add_argument("--min_delta", type=float, default=1e-4, help="Minimal loss drop to be considered as improvement")


    p_inf = sub.add_parser("infer")
    p_inf.add_argument("--model", type=str, default="output/index/embednet_best.pt")
    p_inf.add_argument("--index_dir", type=str, default="output/index")
    p_inf.add_argument("--image", type=str, default="data/origin_spells/132127_000_0.png")
    p_inf.add_argument("--topk", type=int, default=5)
    p_inf.add_argument("--thresh", type=float, default=0.55)
    p_inf.add_argument("--cpu", action="store_true")

    p_dir = sub.add_parser("infer_dir")
    p_dir.add_argument("--model", type=str, default="output/index/embednet_best.pt")
    p_dir.add_argument("--index_dir", type=str, default="output/index")
    p_dir.add_argument("--dir", type=str, default="data/fake_icons")
    p_dir.add_argument("--topk", type=int, default=5)
    p_dir.add_argument("--thresh", type=float, default=0.55)
    p_dir.add_argument("--cpu", action="store_true")

    ensure_default_subcommand(sys.argv, {"train", "infer", "infer_dir"}, "train")
    args = parser.parse_args()

    if args.cmd == "train":
        if args.ui_real_dir == "":
            args.ui_real_dir = None
        train(args)
    elif args.cmd == "infer":
        infer(args)
    elif args.cmd == "infer_dir":
        infer_dir(args)
    else:
        raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
