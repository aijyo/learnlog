import os
import argparse
from typing import List, Tuple

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    # Create directory if not exists.
    os.makedirs(path, exist_ok=True)


def binarize_white_bg(bgr: np.ndarray, thr: int = 200) -> np.ndarray:
    """
    White background binarization.
    Return fg01: uint8 {0,1}, 1 means text foreground.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fg01 = (gray < thr).astype(np.uint8)
    return fg01


def find_row_runs_by_projection(
    fg01: np.ndarray,
    min_run_h: int = 6,
    pad_y: int = 1
) -> List[Tuple[int, int]]:
    """
    Split rows (lines) using horizontal projection.
    Returns list of [y0, y1] inclusive.
    """
    h, w = fg01.shape
    row_sum = fg01.sum(axis=1)

    # A row is active if it contains enough foreground pixels.
    active = row_sum >= max(2, int(0.002 * w))

    runs = []
    y = 0
    while y < h:
        if not bool(active[y]):
            y += 1
            continue
        y0 = y
        while y < h and bool(active[y]):
            y += 1
        y1 = y - 1
        if (y1 - y0 + 1) >= min_run_h:
            y0 = max(0, y0 - pad_y)
            y1 = min(h - 1, y1 + pad_y)
            runs.append((y0, y1))
    return runs


def split_tokens_by_blank_columns(
    line_fg01: np.ndarray,
    blank_col_thr: int = 0,
    min_blank_run: int = 2,
    min_token_w: int = 1
) -> List[Tuple[int, int]]:
    """
    Split a single line into tokens by blank column runs (spaces).
    Returns token ranges [x0, x1] inclusive.
    NOTE: We DO NOT trim token to fg columns to avoid swallowing thin glyphs like '_'.
    """
    h, w = line_fg01.shape
    col_sum = line_fg01.sum(axis=0)
    blank = col_sum <= blank_col_thr

    # Find blank runs (separators)
    blank_runs = []
    x = 0
    while x < w:
        if not bool(blank[x]):
            x += 1
            continue
        x0 = x
        while x < w and bool(blank[x]):
            x += 1
        x1 = x - 1
        if (x1 - x0 + 1) >= min_blank_run:
            blank_runs.append((x0, x1))

    # Tokens are between blank runs
    tokens = []
    prev_end = -1
    for b0, b1 in blank_runs:
        t0 = prev_end + 1
        t1 = b0 - 1
        if t1 >= t0 and (t1 - t0 + 1) >= min_token_w:
            tokens.append((t0, t1))
        prev_end = b1

    if prev_end < w - 1:
        t0 = prev_end + 1
        t1 = w - 1
        if t1 >= t0 and (t1 - t0 + 1) >= min_token_w:
            tokens.append((t0, t1))

    # Optional: drop pure-blank tokens (shouldn't happen often)
    out = []
    for x0, x1 in tokens:
        if line_fg01[:, x0:x1 + 1].sum() == 0:
            continue
        out.append((x0, x1))
    return out


def normalize_keep_line_height(
    crop_bin255: np.ndarray,
    out_h: int = 32,
    out_w: int = 32,
    pad: int = 2
) -> np.ndarray:
    """
    Normalize a token image WITHOUT tight-cropping:
    - Scale so that height fits available height (out_h - 2*pad)
    - Preserve aspect ratio
    - Center pad/crop to out_w
    """
    h, w = crop_bin255.shape[:2]
    h = max(1, h)
    w = max(1, w)

    ah = max(1, out_h - 2 * pad)
    aw = max(1, out_w - 2 * pad)

    # Scale based on height only (keep line height consistent across tokens)
    scale = ah / h
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))

    resized = cv2.resize(crop_bin255, (nw, nh), interpolation=cv2.INTER_NEAREST)

    out = np.zeros((out_h, out_w), dtype=np.uint8)
    # Center
    y0 = (out_h - nh) // 2
    x0 = (out_w - nw) // 2

    # If too wide, center-crop
    if nw > out_w:
        sx = (nw - out_w) // 2
        resized = resized[:, sx:sx + out_w]
        nh, nw = resized.shape
        y0 = (out_h - nh) // 2
        x0 = 0

    # If too tall (unlikely), center-crop
    if nh > out_h:
        sy = (nh - out_h) // 2
        resized = resized[sy:sy + out_h, :]
        nh, nw = resized.shape
        y0 = 0
        x0 = (out_w - nw) // 2

    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def process_image(
    image_path: str,
    out_dir: str,
    thr: int = 200,
    out_size: int = 32,
    pad: int = 2,
    # keep one bg row/col around fg token
    keep_bg_rows: int = 1,
    keep_bg_cols: int = 1,
    # column split
    min_blank_run: int = 2
) -> None:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    ensure_dir(out_dir)
    chars_dir = os.path.join(out_dir, "chars")
    ensure_dir(chars_dir)

    fg01 = binarize_white_bg(bgr, thr=thr)
    h, w = fg01.shape

    # Debug bin
    cv2.imwrite(os.path.join(out_dir, "fg01.png"), (fg01 * 255).astype(np.uint8))

    # Split lines
    lines = find_row_runs_by_projection(fg01, min_run_h=6, pad_y=1)

    overlay = bgr.copy()
    idx = 0

    for li, (y0, y1) in enumerate(lines):
        # Expand one bg row above/below for the entire line (uniform line height)
        yy0 = max(0, y0 - keep_bg_rows)
        yy1 = min(h - 1, y1 + keep_bg_rows)

        # Draw line bbox
        cv2.rectangle(overlay, (0, yy0), (w - 1, yy1), (0, 255, 0), 1)

        line_fg01 = fg01[yy0:yy1 + 1, :]

        # Split by blank columns (space gaps)
        tokens = split_tokens_by_blank_columns(
            line_fg01=line_fg01,
            blank_col_thr=0,
            min_blank_run=min_blank_run,
            min_token_w=1
        )

        for (x0, x1) in tokens:
            # Expand one bg col on both sides (bffb behavior)
            xx0 = max(0, x0 - keep_bg_cols)
            xx1 = min(w - 1, x1 + keep_bg_cols)

            token01 = line_fg01[:, xx0:xx1 + 1]  # keep full line height
            token255 = (token01 * 255).astype(np.uint8)

            # Normalize (keep line height behavior)
            norm = normalize_keep_line_height(token255, out_h=out_size, out_w=out_size, pad=pad)

            out_name = f"tok_{idx:05d}_line{li:02d}_x{xx0:04d}.png"
            cv2.imwrite(os.path.join(chars_dir, out_name), norm)

            # Draw token bbox on overlay
            cv2.rectangle(overlay, (xx0, yy0), (xx1, yy1), (0, 0, 255), 1)

            idx += 1

    cv2.imwrite(os.path.join(out_dir, "overlay.png"), overlay)

    print("Done.")
    print(f"Input : {image_path}")
    print(f"Output: {out_dir}")
    print(f"Lines : {len(lines)}")
    print(f"Tokens: {idx}")
    print("Debug : fg01.png, overlay.png")
    print("Chars : out_dir/chars/*.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=r"D:\code\mycode\python\learn\ocr\ocr_char.png", help="Input screenshot path")
    ap.add_argument("--out", default=r"D:\code\mycode\python\learn\ocr\output", help="Output folder")
    ap.add_argument("--thr", type=int, default=200, help="Binarize threshold for white background")
    ap.add_argument("--size", type=int, default=32, help="Output square size (e.g., 32)")
    ap.add_argument("--pad", type=int, default=2, help="Padding inside output image")
    ap.add_argument("--keep_bg_rows", type=int, default=1, help="Keep N background rows above/below line")
    ap.add_argument("--keep_bg_cols", type=int, default=1, help="Keep N background cols left/right of token")
    ap.add_argument("--min_blank_run", type=int, default=2, help="Min blank columns run length for splitting tokens")
    args = ap.parse_args()

    process_image(
        image_path=args.image,
        out_dir=args.out,
        thr=args.thr,
        out_size=args.size,
        pad=args.pad,
        keep_bg_rows=args.keep_bg_rows,
        keep_bg_cols=args.keep_bg_cols,
        min_blank_run=args.min_blank_run
    )


if __name__ == "__main__":
    main()
