import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

import argparse
import time
from typing import List, Tuple

import numpy as np
import cv2
import mss
from paddleocr import PaddleOCR


def capture_screen(region: Tuple[int, int, int, int] = None) -> np.ndarray:
    """
    English comment:
    Capture the screen (full screen by default) or a specific region using mss.
    region: (left, top, width, height)
    Return: BGR image (OpenCV format).
    """
    with mss.mss() as sct:
        if region is None:
            monitor = sct.monitors[1]  # Primary monitor
        else:
            left, top, width, height = region
            monitor = {"left": left, "top": top, "width": width, "height": height}

        shot = sct.grab(monitor)  # BGRA
        img = np.array(shot)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img_bgr


def parse_region(s: str) -> Tuple[int, int, int, int]:
    """
    English comment:
    Parse region string "left,top,width,height".
    """
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError('Invalid region, expected "left,top,width,height"')
    return parts[0], parts[1], parts[2], parts[3]


def parse_rois(s: str) -> List[Tuple[int, int, int, int]]:
    """
    English comment:
    Parse ROI list string:
      "x,y,w,h; x,y,w,h; ..."
    """
    rois = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [int(x.strip()) for x in chunk.split(",")]
        if len(parts) != 4:
            raise ValueError('Invalid roi item, expected "x,y,w,h"')
        rois.append((parts[0], parts[1], parts[2], parts[3]))
    if not rois:
        raise ValueError("No ROIs provided.")
    return rois


def preprocess_roi(roi_bgr: np.ndarray, upscale: float = 1.0, binarize: bool = False) -> np.ndarray:
    """
    English comment:
    Optional preprocessing to improve OCR on small UI text.
    - upscale: resize factor (>1.0 may improve recognition)
    - binarize: apply adaptive thresholding (can help on high-contrast UI)
    Return: BGR image for recognizer.
    """
    img = roi_bgr

    if upscale and upscale != 1.0:
        h, w = img.shape[:2]
        nh = max(1, int(h * upscale))
        nw = max(1, int(w * upscale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)

    if binarize:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        thr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
        )
        img = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

    return img


def filter_alnum(text: str) -> str:
    """
    English comment:
    Keep only [0-9A-Za-z] characters.
    """
    out = []
    for ch in text:
        if ("0" <= ch <= "9") or ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            out.append(ch)
    return "".join(out)


def recognition_batch(ocr: PaddleOCR, rois_bgr: List[np.ndarray]) -> List[Tuple[str, float]]:
    """
    English comment:
    Recognition-only batch inference (no detection).
    PaddleOCR v3: use ocr.text_recognizer(list_of_images).
    Return: list of (text, score) aligned with input order.
    """
    # Some PaddleOCR versions return list of tuples; others may return dict-like.
    rec_results = ocr.text_recognizer(rois_bgr)

    out: List[Tuple[str, float]] = []
    for res in rec_results:
        if res is None:
            out.append(("", 0.0))
            continue

        # Common case: (text, score)
        if isinstance(res, (list, tuple)) and len(res) >= 2 and isinstance(res[0], str):
            out.append((res[0], float(res[1])))
            continue

        # Fallback: try to parse dict-like
        if isinstance(res, dict):
            text = res.get("text", "") or ""
            score = res.get("score", 0.0) or 0.0
            out.append((str(text), float(score)))
            continue

        # Unknown format
        out.append((str(res), 0.0))

    return out


def crop_rois(img_bgr: np.ndarray, rois: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
    """
    English comment:
    Crop multiple ROIs from a BGR image.
    Each ROI is (x, y, w, h).
    """
    h_img, w_img = img_bgr.shape[:2]
    out = []
    for (x, y, w, h) in rois:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x2 <= x1 or y2 <= y1:
            out.append(np.zeros((1, 1, 3), dtype=np.uint8))
            continue
        out.append(img_bgr[y1:y2, x1:x2].copy())
    return out


def draw_rois(img_bgr: np.ndarray, rois: List[Tuple[int, int, int, int]], texts: List[str]) -> np.ndarray:
    """
    English comment:
    Draw ROI rectangles and recognized text for debugging.
    """
    vis = img_bgr.copy()
    for i, (x, y, w, h) in enumerate(rois):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = texts[i] if i < len(texts) else ""
        cv2.putText(vis, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return vis


def main():
    parser = argparse.ArgumentParser(description="Screen capture + PaddleOCR v3 recognition-only (batch ROIs)")
    parser.add_argument("--lang", type=str, default="en", help="OCR language model: en is good for digits/english")
    parser.add_argument("--interval", type=float, default=0.0, help="If >0, run repeatedly every N seconds")
    parser.add_argument("--region", type=str, default="", help='Capture region "left,top,width,height"')
    parser.add_argument("--rois", type=str, required=True, help='ROI list "x,y,w,h; x,y,w,h; ..."')
    parser.add_argument("--score_th", type=float, default=0.5, help="Min confidence threshold to accept text")
    parser.add_argument("--alnum_only", action="store_true", help="Keep only [0-9A-Za-z] in output text")
    parser.add_argument("--upscale", type=float, default=1.0, help="ROI upscale factor (e.g. 2.0)")
    parser.add_argument("--binarize", action="store_true", help="Apply adaptive threshold to ROI")
    parser.add_argument("--vis", type=str, default="", help="Save visualization image path (overwritten each round)")
    args = parser.parse_args()

    cap_region = parse_region(args.region) if args.region.strip() else None
    rois = parse_rois(args.rois)

    # English comment:
    # For UI OCR, orientation is usually unnecessary; disable for speed.
    ocr = PaddleOCR(
        lang=args.lang,
        use_textline_orientation=False,
    )

    def one_round():
        img = capture_screen(cap_region)
        roi_imgs = crop_rois(img, rois)
        roi_imgs = [preprocess_roi(r, upscale=args.upscale, binarize=args.binarize) for r in roi_imgs]

        # English comment:
        # Batch recognition = "multi-line parallel" (vectorized inference).
        rec = recognition_batch(ocr, roi_imgs)

        texts_for_vis = []
        for i, (text, score) in enumerate(rec):
            if args.alnum_only:
                text = filter_alnum(text)

            ok = score >= args.score_th and len(text.strip()) > 0
            if not ok:
                text_show = ""
            else:
                text_show = text.strip()

            texts_for_vis.append(text_show)

            # Output per ROI line
            print(f"[ROI {i}] score={score:.4f} text={text_show}")

        if args.vis:
            vis = draw_rois(img, rois, texts_for_vis)
            cv2.imwrite(args.vis, vis)
            print(f"Saved vis: {args.vis}")

    if args.interval > 0:
        while True:
            print("=" * 60)
            one_round()
            time.sleep(args.interval)
    else:
        one_round()


if __name__ == "__main__":
    main()
