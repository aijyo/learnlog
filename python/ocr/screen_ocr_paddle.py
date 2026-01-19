import argparse
import time
from typing import List, Tuple

import numpy as np
import cv2
import mss
import os
os.environ["FLAGS_use_mkldnn"] = "0"          # Disable oneDNN/MKLDNN
os.environ["FLAGS_enable_pir_api"] = "0"      # Try to disable PIR API path
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

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

        shot = sct.grab(monitor)
        img = np.array(shot)  # BGRA
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img_bgr


def run_ocr(ocr: PaddleOCR, img_bgr: np.ndarray) -> List[Tuple[str, float, List[List[float]]]]:
    """
    English comment:
    Run PaddleOCR on a BGR image.
    Return list of (text, confidence, box_points).
    """
    result = ocr.ocr(img_bgr, cls=False)
    out = []
    if not result:
        return out

    # result structure: [ [ [box, (text, score)], ... ] ]
    lines = result[0] if isinstance(result, list) else result
    for item in lines:
        if not item or len(item) < 2:
            continue
        box = item[0]
        text, score = item[1]
        out.append((text, float(score), box))
    return out


def draw_ocr_boxes(img_bgr: np.ndarray, items: List[Tuple[str, float, List[List[float]]]]) -> np.ndarray:
    """
    English comment:
    Draw OCR bounding boxes and text on image for debugging.
    """
    vis = img_bgr.copy()
    for text, score, box in items:
        pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        x, y = int(box[0][0]), int(box[0][1])
        cv2.putText(
            vis,
            f"{text} ({score:.2f})",
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return vis


def main():
    parser = argparse.ArgumentParser(description="Screen capture + PaddleOCR demo")
    parser.add_argument("--lang", type=str, default="ch", help="OCR language, e.g. ch/en")
    parser.add_argument("--use_gpu", action="store_false", help="Use GPU if available")
    parser.add_argument("--region", type=str, default="", help='Capture region "left,top,width,height"')
    parser.add_argument("--interval", type=float, default=0.0, help="If >0, run repeatedly every N seconds")
    parser.add_argument("--save", type=str, default="", help="Save screenshot to file path")
    parser.add_argument("--vis", type=str, default="", help="Save visualization with boxes to file path")
    args = parser.parse_args()

    region = None
    if args.region.strip():
        parts = [int(x.strip()) for x in args.region.split(",")]
        if len(parts) != 4:
            raise ValueError('Invalid --region, expected "left,top,width,height"')
        region = (parts[0], parts[1], parts[2], parts[3])

    # English comment:
    # Initialize PaddleOCR once (important for performance).
    ocr = PaddleOCR(
        # use_angle_cls=True,
        lang=args.lang,
        use_textline_orientation=False,
    )
    def one_round():
        img = capture_screen(region)
        if args.save:
            cv2.imwrite(args.save, img)

        items = run_ocr(ocr, img)

        if not items:
            print("No text detected.")
            return

        # English comment:
        # Sort results by confidence (high -> low)
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)

        print("=== OCR detailed results ===")
        all_texts = []  # English comment: collect all recognized texts

        for i, (text, score, box) in enumerate(items_sorted, 1):
            print(f"[{i}] score={score:.4f} text={text}")
            print(f"    box={box}")
            all_texts.append(text)

        # English comment:
        # Print all recognized text together
        print("\n=== OCR all text ===")
        print("\n".join(all_texts))

        if args.vis:
            vis = draw_ocr_boxes(img, items_sorted)
            cv2.imwrite(args.vis, vis)
            print(f"Saved visualization to: {args.vis}")


    if args.interval > 0:
        while True:
            print("=" * 60)
            one_round()
            time.sleep(args.interval)
    else:
        one_round()


if __name__ == "__main__":
    main()
