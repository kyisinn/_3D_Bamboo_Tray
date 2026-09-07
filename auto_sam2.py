"""Automatically isolate the bamboo tray with Meta SAM 2.

The script asks SAM 2 for all object masks, scores them for a large central
tray-like object, removes overlapping lower hand masks, and writes both a binary
mask and a transparent PNG. No missing pixels are generated or inpainted.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent


def choose_device(torch, requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mask_properties(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    height, width = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = xs.mean(), ys.mean()
    touches = sum((x0 == 0, y0 == 0, x1 == width - 1, y1 == height - 1))
    return {
        "area": len(xs), "box": (x0, y0, x1, y1),
        "center": (cx, cy), "touches": touches,
    }


def tray_score(annotation, image_rgb):
    mask = annotation["segmentation"].astype(bool)
    props = mask_properties(mask)
    if props is None:
        return -np.inf
    height, width = mask.shape
    area_fraction = props["area"] / (height * width)
    if not 0.012 <= area_fraction <= 0.78 or props["touches"] >= 3:
        return -np.inf

    cx, cy = props["center"]
    distance = np.hypot((cx-width/2)/width, (cy-height/2)/height)
    centrality = max(0.0, 1.0 - distance / 0.65)
    x0, y0, x1, y1 = props["box"]
    box_area = max(1, (x1-x0+1) * (y1-y0+1))
    solidity = props["area"] / box_area

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    pixels = hsv[mask]
    # Bamboo is generally warm and reasonably bright. This is only a preference;
    # SAM's object boundary and centrality remain the strongest signals.
    warm = np.mean((pixels[:, 0] >= 5) & (pixels[:, 0] <= 45))
    bright = np.mean(pixels[:, 2]) / 255.0
    quality = float(annotation.get("predicted_iou", 0.0))
    stability = float(annotation.get("stability_score", 0.0))
    return (
        2.2 * np.sqrt(area_fraction)
        + 1.6 * centrality + 0.7 * solidity
        + 0.6 * warm + 0.4 * bright
        + 0.8 * quality + 0.5 * stability
        - 0.35 * props["touches"]
    )


def remove_overlapping_hands(tray, annotations):
    """Subtract lower masks that extend from outside into the tray mask."""
    result = tray.copy()
    tray_props = mask_properties(tray)
    if tray_props is None:
        return result
    height, width = tray.shape
    tx0, ty0, tx1, ty1 = tray_props["box"]
    for annotation in annotations:
        candidate = annotation["segmentation"].astype(bool)
        props = mask_properties(candidate)
        if props is None or props["area"] > height * width * 0.22:
            continue
        overlap = np.count_nonzero(candidate & tray)
        outside = np.count_nonzero(candidate & ~tray)
        _, cy = props["center"]
        _, _, _, y1 = props["box"]
        enters_from_below = cy > ty0 + 0.55 * (ty1-ty0) and y1 >= ty1 - 0.08*(ty1-ty0)
        if enters_from_below and overlap > 250 and outside > overlap * 0.20:
            result[candidate] = False
    return result


def clean_mask(mask):
    binary = mask.astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              np.ones((7, 7), np.uint8), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = np.where(labels == largest, 255, 0).astype(np.uint8)
    return binary


def save_result(source, source_root, image_bgr, mask, mask_root, output_root):
    relative = source.relative_to(source_root).with_suffix(".png")
    mask_path, output_path = mask_root / relative, output_root / relative
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    rgba[mask == 0, :3] = 0
    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"Could not write {mask_path}")
    if not cv2.imwrite(str(output_path), rgba):
        raise OSError(f"Could not write {output_path}")
    return mask_path, output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=ROOT / "images" / "new_images")
    parser.add_argument("--masks", type=Path,
                        default=ROOT / "masks" / "sam2")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "images" / "new_images_sam2")
    parser.add_argument("--model", default="facebook/sam2.1-hiera-small",
                        help="SAM 2 Hugging Face model ID")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"),
                        default="auto")
    parser.add_argument("--points-per-side", type=int, default=32,
                        help="higher values improve detail but use more memory")
    parser.add_argument("--keep-hands", action="store_true",
                        help="do not subtract lower overlapping object masks")
    parser.add_argument("--limit", type=int,
                        help="process only the first N images for testing")
    parser.add_argument(
    "--start",
    type=int,
    default=1,
    help="1-based image number to start processing from",
)
    args = parser.parse_args()

    try:
        import torch
        from sam2.build_sam import build_sam2_hf
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as error:
        raise SystemExit(
            "SAM 2 is not installed in this environment. Install PyTorch and "
            "Meta's official sam2 package before running auto_sam2.py.\n"
            f"Import error: {error}"
        )

    source_root = args.input.resolve()
    images = sorted(
        path for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ) if source_root.is_dir() else []
    start_index = max(0, args.start - 1)
    images = images[start_index:]
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"No JPG or PNG files found in {source_root}")

    device = choose_device(torch, args.device)
    print(f"Loading {args.model} on {device}...")
    try:
        model = build_sam2_hf(args.model, device=device)
    except Exception as error:
        raise SystemExit(f"Could not load SAM 2 model '{args.model}': {error}")
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=args.points_per_side,
        points_per_batch=16 if device in {"mps", "cpu"} else 64,
        pred_iou_thresh=0.78,
        stability_score_thresh=0.90,
        min_mask_region_area=400,
        output_mode="binary_mask",
    )

    failures = []
    with torch.inference_mode():
        for index, source in enumerate(images, 1):
            try:
                bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("image could not be read")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                annotations = generator.generate(rgb)
                if not annotations:
                    raise ValueError("SAM 2 generated no masks")
                selected = max(annotations, key=lambda ann: tray_score(ann, rgb))
                if not np.isfinite(tray_score(selected, rgb)):
                    raise ValueError("no plausible central tray mask was found")
                tray = selected["segmentation"].astype(bool)
                if not args.keep_hands:
                    tray = remove_overlapping_hands(tray, annotations)
                mask = clean_mask(tray)
                mask_path, _ = save_result(
                    source, source_root, bgr, mask,
                    args.masks.resolve(), args.output.resolve(),
                )
                print(f"[{index}/{len(images)}] {source.name}: "
                      f"{len(annotations)} candidates -> {mask_path.name}")
            except Exception as error:
                failures.append((source.name, str(error)))
                print(f"[{index}/{len(images)}] FAILED {source.name}: {error}",
                      file=sys.stderr)

    print(f"Processed {len(images)-len(failures)}/{len(images)} images")
    print(f"Masks: {args.masks.resolve()}")
    print(f"Transparent images: {args.output.resolve()}")
    if failures:
        raise SystemExit(f"Failed images: {len(failures)}")


if __name__ == "__main__":
    main()
