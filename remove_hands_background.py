"""Prepare every source image for masked COLMAP reconstruction.

Images are collected recursively from images/, including images/new_images/.
Masked PNGs and matching binary masks are written to the directories consumed
by run_colmap.py.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent


def largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return np.zeros_like(mask)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def components_near_bottom(mask):
    """Keep skin-colored components connected to the lower image region."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result = np.zeros_like(mask)
    height, width = mask.shape
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        touches_lower_region = y + h >= height * 0.78
        large_enough = area >= height * width * 0.00015
        if touches_lower_region and large_enough:
            result[labels == label] = 255
    return result


def detect_tray_rim(image):
    """Detect a closed tray rim from grayscale Canny edges, without circles."""
    height, width = image.shape[:2]
    scale = min(1.0, 1000 / max(height, width))
    detection = cv2.resize(image, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(detection, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 9)
    image_center = np.array([detection.shape[1] / 2, detection.shape[0] / 2])
    image_area = detection.shape[0] * detection.shape[1]
    min_area = image_area * 0.003
    max_area = image_area * 0.90
    candidates = []

    # Several closing sizes handle both a broad frontal rim and a thin edge-on
    # rim while keeping the geometry entirely edge-driven.
    for low, high, kernel_size in ((35, 105, 9), (45, 130, 17), (60, 160, 25)):
        edges = cv2.Canny(gray, low, high)
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        contours, _ = cv2.findContours(
            closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if len(contour) < 20 or not min_area <= area <= max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < detection.shape[1] * 0.10 or h < detection.shape[0] * 0.02:
                continue
            center = np.array([x + w / 2, y + h / 2])
            distance = np.linalg.norm(center - image_center)
            if distance > max(detection.shape) * 0.60:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            compactness = 4 * np.pi * area / (perimeter * perimeter)
            centrality = 1.0 - distance / (max(detection.shape) * 0.60)
            touches_border = (
                x == 0 or y == 0
                or x + w >= detection.shape[1]
                or y + h >= detection.shape[0]
            )
            border_weight = 0.25 if touches_border else 1.0
            edge_support = cv2.mean(edges, mask=cv2.drawContours(
                np.zeros_like(edges), [contour], -1, 255, 2
            ))[0] / 255
            score = (
                area * (0.5 + 0.5 * centrality)
                * (0.3 + compactness) * (0.5 + edge_support)
                * border_weight
            )
            candidates.append((score, contour))

    if not candidates:
        raise ValueError("closed tray rim was not detected by Canny")
    contour = max(candidates, key=lambda candidate: candidate[0])[1]
    hull = cv2.convexHull(contour).astype(np.float32) / scale
    return np.rint(hull).astype(np.int32)


def create_tray_mask(image):
    """Return an 8-bit mask: tray=255, hand/background=0."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Fill the actual Canny rim contour. No ellipse or tray-angle estimate is used.
    outline = np.zeros((height, width), np.uint8)
    rim = detect_tray_rim(image)
    cv2.drawContours(outline, [rim], -1, 255, -1, cv2.LINE_AA)

    # Standard skin-color tests in both YCrCb and HSV. Restrict removal to skin
    # components that reach the lower part of the image to avoid deleting bamboo.
    skin_ycc = cv2.inRange(ycrcb, (35, 132, 72), (255, 180, 135))
    skin_hsv = cv2.inRange(hsv, (0, 35, 45), (22, 230, 255))
    skin = cv2.bitwise_and(skin_ycc, skin_hsv)
    skin = cv2.morphologyEx(
        skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1
    )
    skin = cv2.morphologyEx(
        skin, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8), iterations=2
    )
    # Bamboo and skin overlap strongly in color. Only permit hand removal in the
    # lower section where hands enter the detected tray boundary; otherwise the
    # woven bamboo can incorrectly become one large "skin" component.
    ox, oy, ow, oh = cv2.boundingRect(outline)
    hand_zone = np.zeros_like(skin)
    hand_zone[int(oy + 0.68 * oh): min(height, oy + oh), :] = 255
    skin = cv2.bitwise_and(skin, hand_zone)
    skin = components_near_bottom(skin)
    skin = cv2.dilate(skin, np.ones((11, 11), np.uint8), iterations=1)

    mask = cv2.bitwise_and(outline, cv2.bitwise_not(skin))
    # Preserve a hard mask for mesh/photogrammetry; no inpainting is performed.
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def process(source, source_root, output_dir, mask_dir):
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("image could not be read")
    mask = create_tray_mask(image)
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    rgba[mask == 0, :3] = 0
    relative = source.relative_to(source_root).with_suffix(".png")
    output = output_dir / relative
    mask_output = mask_dir / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    mask_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), rgba):
        raise OSError(f"could not write {output}")
    if not cv2.imwrite(str(mask_output), mask):
        raise OSError(f"could not write {mask_output}")
    return output, mask_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "images")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "images" / "colmap_images")
    parser.add_argument("--masks", type=Path,
                        default=ROOT / "masks" / "colmap_images")
    args = parser.parse_args()
    source_dir = args.input.resolve()
    output_dir, mask_dir = args.output.resolve(), args.masks.resolve()
    excluded_roots = {output_dir, ROOT / "images" / "new_images_masked"}
    files = sorted(
        path for path in source_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and not any(excluded in path.parents for excluded in excluded_roots)
    ) if source_dir.is_dir() else []
    if not files:
        raise SystemExit(f"No JPG or PNG images found in: {source_dir}")

    # Prevent failed reruns from leaving stale images or masks for COLMAP.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if mask_dir.exists():
        shutil.rmtree(mask_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for index, source in enumerate(files, 1):
        try:
            output, _ = process(source, source_dir, output_dir, mask_dir)
            print(f"[{index}/{len(files)}] {source.relative_to(source_dir)} -> {output.relative_to(output_dir)}")
        except Exception as error:
            failures.append((source.name, str(error)))
            print(f"[{index}/{len(files)}] SKIPPED {source.name}: {error}")

    print(f"\nProcessed: {len(files) - len(failures)}")
    print(f"Masked images: {output_dir}")
    print(f"Binary masks: {mask_dir}")
    if failures:
        raise SystemExit(f"Failed: {len(failures)} image(s)")


if __name__ == "__main__":
    main()
