"""Interactively mark tray pixels and remove hands from a photo sequence.

Controls:
  Left click       add a point to the active polygon
  T                switch to tray polygon mode
  H                switch to hand/background removal mode
  Enter            apply the active polygon
  Backspace / U    undo the last point
  C                cancel the active polygon
  R                reset the complete mask for this image
  S                save
  N / Space        save and move to the next image
  P                move to the previous image
  Q / Escape       save and quit
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
WINDOW = "Manual tray mask"


class MaskEditor:
    def __init__(self, images, source_root, mask_root, output_root, max_display=1200):
        self.images = images
        self.source_root = source_root
        self.mask_root = mask_root
        self.output_root = output_root
        self.max_display = max_display
        self.index = 0
        self.image = None
        self.mask = None
        self.points = []
        self.mode = "tray"
        self.scale = 1.0
        self.saved = True

    def relative_png(self, source):
        return source.relative_to(self.source_root).with_suffix(".png")

    def load(self, index):
        self.index = max(0, min(index, len(self.images) - 1))
        source = self.images[self.index]
        self.image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if self.image is None:
            raise RuntimeError(f"Could not read {source}")
        mask_path = self.mask_root / self.relative_png(source)
        existing = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        self.mask = (
            np.where(existing >= 128, 255, 0).astype(np.uint8)
            if existing is not None and existing.shape == self.image.shape[:2]
            else np.zeros(self.image.shape[:2], np.uint8)
        )
        self.scale = min(1.0, self.max_display / max(self.image.shape[:2]))
        self.points = []
        self.mode = "tray"
        self.saved = True

    def mouse(self, event, x, y, _flags, _data):
        if event == cv2.EVENT_LBUTTONDOWN:
            px = int(round(x / self.scale))
            py = int(round(y / self.scale))
            px = int(np.clip(px, 0, self.image.shape[1] - 1))
            py = int(np.clip(py, 0, self.image.shape[0] - 1))
            self.points.append((px, py))
            self.saved = False

    def apply_polygon(self):
        if len(self.points) < 3:
            print("A polygon needs at least three points.")
            return
        polygon = np.asarray(self.points, np.int32)
        color = 255 if self.mode == "tray" else 0
        cv2.fillPoly(self.mask, [polygon], color, lineType=cv2.LINE_AA)
        self.mask = np.where(self.mask >= 128, 255, 0).astype(np.uint8)
        self.points = []
        self.saved = False

    def save(self):
        source = self.images[self.index]
        relative = self.relative_png(source)
        mask_path = self.mask_root / relative
        output_path = self.output_root / relative
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rgba = cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = self.mask
        rgba[self.mask == 0, :3] = 0
        if not cv2.imwrite(str(mask_path), self.mask):
            raise OSError(f"Could not save {mask_path}")
        if not cv2.imwrite(str(output_path), rgba):
            raise OSError(f"Could not save {output_path}")
        self.saved = True
        print(f"Saved mask: {mask_path}")
        print(f"Saved image: {output_path}")

    def frame(self):
        display = self.image.copy()
        dark = display.copy()
        dark[:] = (15, 15, 15)
        alpha = (self.mask.astype(np.float32) / 255.0)[:, :, None]
        display = (display * (0.35 + 0.65 * alpha) + dark * (0.65 * (1-alpha))).astype(np.uint8)

        if self.points:
            points = np.asarray(self.points, np.int32)
            color = (60, 230, 60) if self.mode == "tray" else (40, 40, 255)
            cv2.polylines(display, [points], False, color, 7, cv2.LINE_AA)
            for point in points:
                cv2.circle(display, tuple(point), 9, color, -1, cv2.LINE_AA)

        display = cv2.resize(
            display, None, fx=self.scale, fy=self.scale,
            interpolation=cv2.INTER_AREA,
        )
        status = (
            f"{self.index+1}/{len(self.images)}  {self.images[self.index].name}  "
            f"MODE: {self.mode.upper()}  points: {len(self.points)}"
        )
        cv2.rectangle(display, (0, 0), (display.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(display, status, (10, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)
        return display

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self.mouse)
        self.load(0)
        while True:
            cv2.imshow(WINDOW, self.frame())
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key == ord("t"):
                self.points = []
                self.mode = "tray"
            elif key == ord("h"):
                self.points = []
                self.mode = "remove"
            elif key in (13, 10):
                self.apply_polygon()
            elif key in (8, 127, ord("u")) and self.points:
                self.points.pop()
            elif key == ord("c"):
                self.points = []
            elif key == ord("r"):
                self.mask[:] = 0
                self.points = []
                self.saved = False
            elif key == ord("s"):
                self.save()
            elif key in (ord("n"), ord(" ")):
                if self.points:
                    self.apply_polygon()
                self.save()
                if self.index < len(self.images) - 1:
                    self.load(self.index + 1)
            elif key == ord("p"):
                if not self.saved:
                    self.save()
                if self.index > 0:
                    self.load(self.index - 1)
            elif key in (ord("q"), 27):
                if not self.saved:
                    self.save()
                break
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=ROOT / "images" / "new_images")
    parser.add_argument("--masks", type=Path,
                        default=ROOT / "masks" / "manual")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "images" / "new_images_manually_masked")
    parser.add_argument("--start", help="optional starting filename, e.g. IMG_9420.jpg")
    args = parser.parse_args()
    source_root = args.input.resolve()
    images = sorted(
        path for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ) if source_root.is_dir() else []
    if not images:
        raise SystemExit(f"No JPG or PNG images found in {source_root}")

    editor = MaskEditor(
        images, source_root, args.masks.resolve(), args.output.resolve()
    )
    if args.start:
        matches = [i for i, path in enumerate(images) if path.name == args.start]
        if not matches:
            raise SystemExit(f"Starting image not found: {args.start}")
        editor.index = matches[0]
        editor.load(editor.index)
        # run() normally loads zero; retain the requested start via sliced input.
        editor.images = images[editor.index:]
    editor.run()


if __name__ == "__main__":
    main()
