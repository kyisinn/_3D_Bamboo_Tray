"""Create a rotatable 2.5D tray mesh from one front-facing photograph."""
import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent


def load_image(path):
    image = cv2.imread(str(path))
    if image is not None:
        return image
    # OpenCV may not support HEIC, so ask macOS to convert it temporarily.
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        converted = Path(tmp) / "input.png"
        subprocess.run(
            ["sips", "-s", "format", "png", str(path), "--out", str(converted)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        image = cv2.imread(str(converted))
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    return image


def find_tray(image):
    scale = min(1.0, 1200 / max(image.shape[:2]))
    small = cv2.resize(image, None, fx=scale, fy=scale)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # Golden bamboo against the dark curtain.
    mask = cv2.inRange(hsv, (5, 25, 65), (45, 255, 255))
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > mask.size * 0.05]
    if not contours:
        raise SystemExit("Could not find the tray outline in the photograph.")
    center, axes, angle = cv2.fitEllipse(max(contours, key=cv2.contourArea))
    return np.array(center) / scale, np.array(axes) / scale, np.deg2rad(angle)


def build_mesh(image, ellipse, rings=72, slices=240):
    center, axes, angle = ellipse
    vertices, colors, faces = [], [], []

    def add_vertex(x, y, z, color):
        vertices.append((x, y, z))
        colors.append(tuple(int(v) for v in color))
        return len(vertices) - 1

    # Front center.
    cx, cy = np.rint(center).astype(int)
    center_color = image[np.clip(cy, 0, image.shape[0]-1),
                         np.clip(cx, 0, image.shape[1]-1)][::-1]
    front = [[add_vertex(0, 0, -0.08, center_color)]]
    ca, sa = np.cos(angle), np.sin(angle)
    for ring in range(1, rings + 1):
        r = ring / rings
        row = []
        # Shallow concavity plus a raised outer bamboo rim.
        z = -0.08 * (1 - r * r) + 0.035 * np.exp(-((1-r)/0.055)**2)
        for j in range(slices):
            t = 2 * np.pi * j / slices
            u, v = r * np.cos(t), r * np.sin(t)
            px = center[0] + 0.5 * axes[0] * u * ca - 0.5 * axes[1] * v * sa
            py = center[1] + 0.5 * axes[0] * u * sa + 0.5 * axes[1] * v * ca
            ix = int(np.clip(round(px), 0, image.shape[1]-1))
            iy = int(np.clip(round(py), 0, image.shape[0]-1))
            row.append(add_vertex(u, -v, z, image[iy, ix][::-1]))
        front.append(row)

    for j in range(slices):
        faces.append((front[0][0], front[1][j], front[1][(j+1) % slices]))
    for ring in range(1, rings):
        for j in range(slices):
            k = (j + 1) % slices
            a, b = front[ring][j], front[ring][k]
            c, d = front[ring+1][j], front[ring+1][k]
            faces.extend(((a, c, d), (a, d, b)))

    # Flat back and a closed outer wall make the result a solid mesh.
    back_color = (177, 137, 77)
    back_center = add_vertex(0, 0, -0.16, back_color)
    back = []
    for j in range(slices):
        t = 2 * np.pi * j / slices
        back.append(add_vertex(np.cos(t), -np.sin(t), -0.16, back_color))
    for j in range(slices):
        k = (j + 1) % slices
        faces.append((back_center, back[k], back[j]))
        faces.extend(((front[-1][j], back[j], back[k]),
                      (front[-1][j], back[k], front[-1][k])))
    return vertices, colors, faces


def write_ply(path, vertices, colors, faces):
    with path.open("w") as out:
        out.write("ply\nformat ascii 1.0\n")
        out.write(f"element vertex {len(vertices)}\n")
        out.write("property float x\nproperty float y\nproperty float z\n")
        out.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        out.write(f"element face {len(faces)}\n")
        out.write("property list uchar int vertex_indices\nend_header\n")
        for vertex, color in zip(vertices, colors):
            out.write("{} {} {} {} {} {}\n".format(*vertex, *color))
        for face in faces:
            out.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="front-facing JPG, PNG, or HEIC photo")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "single_image_tray.ply")
    args = parser.parse_args()
    image = load_image(args.image.resolve())
    vertices, colors, faces = build_mesh(image, find_tray(image))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_ply(args.output, vertices, colors, faces)
    print(f"Saved {args.output} ({len(vertices):,} vertices, {len(faces):,} faces)")


if __name__ == "__main__":
    main()
