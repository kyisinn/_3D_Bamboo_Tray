"""Build a closed textured tray mesh from a full rotation of photographs."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from single_image_3d import find_tray, load_image, write_ply


ROOT = Path(__file__).resolve().parent


def prepare(path, max_size=2400):
    image = load_image(path)
    scale = min(1.0, max_size / max(image.shape[:2]))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ellipse = find_tray(image)
    ratio = min(ellipse[1]) / max(ellipse[1])
    return image, ellipse, ratio


def sample(image, ellipse, u, v):
    center, axes, angle = ellipse
    ca, sa = np.cos(angle), np.sin(angle)
    px = center[0] + .5 * axes[0] * u * ca - .5 * axes[1] * v * sa
    py = center[1] + .5 * axes[0] * u * sa + .5 * axes[1] * v * ca
    x = int(np.clip(round(px), 0, image.shape[1] - 1))
    y = int(np.clip(round(py), 0, image.shape[0] - 1))
    return image[y, x][::-1]


def build(front_view, back_view, thickness, rings=144, slices=480):
    vertices, colors, faces = [], [], []

    def vertex(x, y, z, color):
        vertices.append((x, y, z))
        colors.append(tuple(int(c) for c in color))
        return len(vertices) - 1

    def surface(view, is_back=False):
        image, ellipse, _ = view
        grid = [[vertex(0, 0, -thickness if is_back else -0.08,
                        sample(image, ellipse, 0, 0))]]
        for ring in range(1, rings + 1):
            r = ring / rings
            row = []
            for j in range(slices):
                t = 2 * np.pi * j / slices
                u, v = r * np.cos(t), r * np.sin(t)
                if is_back:
                    z = -thickness + 0.025 * (1 - r*r)
                    u_tex = -u
                else:
                    z = -0.08 * (1-r*r) + 0.035*np.exp(-((1-r)/0.055)**2)
                    u_tex = u
                row.append(vertex(u, -v, z, sample(image, ellipse, u_tex, v)))
            grid.append(row)
        return grid

    front, back = surface(front_view), surface(back_view, True)
    for grid, reverse in ((front, False), (back, True)):
        for j in range(slices):
            tri = (grid[0][0], grid[1][j], grid[1][(j+1) % slices])
            faces.append(tri[::-1] if reverse else tri)
        for ring in range(1, rings):
            for j in range(slices):
                k = (j + 1) % slices
                a, b, c, d = grid[ring][j], grid[ring][k], grid[ring+1][j], grid[ring+1][k]
                tris = ((a, c, d), (a, d, b))
                faces.extend(tuple(reversed(t)) if reverse else t for t in tris)
    for j in range(slices):
        k = (j + 1) % slices
        faces.extend(((front[-1][j], back[-1][j], back[-1][k]),
                      (front[-1][j], back[-1][k], front[-1][k])))
    return vertices, colors, faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "multiview_tray_highres.ply")
    args = parser.parse_args()
    paths = sorted(dict.fromkeys(path.resolve() for path in args.images))
    if len(paths) < 6:
        raise SystemExit("Provide at least 6 ordered photographs around the tray.")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing image: {missing[0]}")

    views = []
    for index, path in enumerate(paths, 1):
        view = prepare(path)
        views.append(view)
        print(f"{index:02}/{len(paths)} {path.name}: frontal score={view[2]:.3f}")

    middle = len(views) // 2
    front_index = max(range(middle), key=lambda i: views[i][2])
    back_index = max(range(middle, len(views)), key=lambda i: views[i][2])
    edge_ratio = min(view[2] for view in views)
    thickness = float(np.clip(0.12 + edge_ratio * 0.35, 0.14, 0.28))
    vertices, colors, faces = build(views[front_index], views[back_index], thickness)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_ply(args.output, vertices, colors, faces)
    print(f"Front texture: {paths[front_index].name}")
    print(f"Back texture:  {paths[back_index].name}")
    print(f"Estimated thickness: {thickness:.3f}")
    print(f"Saved {args.output} ({len(vertices):,} vertices, {len(faces):,} faces)")


if __name__ == "__main__":
    main()
