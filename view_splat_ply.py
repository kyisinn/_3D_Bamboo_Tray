"""
Standalone Gaussian Splat viewer.

Reads a Nerfstudio-exported Gaussian PLY and renders it with gsplat.
Does NOT import Nerfstudio.

Default:
    exports/splat/splat.ply
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import torch
import open3d as o3d
from gsplat import rasterization


ROOT = Path(__file__).resolve().parent
DEFAULT_PLY = ROOT / "exports" / "splat" / "splat.ply"

SH_C0 = 0.28209479177387814


# ------------------------------------------------------------
# PLY READER
# ------------------------------------------------------------

def read_binary_ply(path: Path):
    with path.open("rb") as f:
        header_bytes = bytearray()

        while True:
            line = f.readline()

            if not line:
                raise ValueError("PLY header is missing end_header")

            header_bytes.extend(line)

            if line.strip() == b"end_header":
                break

        payload_offset = f.tell()

    header = header_bytes.decode("ascii")

    format_match = re.search(
        r"^format (\S+)",
        header,
        re.MULTILINE,
    )

    if not format_match:
        raise ValueError("PLY format not found")

    if format_match.group(1) != "binary_little_endian":
        raise ValueError(
            "Expected binary_little_endian PLY"
        )

    vertex_match = re.search(
        r"^element vertex (\d+)",
        header,
        re.MULTILINE,
    )

    if not vertex_match:
        raise ValueError(
            "PLY does not contain vertex element"
        )

    vertex_count = int(vertex_match.group(1))

    vertex_section = header[
        header.index("element vertex"):
        header.index("end_header")
    ]

    properties = re.findall(
        r"^property "
        r"(float|double|uchar|char|ushort|short|uint|int) "
        r"(\S+)",
        vertex_section,
        re.MULTILINE,
    )

    if not properties:
        raise ValueError(
            "Could not read PLY vertex properties"
        )

    type_map = {
        "float": "<f4",
        "double": "<f8",
        "uchar": "u1",
        "char": "i1",
        "ushort": "<u2",
        "short": "<i2",
        "uint": "<u4",
        "int": "<i4",
    }

    dtype = np.dtype([
        (name, type_map[type_name])
        for type_name, name in properties
    ])

    vertices = np.fromfile(
        path,
        dtype=dtype,
        count=vertex_count,
        offset=payload_offset,
    )

    required = {
        "x",
        "y",
        "z",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
    }

    names = set(vertices.dtype.names or [])

    missing = required - names

    if missing:
        raise ValueError(
            f"PLY is missing fields: {sorted(missing)}"
        )

    return vertices


# ------------------------------------------------------------
# LOAD GAUSSIANS
# ------------------------------------------------------------

def load_gaussians(path: Path, opacity_threshold: float):
    data = read_binary_ply(path)

    positions = np.column_stack(
        [
            data["x"],
            data["y"],
            data["z"],
        ]
    ).astype(np.float32)

    # Nerfstudio stores Gaussian scales in log space.
    scale_log = np.column_stack(
        [
            data["scale_0"],
            data["scale_1"],
            data["scale_2"],
        ]
    ).astype(np.float32)

    scales = np.exp(scale_log)

    # Quaternion:
    # gsplat expects WXYZ.
    quats = np.column_stack(
        [
            data["rot_0"],
            data["rot_1"],
            data["rot_2"],
            data["rot_3"],
        ]
    ).astype(np.float32)

    # Normalize quaternions.
    quat_norm = np.linalg.norm(
        quats,
        axis=1,
        keepdims=True,
    )

    quats = quats / np.maximum(quat_norm, 1e-8)

    # Nerfstudio stores opacity as a logit.
    opacity = 1.0 / (
        1.0 + np.exp(
            -data["opacity"].astype(np.float32)
        )
    )

    # Filter invalid / transparent Gaussians.
    keep = (
        np.isfinite(positions).all(axis=1)
        & np.isfinite(scales).all(axis=1)
        & np.isfinite(quats).all(axis=1)
        & np.isfinite(opacity)
        & (opacity >= opacity_threshold)
    )

    positions = positions[keep]
    scales = scales[keep]
    quats = quats[keep]
    opacity = opacity[keep]

    # --------------------------------------------------------
    # Spherical Harmonics
    #
    # 45 f_rest values =
    # 15 SH coefficients × 3 color channels.
    #
    # Plus 3 DC coefficients = 16 coefficients/channel.
    # Therefore degree = 3.
    # --------------------------------------------------------

    dc = np.column_stack(
        [
            data["f_dc_0"],
            data["f_dc_1"],
            data["f_dc_2"],
        ]
    ).astype(np.float32)

    rest_names = [
        name
        for name in data.dtype.names
        if name.startswith("f_rest_")
    ]

    rest_names.sort(
        key=lambda name: int(
            name.split("_")[-1]
        )
    )

    rest = np.column_stack(
        [
            data[name]
            for name in rest_names
        ]
    ).astype(np.float32)

    # 45 values = 15 coefficients × RGB.
    if rest.shape[1] != 45:
        raise ValueError(
            f"Expected 45 f_rest values, "
            f"got {rest.shape[1]}"
        )

    rest = rest.reshape(
        -1,
        15,
        3,
    )

    # [DC, SH1...SH15]
    sh = np.concatenate(
        [
            dc[:, None, :],
            rest,
        ],
        axis=1,
    )

    sh = sh[keep]

    print(
        f"Loaded {len(data):,} Gaussians"
    )

    print(
        f"Visible: {len(positions):,}"
    )

    print(
        f"SH degree: 3"
    )

    return (
        positions,
        quats,
        scales,
        opacity,
        sh,
    )


# ------------------------------------------------------------
# CAMERA
# ------------------------------------------------------------

def look_at(
    camera_position,
    target,
    up=np.array([0.0, 0.0, 1.0]),
):
    """
    Create a world-to-camera matrix.

    Camera convention:
        X = right
        Y = down
        Z = forward
    """

    camera_position = np.asarray(
        camera_position,
        dtype=np.float32,
    )

    target = np.asarray(
        target,
        dtype=np.float32,
    )

    up = np.asarray(
        up,
        dtype=np.float32,
    )

    forward = target - camera_position
    forward /= np.linalg.norm(forward)

    right = np.cross(
        forward,
        up,
    )

    right /= np.linalg.norm(right)

    true_up = np.cross(
        right,
        forward,
    )

    R = np.stack(
        [
            right,
            -true_up,
            forward,
        ],
        axis=0,
    )

    t = -R @ camera_position

    view = np.eye(
        4,
        dtype=np.float32,
    )

    view[:3, :3] = R
    view[:3, 3] = t

    return view


def mouse_callback(event, x, y, flags, camera):
    """Update orbit camera state from OpenCV mouse input."""

    previous = camera["mouse_position"]

    if event == camera["cv2"].EVENT_LBUTTONDOWN:
        camera["rotate_active"] = True
        camera["mouse_position"] = (x, y)
        return

    if event == camera["cv2"].EVENT_MBUTTONDOWN:
        camera["pan_active"] = True
        camera["mouse_position"] = (x, y)
        return

    if event == camera["cv2"].EVENT_LBUTTONUP:
        camera["rotate_active"] = False
        return

    if event == camera["cv2"].EVENT_MBUTTONUP:
        camera["pan_active"] = False
        return

    if event == camera["cv2"].EVENT_MOUSEWHEEL:
        camera["distance"] *= 0.9 ** (1 if flags > 0 else -1)
        camera["distance"] = float(
            np.clip(
                camera["distance"],
                camera["min_distance"],
                camera["max_distance"],
            )
        )
        camera["dirty"] = True
        return

    if event != camera["cv2"].EVENT_MOUSEMOVE:
        return

    dx = x - previous[0]
    dy = y - previous[1]
    camera["mouse_position"] = (x, y)

    if camera["rotate_active"]:
        camera["azimuth"] -= dx * 0.01
        camera["elevation"] = float(
            np.clip(
                camera["elevation"] + dy * 0.01,
                -math.radians(89.0),
                math.radians(89.0),
            )
        )
        camera["dirty"] = True

    if camera["pan_active"]:
        distance_scale = camera["distance"] * 0.0015
        camera["pan"][0] -= dx * distance_scale
        camera["pan"][1] += dy * distance_scale
        camera["dirty"] = True


# ------------------------------------------------------------
# RENDER
# ------------------------------------------------------------

def render(
    positions,
    quats,
    scales,
    opacity,
    sh,
    camera_position,
    target,
    width,
    height,
    fov,
    device,
):
    means = torch.from_numpy(
        positions
    ).to(device)

    quats_t = torch.from_numpy(
        quats
    ).to(device)

    scales_t = torch.from_numpy(
        scales
    ).to(device)

    opacity_t = torch.from_numpy(
        opacity
    ).to(device)

    sh_t = torch.from_numpy(
        sh
    ).to(device)

    viewmat = torch.from_numpy(
        look_at(
            camera_position,
            target,
        )
    ).to(device)

    # Pinhole camera.
    focal = (
        0.5
        * width
        / np.tan(
            np.deg2rad(fov) / 2.0
        )
    )

    K = torch.tensor(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    K = K.unsqueeze(0)

    viewmat = viewmat.unsqueeze(0)

    with torch.no_grad():
        colors, alphas, _ = rasterization(
            means=means,
            quats=quats_t,
            scales=scales_t,
            opacities=opacity_t,
            colors=sh_t,
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            sh_degree=3,
            packed=True,
            rasterize_mode="antialiased",
            backgrounds=torch.ones(
                (3,),
                dtype=torch.float32,
                device=device,
            ),
        )

    image = colors[0].clamp(
        0.0,
        1.0,
    )

    image = (
        image.cpu()
        .numpy()
        * 255
    ).astype(np.uint8)

    return image


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "ply",
        nargs="?",
        type=Path,
        default=DEFAULT_PLY,
    )

    parser.add_argument(
        "--opacity-threshold",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=750,
    )

    parser.add_argument(
        "--fov",
        type=float,
        default=60.0,
    )

    args = parser.parse_args()

    ply_path = (
        args.ply
        .expanduser()
        .resolve()
    )

    if not ply_path.is_file():
        raise SystemExit(
            f"PLY not found:\n{ply_path}"
        )

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU is required for this "
            "gsplat viewer."
        )

    device = torch.device("cuda")

    (
        positions,
        quats,
        scales,
        opacity,
        sh,
    ) = load_gaussians(
        ply_path,
        args.opacity_threshold,
    )

    center = np.median(
        positions,
        axis=0,
    )

    extent = (
        np.percentile(
            positions,
            98,
            axis=0,
        )
        - np.percentile(
            positions,
            2,
            axis=0,
        )
    )

    radius = float(
        np.max(extent)
    )

    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0

    initial_offset = np.array(
        [
            0.0,
            -2.5 * radius,
            0.8 * radius,
        ],
        dtype=np.float32,
    )

    horizontal_distance = math.hypot(
        initial_offset[0],
        initial_offset[1],
    )

    camera = {
        "cv2": None,
        "target": center.copy(),
        "distance": float(np.linalg.norm(initial_offset)),
        "min_distance": max(radius * 0.05, 1e-4),
        "max_distance": max(radius * 100.0, 1.0),
        "azimuth": math.atan2(
            initial_offset[1],
            initial_offset[0],
        ),
        "elevation": math.atan2(
            initial_offset[2],
            horizontal_distance,
        ),
        "pan": np.zeros(2, dtype=np.float32),
        "mouse_position": (0, 0),
        "rotate_active": False,
        "pan_active": False,
        "dirty": True,
    }

    print()
    print("==========================================")
    print("Standalone Gaussian Splat Viewer")
    print("==========================================")
    print(f"PLY:       {ply_path}")
    print(f"Gaussians: {len(positions):,}")
    print(f"GPU:       {torch.cuda.get_device_name(0)}")
    print("Renderer:  gsplat")
    print()
    print("Rendering initial view...")
    print()

    # Display the rendered image using OpenCV.
    try:
        import cv2
    except ImportError:
        raise SystemExit(
            "OpenCV is required.\n"
            "Install it with:\n"
            "pip install opencv-python"
        )

    camera["cv2"] = cv2
    window_name = "Bamboo Tray - Gaussian Splat"

    def camera_pose():
        horizontal = camera["distance"] * math.cos(
            camera["elevation"]
        )
        camera_position = camera["target"] + np.array(
            [
                horizontal * math.cos(camera["azimuth"]),
                horizontal * math.sin(camera["azimuth"]),
                camera["distance"] * math.sin(camera["elevation"]),
            ],
            dtype=np.float32,
        )

        target = camera["target"].copy()
        target[:2] += camera["pan"]
        camera_position[:2] += camera["pan"]
        return camera_position, target

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        window_name,
        args.width,
        args.height,
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
        camera,
    )

    image = None

    while True:
        if camera["dirty"]:
            camera_position, target = camera_pose()
            image = render(
                positions,
                quats,
                scales,
                opacity,
                sh,
                camera_position,
                target,
                args.width,
                args.height,
                args.fov,
                device,
            )
            camera["dirty"] = False

        cv2.imshow(
            window_name,
            cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR,
            ),
        )

        key = cv2.waitKey(30) & 0xFF

        if key == 27 or key == ord("q"):
            break

        if key == ord("r"):
            camera["target"] = center.copy()
            camera["distance"] = float(np.linalg.norm(initial_offset))
            camera["azimuth"] = math.atan2(
                initial_offset[1],
                initial_offset[0],
            )
            camera["elevation"] = math.atan2(
                initial_offset[2],
                horizontal_distance,
            )
            camera["pan"][:] = 0.0
            camera["dirty"] = True

        if key == ord("s"):
            output = (
                ROOT
                / "exports"
                / "splat"
                / "preview.png"
            )

            cv2.imwrite(
                str(output),
                cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2BGR,
                ),
            )

            print(
                f"Saved preview: {output}"
            )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()