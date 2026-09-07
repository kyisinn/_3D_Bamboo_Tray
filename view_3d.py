"""Open the reconstructed bamboo tray in an interactive 3D viewer."""
from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CANDIDATES = (
    #RESULTS / "tray_mesh_poisson.ply",
    #RESULTS / "dense_tray.ply",
    #RESULTS / "multiview_tray_highres.ply",
    #RESULTS / "multiview_tray.ply",
    #RESULTS / "single_image_tray.ply",
    RESULTS / "colmap_sparse_tray.ply",
    #RESULTS / "sparse_tray.ply",
)


def main():
    geometry_path = next((path for path in CANDIDATES if path.is_file()), None)
    if geometry_path is None:
        raise SystemExit(
            f"No 3D result found in {RESULTS}\n"
            "Run run_colmap.py for a dense model, or reconstruct_sparse_3d.py "
            "for the basic sparse result."
        )

    mesh = o3d.io.read_triangle_mesh(str(geometry_path))
    if mesh.has_triangles():
        geometry = mesh
        geometry.compute_vertex_normals()
        kind = "mesh"
    else:
        geometry = o3d.io.read_point_cloud(str(geometry_path))
        kind = "point cloud"

    if geometry.is_empty():
        raise SystemExit(f"Open3D could not read any geometry from {geometry_path}")

    # Remove invalid/outlier points that can make the camera zoom so far out that
    # the reconstruction appears blank, then center and normalize the geometry.
    if kind == "point cloud":
        geometry.remove_non_finite_points()
        if len(geometry.points) >= 20:
            geometry, _ = geometry.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )

    points = np.asarray(
        geometry.vertices if kind == "mesh" else geometry.points
    )
    center = np.median(points, axis=0)
    points -= center
    extent = np.percentile(points, 98, axis=0) - np.percentile(points, 2, axis=0)
    scale = float(np.max(extent))
    if np.isfinite(scale) and scale > 0:
        points /= scale

    print(f"Loaded {kind}: {geometry_path.name} ({len(points):,} vertices)")
    print("Drag to rotate, scroll to zoom, and press R to reset the camera.")

    o3d.visualization.draw_geometries(
        [geometry],
        window_name="Bamboo Tray - 3D Reconstruction",
        width=1000,
        height=800,
        point_show_normal=False,
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()
