import open3d as o3d

# Load the reconstructed point cloud
pcd = o3d.io.read_point_cloud("results/sparse_tray.ply")

print("3D point cloud loaded")
print("Number of points:", len(pcd.points))

# Open interactive 3D viewer
o3d.visualization.draw_geometries(
    [pcd],
    window_name="Bamboo Tray - 3D Reconstruction",
    width=1000,
    height=800
)