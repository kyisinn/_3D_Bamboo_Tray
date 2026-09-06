# Chapter 14 — Image-Based Rendering: Bamboo Tray

This project uses the 36 HEIC photographs from your uploaded `Archive 2.zip`.

## What it produces

**Chapter 14 image-based rendering outputs:** light-field-style multi-view visualization, approximate depth representation, depth-based virtual viewpoints, view interpolation, feature matching, aligned interpolation, and a smooth multi-view animation.

**3D outputs:** an educational OpenCV sparse point cloud (`sparse_tray.ply`) and an optional stronger COLMAP dense point cloud / mesh (`dense_tray.ply`, `tray_mesh_poisson.ply`).

## 1. Setup on macOS

```bash
cd Chapter14_3D_Bamboo_Tray
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Convert the included HEIC photographs to JPG

```bash
python preprocess_heic.py
```

The script uses macOS `sips` and creates `images/view_001.jpg` ... in filename order.

## 3. Run the Chapter 14 IBR experiments

```bash
python main.py
```

Results are saved in `results/`:
- `01_light_field.jpg`
- `02_depth_map.jpg`
- `03_virtual_viewpoints.jpg`
- `04_view_interpolation.jpg`
- `05_feature_matching.jpg`
- `06_aligned_interpolation.jpg`
- `07_multiview_animation.mp4`
- `08_final_comparison.jpg`

## 4. Create an educational sparse 3D result

```bash
python reconstruct_sparse_3d.py
```

Output: `results/sparse_tray.ply`.

View it interactively with:

```bash
python view_3d.py
```

The viewer automatically opens the best available result in this order: the
Poisson mesh, dense COLMAP cloud, COLMAP sparse cloud, then the educational
sparse cloud. On systems where COLMAP dense reconstruction requires unavailable
CUDA support, the script exits successfully after creating the COLMAP sparse
fallback.

You can also open `.ply` files in MeshLab, Blender, CloudCompare, or another 3D viewer.

## Single-photo 3D rendering

A front-facing photo can be converted into a textured, shallow 2.5D mesh:

```bash
python single_image_3d.py /path/to/photo.HEIC
python view_3d.py
```

This produces a rotatable front, rim, and back, but cannot recover details that
are hidden in the one source photograph.

For an ordered full rotation, generate a two-sided mesh using all photographs:

```bash
python multiview_3d.py /path/to/IMG_*.HEIC
python view_3d.py
```

## 5. Optional: stronger dense 3D reconstruction with COLMAP

After installing COLMAP and ensuring the `colmap` command works:

```bash
python run_colmap.py
```

If sparse models already exist and only the dense stage needs to be rerun, use
`python run_colmap.py --reuse`.

Expected outputs include:
- `results/dense_tray.ply`
- `results/tray_mesh_poisson.ply` (if meshing succeeds)

## Important dataset note

Your uploaded photographs contain a moving hand and the tray appears to be hand-held. This is acceptable for the Chapter 14 interpolation demonstrations, but it can reduce 3D reconstruction quality because a true Structure-from-Motion pipeline assumes the scene/object is static while the camera viewpoint changes. If the dense 3D model is weak, recapture a 360° set with the tray fixed in place and move only the camera.
