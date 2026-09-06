"""Run a full COLMAP pipeline if the `colmap` command is installed.
Outputs a dense point cloud and, when possible, a Poisson mesh.
"""
from pathlib import Path
import subprocess, shutil, sys

if shutil.which('colmap') is None:
    raise SystemExit('COLMAP is not installed or not on PATH. Install COLMAP on your Mac, then rerun this script.')
root=Path.cwd(); images=root/'images'; ws=root/'colmap_workspace'; ws.mkdir(exist_ok=True)
db=ws/'database.db'; sparse=ws/'sparse'; dense=ws/'dense'; sparse.mkdir(exist_ok=True)

def run(*args):
    print('\n>', ' '.join(map(str,args))); subprocess.run(list(map(str,args)),check=True)

if db.exists(): db.unlink()
run('colmap','feature_extractor','--database_path',db,'--image_path',images,'--ImageReader.single_camera','1')
run('colmap','sequential_matcher','--database_path',db)
run('colmap','mapper','--database_path',db,'--image_path',images,'--output_path',sparse)
models=sorted(sparse.iterdir())
if not models: raise SystemExit('COLMAP could not create a sparse model.')
model=models[0]
run('colmap','image_undistorter','--image_path',images,'--input_path',model,'--output_path',dense,'--output_type','COLMAP')
run('colmap','patch_match_stereo','--workspace_path',dense,'--workspace_format','COLMAP','--PatchMatchStereo.geom_consistency','true')
fused=root/'results'/'dense_tray.ply'
run('colmap','stereo_fusion','--workspace_path',dense,'--workspace_format','COLMAP','--input_type','geometric','--output_path',fused)
mesh=root/'results'/'tray_mesh_poisson.ply'
try:
    run('colmap','poisson_mesher','--input_path',fused,'--output_path',mesh)
except subprocess.CalledProcessError:
    print('Poisson meshing failed, but dense point cloud was created:',fused)
print('\nDone. Check results/.')
