"""Run a masked full COLMAP pipeline if the `colmap` command is installed.
Outputs a dense point cloud and, when possible, a Poisson mesh.
"""
from pathlib import Path
import argparse, subprocess, shutil, tempfile

parser=argparse.ArgumentParser()
parser.add_argument('--reuse', action='store_true', help='reuse the existing sparse models')
args=parser.parse_args()

if shutil.which('colmap') is None:
    raise SystemExit('COLMAP is not installed or not on PATH. Install COLMAP on your Mac, then rerun this script.')
root=Path(__file__).resolve().parent
images=root/'images'/'new_images_sam2'
masks=root/'masks'/'sam2'
ws=root/'colmap_workspace'; ws.mkdir(exist_ok=True)
db=ws/'database.db'; sparse=ws/'sparse'; dense=ws/'dense'; sparse.mkdir(exist_ok=True)

if (not images.is_dir() or not any(images.rglob('*.png'))
    or not masks.is_dir() or not any(masks.rglob('*.png'))):
    raise SystemExit('No masked PNG images found. Run auto_sam2.py first.')

def run(*args):
    print('\n>', ' '.join(map(str,args))); subprocess.run(list(map(str,args)),check=True)

if not args.reuse:
    if db.exists(): db.unlink()
    run('colmap','feature_extractor','--database_path',db,'--image_path',images,
        '--ImageReader.mask_path',masks,'--ImageReader.single_camera','1')
    run('colmap','sequential_matcher','--database_path',db)
    run('colmap','mapper','--database_path',db,'--image_path',images,'--output_path',sparse)
models=sorted(path for path in sparse.iterdir() if path.is_dir())
if not models: raise SystemExit('COLMAP could not create a sparse model.')

def model_score(path):
    """Return (registered images, 3D points) for a COLMAP model."""
    with tempfile.TemporaryDirectory() as tmp:
        run('colmap','model_converter','--input_path',path,
            '--output_path',tmp,'--output_type','TXT')
        converted=Path(tmp)
        images=sum(1 for line in (converted/'images.txt').read_text().splitlines()
                   if line and not line.startswith('#')) // 2
        points=sum(1 for line in (converted/'points3D.txt').read_text().splitlines()
                   if line and not line.startswith('#'))
        return images, points

scored=[(model_score(path), path) for path in models]
score,model=max(scored)
print(f'Using sparse model {model.name}: {score[0]} images, {score[1]} points')
if score[1] == 0:
    raise SystemExit('COLMAP models contain no 3D points. Recapture the object without moving the background or hand.')

sparse_ply=root/'results'/'colmap_sparse_tray.ply'
run('colmap','model_converter','--input_path',model,
    '--output_path',sparse_ply,'--output_type','PLY')

if dense.exists(): shutil.rmtree(dense)
run('colmap','image_undistorter','--image_path',images,'--input_path',model,'--output_path',dense,'--output_type','COLMAP')
try:
    run('colmap','patch_match_stereo','--workspace_path',dense,'--workspace_format','COLMAP','--PatchMatchStereo.geom_consistency','true')
except subprocess.CalledProcessError:
    print('\nDense reconstruction is unavailable (this COLMAP build may require CUDA).')
    print('Created the best sparse fallback:', sparse_ply)
    raise SystemExit(0)
fused=root/'results'/'dense_tray.ply'
run('colmap','stereo_fusion','--workspace_path',dense,'--workspace_format','COLMAP','--input_type','geometric','--output_path',fused)
mesh=root/'results'/'tray_mesh_poisson.ply'
try:
    run('colmap','poisson_mesher','--input_path',fused,'--output_path',mesh)
except subprocess.CalledProcessError:
    print('Poisson meshing failed, but dense point cloud was created:',fused)
print('\nDone. Check results/.')
