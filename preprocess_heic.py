from pathlib import Path
import platform, subprocess, shutil, sys

SRC = Path('heic_originals')
DST = Path('images')
DST.mkdir(exist_ok=True)
files = sorted(SRC.glob('*.HEIC'))
if not files:
    raise SystemExit('No HEIC files found in heic_originals/')

if platform.system() != 'Darwin' or shutil.which('sips') is None:
    raise SystemExit('This converter uses macOS sips. Run this project on your Mac, or export HEIC files to JPG manually.')

for i, src in enumerate(files, 1):
    dst = DST / f'view_{i:03d}.jpg'
    subprocess.run(['sips', '-s', 'format', 'jpeg', str(src), '--out', str(dst)], check=True,
                   stdout=subprocess.DEVNULL)
    print(f'{src.name} -> {dst.name}')
print(f'Converted {len(files)} images.')
