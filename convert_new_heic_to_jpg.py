"""Convert photos in ``heic_originals/new images`` to JPG files."""
from pathlib import Path
import platform
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "heic_originals" / "new images"
OUTPUT_DIR = ROOT / "images" / "new_images"


def main():
    if platform.system() != "Darwin" or shutil.which("sips") is None:
        raise SystemExit("This converter requires the macOS 'sips' command.")
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"Source folder not found: {SOURCE_DIR}")

    photos = sorted(
        path for path in SOURCE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".heic", ".heif"}
    )
    if not photos:
        raise SystemExit(f"No HEIC photos found in: {SOURCE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    converted = 0
    for source in photos:
        destination = OUTPUT_DIR / f"{source.stem}.jpg"
        subprocess.run(
            [
                "sips",
                "-s", "format", "jpeg",
                "-s", "formatOptions", "best",
                str(source),
                "--out", str(destination),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        converted += 1
        print(f"{source.name} -> {destination.name}")

    print(f"Converted {converted} photos.")
    print(f"JPG output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
