"""Create the project-local Jupyter kernel spec for the current Python.

Run with the geoml interpreter:
    conda run -n geoml python tools/prepare_kernel.py
"""
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
KERNEL_DIR = PROJECT / ".jupyter" / "kernels" / "geoml"


def main() -> None:
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    prefix = os.environ.get("CONDA_PREFIX", str(Path(sys.executable).parents[1]))
    share_root = (Path(prefix) / "Library" / "share") if sys.platform == "win32" else (Path(prefix) / "share")
    spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python (geoml)",
        "language": "python",
        "env": {
            "CONDA_PREFIX": prefix,
            "GDAL_DATA": str(share_root / "gdal"),
            "PROJ_DATA": str(share_root / "proj"),
        },
    }
    (KERNEL_DIR / "kernel.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote", KERNEL_DIR / "kernel.json")
    print("python:", sys.executable)


if __name__ == "__main__":
    main()
