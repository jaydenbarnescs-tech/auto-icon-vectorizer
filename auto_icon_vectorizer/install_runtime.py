"""Install the Node Potrace dependency used by the vectorizer runtime."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"


def main() -> None:
    parser = argparse.ArgumentParser(description="Install Node dependencies for auto-icon-vectorizer.")
    parser.add_argument("--force", action="store_true", help="Run npm ci even if node_modules/potrace already exists.")
    args = parser.parse_args()

    if not args.force and (RUNTIME_DIR / "node_modules" / "potrace").exists():
        print(f"Node runtime already installed: {RUNTIME_DIR / 'node_modules'}")
        return
    if shutil.which("npm") is None:
        raise SystemExit("npm is required because Potrace is called through the Node package 'potrace'.")
    subprocess.run(["npm", "ci"], cwd=RUNTIME_DIR, check=True)
    print(f"Installed Node runtime: {RUNTIME_DIR / 'node_modules'}")


if __name__ == "__main__":
    main()
