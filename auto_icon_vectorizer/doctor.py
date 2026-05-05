"""Runtime readiness checks for auto-icon-vectorizer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = ROOT / "runtime"
CHECKPOINTS = [
    RUNTIME_DIR / "nn-seg-results" / "best-gated-unet.pt",
    RUNTIME_DIR / "nn-seg-results" / "best-filled-silhouette-unet.pt",
]
OPTIONAL_TRAINING_ARTIFACTS = [
    RUNTIME_DIR / "nn-seg-results" / "feature-cache-v2.npz",
    RUNTIME_DIR / "nn-seg-results" / "filled-silhouette-feature-cache-v1.npz",
    RUNTIME_DIR / "truth-stress-eval" / "latest-run.json",
    RUNTIME_DIR / "nn-training-corpus-v2" / "latest-run.json",
]
PYTHON_IMPORTS = ["PIL", "cv2", "numpy", "torch", "cairosvg", "skimage"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether auto-icon-vectorizer is ready to run.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable status.")
    args = parser.parse_args()

    status = build_status()
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print_report(status)
    if not status["ready"]:
        raise SystemExit(1)


def build_status() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    for name in PYTHON_IMPORTS:
        checks.append(
            {
                "name": f"python import: {name}",
                "required": True,
                "ok": importlib.util.find_spec(name) is not None,
                "fix": "Install Python dependencies with: python3 -m pip install -e .",
            }
        )

    checks.extend(
        [
            {
                "name": "node executable",
                "required": True,
                "ok": shutil.which("node") is not None,
                "fix": "Install Node.js.",
            },
            {
                "name": "npm executable",
                "required": True,
                "ok": shutil.which("npm") is not None,
                "fix": "Install npm, usually bundled with Node.js.",
            },
            {
                "name": "runtime package.json",
                "required": True,
                "ok": (RUNTIME_DIR / "package.json").exists(),
                "fix": "Use a full source checkout or reinstall the package.",
            },
            {
                "name": "node_modules/potrace",
                "required": True,
                "ok": (RUNTIME_DIR / "node_modules" / "potrace").exists(),
                "fix": "Run: python3 -m auto_icon_vectorizer.install_runtime",
            },
        ]
    )

    for checkpoint in CHECKPOINTS:
        checks.append(
            {
                "name": f"model checkpoint: {checkpoint.name}",
                "required": True,
                "ok": checkpoint.exists(),
                "path": str(checkpoint),
                "fix": "The public repo includes checkpoints. Re-clone or reinstall if this file is missing.",
            }
        )

    for artifact in OPTIONAL_TRAINING_ARTIFACTS:
        checks.append(
            {
                "name": f"optional training artifact: {artifact.relative_to(RUNTIME_DIR)}",
                "required": False,
                "ok": artifact.exists(),
                "path": str(artifact),
                "fix": "Only needed for retraining. See docs/RUNTIME_ASSETS.md.",
            }
        )

    required = [check for check in checks if check["required"]]
    return {
        "ready": all(bool(check["ok"]) for check in required),
        "runtimeDir": str(RUNTIME_DIR),
        "checks": checks,
    }


def print_report(status: dict[str, Any]) -> None:
    print("auto-icon-vectorizer runtime check")
    print(f"runtimeDir: {status['runtimeDir']}")
    print()
    for check in status["checks"]:
        marker = "OK" if check["ok"] else "MISSING"
        required = "required" if check["required"] else "optional"
        print(f"[{marker}] {check['name']} ({required})")
        if not check["ok"]:
            print(f"       fix: {check['fix']}")
    print()
    if status["ready"]:
        print("Ready for inference.")
    else:
        print("Not ready. Fix the missing required items above.")


if __name__ == "__main__":
    main()
