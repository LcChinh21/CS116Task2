#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [
    REPO_ROOT / "outputs/submission_best_39_0.pkl",
    REPO_ROOT / "outputs/submission_larger_sample_inv_y_raw.pkl",
    REPO_ROOT / "outputs/submission_larger_sample_inv_y_5group_scale.pkl",
    REPO_ROOT / "outputs/submission_larger_sample_tet_inv_y_raw.pkl",
    REPO_ROOT / "outputs/submission_larger_sample_tet_event_inv_y_5group_scale.pkl",
]


def main() -> int:
    status = 0
    for path in CANDIDATES:
        if not path.exists() and path.name.endswith("_5group_scale.pkl"):
            fallback = path.with_name(path.name.replace("_5group_scale.pkl", "_raw.pkl"))
            if fallback.exists():
                print(f"skip missing group-scale candidate, checking fallback raw: {fallback}")
                path = fallback
        if not path.exists():
            print(f"missing candidate: {path}", file=sys.stderr)
            status = 1
            continue
        result = subprocess.run([sys.executable, "src/check_submission.py", str(path)], cwd=REPO_ROOT)
        if result.returncode != 0:
            status = result.returncode
    return status


if __name__ == "__main__":
    raise SystemExit(main())
