#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"
PLAN_PATH = REPO_ROOT / "submit_plan.md"


SLOTS = [
    {
        "role": "control_best_39_0",
        "file": OUTPUT_DIR / "submission_best_39_0.pkl",
        "meta": None,
        "kind": "control",
    },
    {
        "role": "larger_sample_inv_y_raw",
        "file": OUTPUT_DIR / "submission_larger_sample_inv_y_raw.pkl",
        "meta": OUTPUT_DIR / "candidate_validation_larger_sample.json",
        "kind": "raw",
    },
    {
        "role": "larger_sample_inv_y_raw + 5_group_scale",
        "file": OUTPUT_DIR / "submission_larger_sample_inv_y_5group_scale.pkl",
        "meta": OUTPUT_DIR / "candidate_validation_larger_sample.json",
        "kind": "group",
    },
    {
        "role": "larger_sample_inv_y_raw + Tet",
        "file": OUTPUT_DIR / "submission_larger_sample_tet_inv_y_raw.pkl",
        "meta": OUTPUT_DIR / "candidate_validation_larger_sample_tet.json",
        "kind": "raw",
    },
    {
        "role": "larger_sample_inv_y_raw + Tet + event_features/group_scale",
        "file": OUTPUT_DIR / "submission_larger_sample_tet_event_inv_y_5group_scale.pkl",
        "fallback_file": OUTPUT_DIR / "submission_larger_sample_tet_event_inv_y_raw.pkl",
        "meta": OUTPUT_DIR / "candidate_validation_larger_sample_tet_event.json",
        "kind": "group_or_raw",
    },
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_meta(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def slot_file(slot: dict, meta: dict) -> Path:
    path = slot["file"]
    if slot["kind"] == "group_or_raw" and not path.exists():
        fallback = slot.get("fallback_file")
        if fallback is not None:
            return fallback
    return path


def validation_text(slot: dict, meta: dict) -> list[str]:
    if slot["kind"] == "control":
        return [
            "   - public score: 39.0",
            "   - reason: locked control from current best; do not overwrite",
        ]
    if not meta:
        return [
            "   - validation MAPE: pending",
            "   - reason: candidate not generated yet",
        ]
    if slot["kind"] == "raw":
        global_params = meta.get("global", {})
        return [
            f"   - validation MAPE: {global_params.get('mape_quantity', float('nan')):.6f}",
            f"   - scale/postprocess: global scale {global_params.get('scale', 1.0):.3f}",
            "   - reason: new model/feature candidate, not random scale probing",
        ]
    refined = meta.get("refined_5_group", {})
    selected = bool(meta.get("selected_group_scale", meta.get("refined_beats_global", False)))
    scales = refined.get("group_scales", {})
    return [
        f"   - validation MAPE: {refined.get('mape_quantity', float('nan')):.6f}",
        f"   - selected group scale: {selected}",
        f"   - scales: `{json.dumps(scales, sort_keys=True)}`",
        "   - reason: 5-group scale kept only if validation beats raw",
    ]


def main() -> int:
    lines = [
        "# Submit Plan",
        "",
        "Only these 5 candidates are tracked. No random scale variants.",
        "",
    ]
    for idx, slot in enumerate(SLOTS, 1):
        meta = load_meta(slot["meta"])
        path = slot_file(slot, meta)
        lines.append(f"{idx}. `{rel(path)}`")
        lines.append(f"   - role: {slot['role']}")
        lines.extend(validation_text(slot, meta))
        if slot["meta"] is not None:
            lines.append(f"   - metadata: `{rel(slot['meta'])}`")
        lines.append("")

    PLAN_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {rel(PLAN_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
