"""Validation and path resolution for kinematic production plans."""

from __future__ import annotations

import json
from pathlib import Path


SCHEMAS = {
    "imu_motion_simulator.kinematic_plan.v1",
    "imu_motion_simulator.kinematic_plan.v2",
}


def relative(value: str, field: str) -> str:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{field} must be a contained relative path")
    return str(path)


def resolve_inside(root: Path, relative_path: str) -> Path:
    root = Path(root).resolve()
    target = (root / relative(relative_path, "path")).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Resolved path escaped its root")
    return target


def load_plan(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    schema = value.get("schema")
    if schema not in SCHEMAS:
        raise ValueError("Only kinematic production plans are supported")
    required = {
        "schema", "study_id", "description", "inputs", "sensor",
        "review", "clips",
    }
    if schema == "imu_motion_simulator.kinematic_plan.v2":
        required.add("source_adapter")
    if set(value) != required or not value["study_id"] or not value["description"]:
        raise ValueError("Invalid kinematic pipeline plan fields")
    if schema.endswith(".v2") and value["source_adapter"] != {
        "id": "stageii-smplh-v1",
        "member_suffix": "_stageii.npz",
        "dynamic_shape": "disabled-zero",
    }:
        raise ValueError("Unsupported kinematic source adapter")
    if set(value["inputs"]) != {"amass_archive", "smplh_archive", "dmpl_archive"}:
        raise ValueError("Kinematic input fields changed")
    for key, input_path in value["inputs"].items():
        relative(input_path, "inputs." + key)
    if set(value["sensor"]) != {"layout", "profile"}:
        raise ValueError("Kinematic sensor fields changed")
    for key, sensor_path in value["sensor"].items():
        relative(sensor_path, "sensor." + key)
    if value["review"] != {
        "primary": "threejs",
        "mp4": "on-demand",
        "policy": "automatic-all-risk-stratified-human",
    }:
        raise ValueError("Kinematic review policy changed")
    if not isinstance(value["clips"], list) or not value["clips"]:
        raise ValueError("Kinematic clips are required")

    fields = {
        "id", "source_dataset", "source_member", "expected_gender",
        "expected_frames", "frame_range", "label_candidates",
    }
    ids: set[str] = set()
    source_members: set[str] = set()
    for clip in value["clips"]:
        frame_range = clip.get("frame_range")
        if (
            set(clip) != fields
            or not clip["id"]
            or clip["id"] in ids
            or clip["source_member"] in source_members
            or not clip["source_dataset"]
            or clip["expected_gender"] not in {"male", "female", "neutral"}
            or type(clip["expected_frames"]) is not int
            or clip["expected_frames"] < 2
        ):
            raise ValueError("Invalid kinematic clip")
        relative(clip["source_member"], "source_member")
        if (
            not isinstance(frame_range, list)
            or len(frame_range) != 2
            or any(type(frame) is not int for frame in frame_range)
            or not 0 <= frame_range[0] < frame_range[1] <= clip["expected_frames"]
        ):
            raise ValueError("Invalid kinematic frame range")
        if not isinstance(clip["label_candidates"], list):
            raise ValueError("Invalid kinematic label candidates")
        ids.add(clip["id"])
        source_members.add(clip["source_member"])
    return value


def describe_plan(path: Path) -> dict:
    value = load_plan(path)
    return {
        "valid": True,
        "mode": "kinematic",
        "study_id": value["study_id"],
        "clips": len(value["clips"]),
        "sensor_layout": value["sensor"]["layout"],
        "review": value["review"]["primary"],
    }
