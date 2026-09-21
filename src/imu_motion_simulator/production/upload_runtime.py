"""Operational upload settings, deliberately separate from frozen job identity."""
from __future__ import annotations

import json
from pathlib import Path


SCHEMA = "imu_motion_simulator.upload_runtime.v1"
FIELDS = {"schema", "mode", "workers", "max_pending_bytes", "min_free_bytes"}


def load_upload_profile(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or set(value) != FIELDS
            or value["schema"] != SCHEMA or value["mode"] != "durable"
            or type(value["workers"]) is not int or not 1 <= value["workers"] <= 8
            or type(value["max_pending_bytes"]) is not int
            or value["max_pending_bytes"] < 1
            or type(value["min_free_bytes"]) is not int
            or value["min_free_bytes"] < 0):
        raise ValueError("Invalid operational upload profile")
    return value
