"""Compatibility entry point for the kinematic production pipeline."""

from __future__ import annotations

from pathlib import Path

from .kinematic import run_kinematic_pipeline
from .plan import load_plan


def run_pipeline(
    plan_path: Path,
    library_root: Path,
    workspace: Path,
    checkout: Path,
    output: Path,
    *,
    source_artifact: Path | None = None,
    blender: Path | None = None,
    through: str | None = None,
    sensor_workers: int = 1,
) -> dict:
    """Run a kinematic plan while retaining the legacy call signature."""
    del workspace, blender
    plan_path = Path(plan_path).resolve()
    plan = load_plan(plan_path)
    stage = through or "review"
    if stage not in {"convert", "sensors", "review"}:
        raise ValueError("Unknown kinematic pipeline stage")
    return run_kinematic_pipeline(
        plan_path,
        plan,
        library_root,
        checkout,
        output,
        source_artifact=source_artifact,
        through=stage,
        sensor_workers=sensor_workers,
    )
