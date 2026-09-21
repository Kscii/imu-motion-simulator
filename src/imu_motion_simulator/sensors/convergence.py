"""Numerical convergence QA kept separate from immutable sensor derivation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..contracts.common import sha256_file
from .derive import (_derive_values, _motion_mount_trajectories,
                     _resolved_work_hz)
from .layout import load_layout, load_profile


def convergence_recipe_sha256():
    """Bind QA to its policy and to the trajectory implementation it compares."""
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name('derive.py')):
        digest.update(path.name.encode())
        digest.update(b'\0')
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def upgrade_legacy_convergence(value):
    """Upgrade reports that could only have existed with an evaluable guard."""
    required = {
        'work_hz', 'comparison_work_hz', 'acceleration_rms_m_s2',
        'acceleration_peak_m_s2', 'angular_velocity_rms_rad_s',
        'angular_velocity_peak_rad_s', 'boundary_guard_s',
        'boundary_guard_samples', 'evaluated_samples'}
    if not isinstance(value, dict) or not required <= set(value):
        return None
    if 'boundary_guard_complete' in value:
        return value
    return {**value, 'boundary_guard_complete': True}


def convergence_report(motion, model_archive, layout_path, profile_path, *,
                       selection=None, model=None):
    layout, profile = load_layout(layout_path), load_profile(profile_path)
    trajectories = _motion_mount_trajectories(
        motion, model_archive, layout, selection=selection, model=model)
    work_hz = _resolved_work_hz(profile, trajectories[1])
    one = _derive_values(
        motion, model_archive, layout, profile, selection=selection,
        work_hz=work_hz, trajectories=trajectories)
    two = _derive_values(
        motion, model_archive, layout, profile, selection=selection,
        work_hz=2 * work_hz, trajectories=trajectories)
    force = np.linalg.norm(one[-2] - two[-2], axis=-1)
    gyro = np.linalg.norm(one[-1] - two[-1], axis=-1)
    boundary_guard_s = float(profile.get('boundary_guard_s', 0.))
    boundary_guard_samples = int(round(
        boundary_guard_s * profile['output_hz']))
    boundary_guard_complete = len(force) > 2 * boundary_guard_samples
    evaluated = (slice(boundary_guard_samples, len(force) - boundary_guard_samples)
                 if boundary_guard_samples and boundary_guard_complete
                 else slice(None))
    force, gyro = force[evaluated], gyro[evaluated]
    return {
        'work_hz': work_hz, 'comparison_work_hz': 2 * work_hz,
        'boundary_guard_s': boundary_guard_s,
        'boundary_guard_samples': boundary_guard_samples,
        'boundary_guard_complete': boundary_guard_complete,
        'evaluated_samples': len(force),
        'acceleration_rms_m_s2': float(np.sqrt(np.mean(force ** 2))),
        'acceleration_peak_m_s2': float(force.max(initial=0)),
        'angular_velocity_rms_rad_s': float(np.sqrt(np.mean(gyro ** 2))),
        'angular_velocity_peak_rad_s': float(gyro.max(initial=0))}
