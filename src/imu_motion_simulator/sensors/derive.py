"""Continuous-trajectory ideal and calibrated virtual IMU generation."""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

from ..contracts.common import sha256_file
from ..contracts.internal import artifact_parent, new_metadata, read_internal, write_internal
from ..motion.kinematics import joint_index, joint_transforms, load_model_member
from ..motion.selection import selection_slice
from .layout import load_layout, load_profile


def _producer():
    files = [Path(__file__), Path(__file__).with_name('layout.py'),
             Path(__file__).parents[1] / 'motion/kinematics.py']
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode()); digest.update(b'\0')
        digest.update(bytes.fromhex(sha256_file(path)))
    return {'name': 'imu_motion_simulator.sensors', 'version': '2.0.0',
            'code_sha256': digest.hexdigest()}


def sensor_recipe_sha256():
    return _producer()['code_sha256']


def _work_grid(duration, hz):
    count = int(np.floor(duration * hz + 1e-9)) + 1
    return np.arange(count, dtype=np.float64) / hz


def _angular_velocity_world(rotation):
    matrices = rotation.as_matrix(); count = len(matrices)
    if count < 3:
        raise ValueError('Trajectory is too short for angular velocity')
    relative = np.einsum('tij,tjk->tik', matrices[2:],
                         matrices[:-2].transpose(0, 2, 1))
    middle = Rotation.from_matrix(relative).as_rotvec() / 2
    first = Rotation.from_matrix(matrices[1] @ matrices[0].T).as_rotvec()
    last = Rotation.from_matrix(matrices[-1] @ matrices[-2].T).as_rotvec()
    return np.vstack((first, middle, last))


def _filter(values, work_hz, cutoff):
    sos = signal.butter(6, cutoff, btype='lowpass', fs=work_hz, output='sos')
    pad = 3 * (2 * len(sos) + 1)
    if len(values) <= pad:
        return signal.sosfilt(sos, values, axis=0,
                              zi=signal.sosfilt_zi(sos)[:, :, None]
                              * values[0][None, None, :])[0]
    return signal.sosfiltfilt(sos, values, axis=0)


def _trajectory(source_time, position, rotation, profile, *, work_hz=None):
    work_hz = profile['work_hz'] if work_hz is None else work_hz
    work_t = _work_grid(float(source_time[-1]), work_hz)
    output_t = _work_grid(float(source_time[-1]), profile['output_hz'])
    position_curve = CubicSpline(source_time, position, axis=0,
                                 bc_type='natural')
    acceleration_world = position_curve(work_t, 2)
    rotation_curve = RotationSpline(source_time, Rotation.from_matrix(rotation))
    work_rotation = rotation_curve(work_t)
    omega_world = _angular_velocity_world(work_rotation) * work_hz
    gravity = np.array([0., 0., -profile['gravity_m_s2']])
    specific_force = np.einsum(
        'tji,tj->ti', work_rotation.as_matrix(), acceleration_world - gravity)
    angular_velocity = np.einsum(
        'tji,tj->ti', work_rotation.as_matrix(), omega_world)
    combined = np.c_[specific_force, angular_velocity]
    # Explicit stationary endpoint continuation moves filtfilt's numerical
    # boundary far away from the requested motion.  Without it, the first and
    # last acceleration samples depend strongly on whether the work grid is
    # 240 or 480 Hz even when the interior has converged.
    padding = int(math.ceil(work_hz))
    padded = np.pad(combined, ((padding, padding), (0, 0)), mode='edge')
    combined = _filter(padded, work_hz, profile['lowpass_hz'])[
        padding:padding + len(combined)]
    sampled = np.stack([np.interp(output_t, work_t, combined[:, column])
                        for column in range(combined.shape[1])], axis=1)
    return output_t, sampled[:, :3], sampled[:, 3:]


def _resolved_work_hz(profile, motion_metadata):
    """Resolve the fixed or source-rate-aligned internal sampling grid."""
    minimum = float(profile['work_hz'])
    if profile.get('work_grid_policy') != 'source-rate-integer-multiple':
        return minimum
    source_hz = float(motion_metadata['kind_metadata']['source_fps_hz'])
    if not np.isfinite(source_hz) or source_hz <= 0:
        raise ValueError('Motion source rate is invalid')
    multiple = max(1, math.ceil(minimum / source_hz - 1e-12))
    return multiple * source_hz


def _motion_mount_trajectories(motion, model_archive, layout, selection=None, *,
                               model=None):
    description, metadata, arrays = read_internal(motion, 'motion')
    if description.get('motion_contract_version') != 2:
        raise ValueError('Kinematic sensors require canonical motion-v2')
    selected = slice(None); selection_value = None
    if selection is not None:
        selected, selection_value = selection_slice(selection, motion)
        arrays = {key: (value[selected] if value.ndim and len(value) == description['frames']
                        else value) for key, value in arrays.items()}
    time = arrays['time_ns'].astype(np.float64)
    time = (time - time[0]) / 1e9
    if model is None:
        model = load_model_member(
            model_archive, metadata['kind_metadata']['source_gender'])
    joint_position, joint_rotation = joint_transforms(
        model, arrays['betas'], arrays['root_position_m'],
        arrays['joint_local_quaternion_wxyz'])
    positions, rotations = [], []
    for mount in layout['mounts']:
        index = joint_index(mount['joint'])
        offset = np.asarray(mount['position_joint_m'], dtype=np.float64)
        mount_rotation = Rotation.from_quat(np.asarray(
            mount['quaternion_joint_from_sensor_wxyz'])[[1, 2, 3, 0]]).as_matrix()
        rotations.append(joint_rotation[:, index] @ mount_rotation)
        positions.append(joint_position[:, index] + np.einsum(
            'tij,j->ti', joint_rotation[:, index], offset))
    return description, metadata, arrays, time, positions, rotations, selection_value


def _derive_values(motion, model_archive, layout, profile, *, selection=None,
                   work_hz=None, model=None, trajectories=None):
    result = trajectories
    if result is None:
        result = _motion_mount_trajectories(
            motion, model_archive, layout, selection=selection, model=model)
    description, metadata, arrays, source_time, positions, rotations, selection_value = result
    work_hz = (_resolved_work_hz(profile, metadata)
               if work_hz is None else work_hz)
    forces, gyros, output_time = [], [], None
    for position, rotation in zip(positions, rotations):
        times, force, gyro = _trajectory(source_time, position, rotation,
                                         profile, work_hz=work_hz)
        if output_time is not None and not np.array_equal(times, output_time):
            raise ValueError('Sensor mount clocks differ')
        output_time = times; forces.append(force); gyros.append(gyro)
    return (description, metadata, arrays, selection_value, output_time,
            np.stack(forces, axis=1), np.stack(gyros, axis=1))


def convergence_report(motion, model_archive, layout_path, profile_path, *,
                       selection=None, model=None):
    layout, profile = load_layout(layout_path), load_profile(profile_path)
    trajectories = _motion_mount_trajectories(
        motion, model_archive, layout, selection=selection, model=model)
    work_hz = _resolved_work_hz(profile, trajectories[1])
    one = _derive_values(motion, model_archive, layout, profile,
                         selection=selection, work_hz=work_hz,
                         trajectories=trajectories)
    two = _derive_values(motion, model_archive, layout, profile,
                         selection=selection, work_hz=2 * work_hz,
                         trajectories=trajectories)
    force = np.linalg.norm(one[-2] - two[-2], axis=-1)
    gyro = np.linalg.norm(one[-1] - two[-1], axis=-1)
    boundary_guard_s = float(profile.get('boundary_guard_s', 0.))
    boundary_guard_samples = int(round(
        boundary_guard_s * profile['output_hz']))
    if len(force) <= 2 * boundary_guard_samples:
        raise ValueError('Sensor trajectory is shorter than its boundary guard')
    evaluated = (slice(boundary_guard_samples, len(force) - boundary_guard_samples)
                 if boundary_guard_samples else slice(None))
    force, gyro = force[evaluated], gyro[evaluated]
    return {
        'work_hz': work_hz, 'comparison_work_hz': 2 * work_hz,
        'boundary_guard_s': boundary_guard_s,
        'boundary_guard_samples': boundary_guard_samples,
        'evaluated_samples': len(force),
        'acceleration_rms_m_s2': float(np.sqrt(np.mean(force ** 2))),
        'acceleration_peak_m_s2': float(force.max(initial=0)),
        'angular_velocity_rms_rad_s': float(np.sqrt(np.mean(gyro ** 2))),
        'angular_velocity_peak_rad_s': float(gyro.max(initial=0)),
    }


def derive_ideal(motion, model_archive, layout_path, profile_path, output, *,
                 selection=None, model=None, model_archive_sha256=None):
    motion, output = Path(motion), Path(output)
    layout, profile = load_layout(layout_path), load_profile(profile_path)
    result = _derive_values(motion, model_archive, layout, profile,
                            selection=selection, model=model)
    _, motion_metadata, _, selection_value, output_time, force, gyro = result
    resolved_work_hz = _resolved_work_hz(profile, motion_metadata)
    rate = Fraction(profile['output_hz'], 1)
    times = np.rint(output_time * 1e9).astype(np.int64)
    valid = np.isfinite(force).all(axis=2) & np.isfinite(gyro).all(axis=2)
    info = motion_metadata['kind_metadata']; sensor_id = str(uuid4())
    arrays = {'time_ns': times, 'specific_force_m_s2': force,
              'angular_velocity_rad_s': gyro, 'valid': valid}
    metadata = new_metadata(
        producer=_producer(), parents=[artifact_parent(motion, 'motion')],
        kind_metadata={
            'sensor_contract_version': 2, 'sensor_id': sensor_id,
            'motion_id': info['motion_id'],
            'selection_id': None if selection_value is None else selection_value['selection_id'],
            'layout_id': layout['layout_id'],
            'mount_ids': [mount['id'] for mount in layout['mounts']],
            'profile_id': profile['profile_id'], 'variant': 'ideal'},
        provenance={
            'input_motion_sha256': sha256_file(motion),
            'input_model_archive_sha256': (model_archive_sha256
                                           or sha256_file(model_archive)),
            'layout_sha256': sha256_file(layout_path),
            'profile_sha256': sha256_file(profile_path),
            'selection_sha256': None if selection is None else sha256_file(selection),
            'limitations': ['Rigid sensor mounting',
                            'No device bias, noise, quantization or soft-tissue motion',
                            ('Endpoint values use stationary continuation; convergence QA '
                             'uses the profile boundary guard')]},
        clocks={'sensor': {'numerator': rate.denominator,
                           'denominator': rate.numerator,
                           'origin': 'motion-start'}},
        resolved_config={'layout': layout, 'profile': profile,
                         'resolved_work_hz': resolved_work_hz,
                         'surface_policy': 'bone-rigid'})
    return write_internal(output, 'sensors', metadata, arrays)


def derive_calibrated(ideal_sensors, calibration_profile_path, output, *, seed):
    description, source_metadata, arrays = read_internal(ideal_sensors, 'sensors')
    if description.get('sensor_contract_version') != 2 \
            or source_metadata['kind_metadata']['variant'] != 'ideal':
        raise ValueError('Calibration requires ideal sensor-v2 input')
    profile = load_profile(calibration_profile_path, calibrated=True)
    rng = np.random.default_rng(seed)
    calibrated = {}
    for name, key in [('specific_force_m_s2', 'acceleration'),
                      ('angular_velocity_rad_s', 'angular_velocity')]:
        spec = profile[key]
        calibrated[name] = (arrays[name] * np.asarray(spec['scale'])[None, None]
                            + np.asarray(spec['bias'])[None, None]
                            + rng.normal(0, np.asarray(spec['noise_std']), arrays[name].shape))
    calibrated['time_ns'] = arrays['time_ns'].copy()
    calibrated['valid'] = arrays['valid'].copy()
    info = dict(source_metadata['kind_metadata'])
    info.update(sensor_id=str(uuid4()), profile_id=profile['profile_id'],
                variant='calibrated')
    metadata = new_metadata(
        producer=_producer(), parents=[artifact_parent(ideal_sensors, 'sensors')],
        kind_metadata=info,
        provenance={'input_ideal_sha256': sha256_file(ideal_sensors),
                    'calibration_profile_sha256': sha256_file(calibration_profile_path),
                    'random_seed': int(seed),
                    'limitations': ['Engineering calibration profile; no device validity implied']},
        clocks=source_metadata['clocks'],
        resolved_config={'profile': profile, 'random_seed': int(seed)})
    return write_internal(output, 'sensors', metadata, calibrated)
