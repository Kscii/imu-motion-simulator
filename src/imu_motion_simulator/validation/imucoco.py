"""Read-only IMUCoCo pose/orientation reality baseline."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation

from ..contracts.common import require, sha256_file


OFFICIAL_REPOSITORY = 'https://github.com/cmusmashlab/IMUCoCo'
OFFICIAL_COMMIT = 'd7dd17c70abccfc63935ce9e25452c6d312df630'
RATE_HZ = 60.
PARENTS = np.asarray([
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
    16, 17, 18, 19, 20, 21], dtype=np.int64)
PILOT_PARTICIPANTS = ('P01', 'P05', 'P09', 'P12')
PILOT_FOCUSES = ('Upper', 'Lower', 'Torso')
PILOT_ACTIVITIES = ('Walking', 'Running', 'Squats')
PILOT_ABSENT = {('P09', 'Upper', 'Squats')}
DEVICES = ('wrist', 'pocket', 'ear', 'a1', 'a2', 'a3', 'a4', 'a5')

# Joint indices are the exact selected-vertex categories used by official
# utils/imu_config.py at OFFICIAL_COMMIT.  Left/right switches with dominance.
DEVICE_JOINTS = {
    'right': {
        'wrist': 19, 'pocket': 0, 'ear': 15,
        'Upper': (17, 17, 17, 19, 21),
        'Lower': (1, 1, 4, 4, 7),
        'Torso': (12, 9, 6, 3, 0)},
    'left': {
        'wrist': 18, 'pocket': 0, 'ear': 15,
        'Upper': (16, 16, 16, 18, 20),
        'Lower': (2, 2, 5, 5, 8),
        'Torso': (12, 9, 6, 3, 0)},
}


def pilot_takes():
    return tuple(
        (participant, focus, activity)
        for participant in PILOT_PARTICIPANTS
        for focus in PILOT_FOCUSES
        for activity in PILOT_ACTIVITIES
        if (participant, focus, activity) not in PILOT_ABSENT)


def all_paired_takes(archive):
    """List complete pose, IMU and calibration triplets in the release ZIP."""
    index = _member_index(archive)
    takes = set()
    for key in index:
        match = re.fullmatch(r'(P\d{2})/(Upper|Lower|Torso)_([^/]+)_imu\.npz', key)
        if match is None:
            continue
        participant, focus, activity = match.groups()
        stem = f'{participant}/{focus}_{activity}'
        if all(stem + suffix in index for suffix in (
                '.npz', '_imu.npz', '_calibration_lab.npz')):
            takes.add((participant, focus, activity))
    return tuple(sorted(takes))


def _member_index(archive):
    result = {}
    for name in archive.namelist():
        path = PurePosixPath(name)
        if path.name:
            result['/'.join(path.parts[-2:])] = name
            result[path.name] = name
    return result


def _npz(archive, member, *, pickle=False):
    with np.load(io.BytesIO(archive.read(member)), allow_pickle=pickle) as data:
        return {name: data[name] for name in data.files}


def _participant_info(archive):
    candidates = [name for name in archive.namelist()
                  if PurePosixPath(name).name == 'participant_info.csv']
    require(len(candidates) == 1, 'IMUCoCo participant_info.csv')
    text = archive.read(candidates[0]).decode('utf-8-sig')
    rows = csv.DictReader(io.StringIO(text))
    result = {}
    for row in rows:
        participant = row.get('participant_id', '').strip()
        hand = row.get('dominant_hand', '').strip().lower()
        require(participant and hand in {'left', 'right'},
                'IMUCoCo participant metadata')
        result[participant] = hand
    return result


def audit_imucoco(path):
    path = Path(path).resolve()
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        require(corrupt is None, 'IMUCoCo ZIP CRC failure: ' + str(corrupt))
        index = _member_index(archive)
        participants = _participant_info(archive)
        available, missing = [], []
        for participant, focus, activity in pilot_takes():
            stem = f'{participant}/{focus}_{activity}'
            required = [stem + suffix for suffix in (
                '.npz', '_imu.npz', '_calibration_lab.npz')]
            if all(member in index for member in required):
                available.append({'participant': participant, 'focus': focus,
                                  'activity': activity})
            else:
                missing.append(stem)
    return {
        'schema': 'imu_motion_simulator.imucoco_audit.v1',
        'archive': str(path), 'archive_sha256': sha256_file(path),
        'official_repository': OFFICIAL_REPOSITORY,
        'official_code_commit': OFFICIAL_COMMIT,
        'participants': len(participants),
        'pilot_expected_takes': len(pilot_takes()),
        'pilot_available_takes': len(available),
        'pilot_missing': missing, 'pilot': available,
        'passed': not missing}


def _global_rotations(local):
    local = np.asarray(local, dtype=np.float64)
    require(local.ndim == 4 and local.shape[1:] == (24, 3, 3)
            and np.isfinite(local).all(), 'IMUCoCo pose_local')
    result = np.empty_like(local)
    result[:, 0] = local[:, 0]
    for joint in range(1, 24):
        result[:, joint] = result[:, PARENTS[joint]] @ local[:, joint]
    return result


def _quaternion_matrix(wxyz):
    quaternion = np.asarray(wxyz, dtype=np.float64)
    require(quaternion.ndim == 2 and quaternion.shape[1] == 4
            and np.isfinite(quaternion).all(), 'IMUCoCo quaternion')
    norms = np.linalg.norm(quaternion, axis=1)
    require(np.all(np.abs(norms - 1) < 5e-3),
            'IMUCoCo quaternion norm')
    quaternion = quaternion / norms[:, None]
    return Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_matrix()


def _world_angular_velocity(matrices):
    matrices = np.asarray(matrices, dtype=np.float64)
    require(len(matrices) >= 3, 'IMUCoCo take too short')
    relative = matrices[2:] @ matrices[:-2].transpose(0, 2, 1)
    middle = Rotation.from_matrix(relative).as_rotvec() * (RATE_HZ / 2)
    first = Rotation.from_matrix(matrices[1] @ matrices[0].T).as_rotvec() * RATE_HZ
    last = Rotation.from_matrix(matrices[-1] @ matrices[-2].T).as_rotvec() * RATE_HZ
    return np.vstack((first, middle, last))


def _array_metrics(reference, estimate):
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    require(reference.shape == estimate.shape and reference.ndim == 2
            and reference.shape[1] == 3, 'metric vector shape')
    delta = estimate - reference
    norm = np.linalg.norm(delta, axis=1)
    x, y = reference.reshape(-1), estimate.reshape(-1)
    correlation = None
    if np.std(x) > 0 and np.std(y) > 0:
        correlation = float(np.corrcoef(x, y)[0, 1])
    return {
        'mae_vector_norm': float(np.mean(norm)),
        'rmse_per_component': float(np.sqrt(np.mean(delta ** 2))),
        'pearson_correlation': correlation,
        'p95_vector_norm': float(np.quantile(norm, .95)),
        'samples': len(reference)}


def _orientation_metrics(reference, estimate):
    relative = reference.transpose(0, 2, 1) @ estimate
    return _orientation_error_metrics(Rotation.from_matrix(relative).magnitude())


def _orientation_error_metrics(errors):
    errors = np.asarray(errors, dtype=np.float64)
    return {
        'mean_geodesic_deg': float(np.degrees(np.mean(errors))),
        'rmse_geodesic_deg': float(np.degrees(np.sqrt(np.mean(errors ** 2)))),
        'p95_geodesic_deg': float(np.degrees(np.quantile(errors, .95))),
        'samples': len(errors)}


def _device_joint(device, focus, dominant):
    mapping = DEVICE_JOINTS[dominant]
    if device in {'wrist', 'pocket', 'ear'}:
        return mapping[device]
    match = re.fullmatch(r'a([1-5])', device)
    require(match is not None, 'unknown IMUCoCo device: ' + device)
    return mapping[focus][int(match.group(1)) - 1]


def _take(archive, index, participant, focus, activity, dominant):
    stem = f'{participant}/{focus}_{activity}'
    pose = _npz(archive, index[stem + '.npz'])
    imu = _npz(archive, index[stem + '_imu.npz'], pickle=True)
    calibration = _npz(
        archive, index[stem + '_calibration_lab.npz'], pickle=True)
    global_pose = _global_rotations(pose['pose_local'])
    devices = [item.decode() if isinstance(item, bytes) else str(item)
               for item in np.asarray(imu['devices']).tolist()]
    require(set(devices) == set(DEVICES) and len(devices) == len(DEVICES),
            'IMUCoCo device set')
    declared_frames = int(np.asarray(imu['n_frames']).item())
    frames = min(len(global_pose), declared_frames)
    require(frames >= 3, 'IMUCoCo take too short')
    nav_to_model = np.asarray(calibration['R_nav2model'], dtype=np.float64)
    require(nav_to_model.shape == (3, 3)
            and np.isfinite(nav_to_model).all(), 'IMUCoCo R_nav2model')
    rows, orientation_errors, gyro_reference, gyro_estimate = [], [], [], []
    lag_errors = {lag: [] for lag in (-2, -1, 0, 1, 2)}
    for device in devices:
        measured = _quaternion_matrix(imu[device + '_quaternion'][:frames])
        bone_to_sensor = np.asarray(
            calibration[device + '_R_bone2sensor'], dtype=np.float64)
        require(bone_to_sensor.shape == (3, 3)
                and np.isfinite(bone_to_sensor).all(),
                'IMUCoCo R_bone2sensor')
        measured = nav_to_model @ measured @ bone_to_sensor
        predicted = global_pose[:frames, _device_joint(
            device, focus, dominant)]
        orientation = _orientation_metrics(measured, predicted)
        measured_gyro = _world_angular_velocity(measured)
        predicted_gyro = _world_angular_velocity(predicted)
        gyro = _array_metrics(measured_gyro, predicted_gyro)
        relative = measured.transpose(0, 2, 1) @ predicted
        orientation_errors.append(Rotation.from_matrix(relative).magnitude())
        for lag in lag_errors:
            if lag < 0:
                shifted_measured, shifted_predicted = measured[-lag:], predicted[:lag]
            elif lag > 0:
                shifted_measured, shifted_predicted = measured[:-lag], predicted[lag:]
            else:
                shifted_measured, shifted_predicted = measured, predicted
            relative_lag = shifted_measured.transpose(0, 2, 1) @ shifted_predicted
            lag_errors[lag].append(Rotation.from_matrix(relative_lag).magnitude())
        gyro_reference.append(measured_gyro); gyro_estimate.append(predicted_gyro)
        rows.append({'device': device,
                     'predicted_joint_index': _device_joint(
                         device, focus, dominant),
                     'orientation': orientation,
                     'angular_velocity_from_orientation_rad_s': gyro})
    return {
        'participant': participant, 'dominant_hand': dominant,
        'focus': focus, 'activity': activity, 'frames': frames,
        'duration_s': (frames - 1) / RATE_HZ, 'devices': rows,
    }, orientation_errors, gyro_reference, gyro_estimate, lag_errors


def benchmark_imucoco(path, output, *, takes=None, all_takes=False):
    """Compare OptiTrack pose to calibrated device orientation without extraction."""
    path, output = Path(path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if all_takes and takes is not None:
        raise ValueError('takes and all_takes are mutually exclusive')
    with zipfile.ZipFile(path) as archive:
        index = _member_index(archive)
        participants = _participant_info(archive)
        requested = tuple(all_paired_takes(archive) if all_takes else
                          pilot_takes() if takes is None else takes)
        rows, angle_chunks, gyro_ref, gyro_pred = [], [], [], []
        failures = []
        lag_sums = {lag: [0., 0] for lag in (-2, -1, 0, 1, 2)}
        grouped = {
            dimension: {} for dimension in (
                'participant', 'focus', 'activity', 'device')}
        for participant, focus, activity in requested:
            require(participant in participants,
                    'unknown IMUCoCo participant: ' + participant)
            stem = f'{participant}/{focus}_{activity}'
            require(all(stem + suffix in index for suffix in (
                '.npz', '_imu.npz', '_calibration_lab.npz')),
                'missing IMUCoCo take: ' + stem)
            try:
                row, angles, references, estimates, lags = _take(
                    archive, index, participant, focus, activity,
                    participants[participant])
            except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
                if not all_takes:
                    raise
                failures.append({'participant': participant, 'focus': focus,
                                 'activity': activity, 'error_type': type(error).__name__,
                                 'error': str(error)})
                continue
            rows.append(row); angle_chunks.extend(angles)
            gyro_ref.extend(references); gyro_pred.extend(estimates)
            for lag, chunks in lags.items():
                lag_sums[lag][0] += sum(float(np.sum(chunk)) for chunk in chunks)
                lag_sums[lag][1] += sum(len(chunk) for chunk in chunks)
            for device_row, angle, reference, estimate in zip(
                    row['devices'], angles, references, estimates):
                keys = {'participant': participant, 'focus': focus,
                        'activity': activity,
                        'device': device_row['device']}
                for dimension, key in keys.items():
                    entry = grouped[dimension].setdefault(
                        key, {'angles': [], 'references': [], 'estimates': []})
                    entry['angles'].append(angle)
                    entry['references'].append(reference)
                    entry['estimates'].append(estimate)
    require(bool(angle_chunks), 'No complete IMUCoCo takes could be benchmarked')
    all_angles = np.concatenate(angle_chunks)
    orientation = {
        'mean_geodesic_deg': float(np.degrees(np.mean(all_angles))),
        'rmse_geodesic_deg': float(np.degrees(
            np.sqrt(np.mean(all_angles ** 2)))),
        'p95_geodesic_deg': float(np.degrees(np.quantile(all_angles, .95))),
        'samples': len(all_angles)}
    gyro = _array_metrics(np.concatenate(gyro_ref), np.concatenate(gyro_pred))
    grouped_metrics = {}
    for dimension, entries in grouped.items():
        grouped_metrics[dimension] = {
            key: {
                'orientation': _orientation_error_metrics(
                    np.concatenate(value['angles'])),
                'angular_velocity_from_orientation_rad_s': _array_metrics(
                    np.concatenate(value['references']),
                    np.concatenate(value['estimates']))}
            for key, value in sorted(entries.items())}
    report = {
        'schema': 'imu_motion_simulator.imucoco_baseline.v1',
        'archive': str(path), 'archive_sha256': sha256_file(path),
        'official_repository': OFFICIAL_REPOSITORY,
        'official_code_commit': OFFICIAL_COMMIT,
        'sampling_rate_hz': RATE_HZ,
        'requested_takes': len(requested), 'completed_takes': len(rows),
        'failed_takes': failures,
        'code_sha256': sha256_file(__file__),
        'scope': 'bone-rigid orientation baseline against real Apple Watch orientation',
        'takes': rows, 'grouped': grouped_metrics,
        'time_offset_diagnostic': {
            'description': 'Orientation geodesic mean for integer frame shifts; diagnostic only, zero-shift baseline retained',
            'positive_lag_means_predicted_pose_later': True,
            'mean_geodesic_deg_by_lag_frames': {
                str(lag): float(np.degrees(total / count))
                for lag, (total, count) in lag_sums.items()}},
        'aggregate': {
            'orientation': orientation,
            'angular_velocity_from_orientation_rad_s': gyro,
            'raw_acceleration': {
                'status': 'blocked',
                'reason': ('official processing does not declare or convert '
                           'user_acceleration physical units')},
            'raw_rotation_rate': {
                'status': 'blocked',
                'reason': ('official processing does not establish the raw '
                           'rotation_rate frame transform used for comparison')}},
        'limitations': [
            'Pose and device streams follow the official frame-zero truncation rule.',
            'Angular velocity is differentiated from calibrated device orientation; it is not the raw gyroscope stream.',
            'This benchmark is scientific comparison evidence, not a production acceptance gate.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.partial')
    temporary.write_text(json.dumps(
        report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    return report
