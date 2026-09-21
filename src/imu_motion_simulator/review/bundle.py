"""Self-contained kinematic review bundles backed by canonical motion-v2."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation

from ..contracts.common import sha256_file
from ..contracts.internal import read_internal
from ..motion.kinematics import load_model_member, shaped_rest
from ..motion.selection import selection_slice
from ..sensors.layout import load_layout


WEB = Path(__file__).with_name('web')


def _dynamic_shape(metadata):
    return metadata['resolved_config'].get('dynamic_shape', {
        'source_available': True, 'effective_policy': 'source',
        'components': 8})


def _atomic_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def _producer_hash():
    digest = hashlib.sha256()
    for path in (Path(__file__), WEB / 'app.js', WEB / 'index.html',
                 WEB / 'three.module.js', WEB / 'three.core.min.js'):
        digest.update(path.name.encode()); digest.update(b'\0')
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def review_recipe_sha256():
    return _producer_hash()


def automatic_qa(motion, sensors, *, selection=None, convergence=None):
    motion_description, motion_metadata, motion_arrays = read_internal(motion, 'motion')
    sensor_description, sensor_metadata, sensor_arrays = read_internal(sensors, 'sensors')
    if motion_description.get('motion_contract_version') != 2 \
            or sensor_description.get('sensor_contract_version') != 2:
        raise ValueError('Kinematic QA requires motion-v2 and sensors-v2')
    if sensor_metadata['kind_metadata']['motion_id'] \
            != motion_metadata['kind_metadata']['motion_id']:
        raise ValueError('Sensor/motion identity mismatch')
    selected = slice(None)
    if selection is not None:
        selected, value = selection_slice(selection, motion)
        if sensor_metadata['kind_metadata']['selection_id'] != value['selection_id']:
            raise ValueError('Sensor/selection identity mismatch')
    time = motion_arrays['time_ns'][selected] / 1e9
    root = motion_arrays['root_position_m'][selected]
    local = motion_arrays['joint_local_quaternion_wxyz'][selected]
    dt = np.diff(time)
    root_speed = np.linalg.norm(np.diff(root, axis=0) / dt[:, None], axis=1)
    matrices = Rotation.from_quat(
        local[..., [1, 2, 3, 0]].reshape(-1, 4)
    ).as_matrix().reshape(len(local), 52, 3, 3)
    relative = np.einsum('tjab,tjcb->tjac', matrices[1:], matrices[:-1])
    angular_step = np.linalg.norm(Rotation.from_matrix(
        relative.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 52, 3)
        / dt[:, None, None], axis=2)
    force = np.linalg.norm(sensor_arrays['specific_force_m_s2'], axis=2)
    gyro = np.linalg.norm(sensor_arrays['angular_velocity_rad_s'], axis=2)
    convergence = convergence or {}
    convergence_fields = {
        'work_hz', 'comparison_work_hz', 'acceleration_rms_m_s2',
        'acceleration_peak_m_s2', 'angular_velocity_rms_rad_s',
        'angular_velocity_peak_rad_s', 'boundary_guard_s',
        'boundary_guard_samples', 'boundary_guard_complete',
        'evaluated_samples'}
    convergence_pass = (
        convergence_fields <= set(convergence)
        and convergence['boundary_guard_complete']
        and convergence['acceleration_rms_m_s2'] <= .10
        and convergence['angular_velocity_rms_rad_s'] <= .02)
    checks = {
        'structure': {
            'pass': bool(motion_arrays['valid'][selected].all()
                         and sensor_arrays['valid'].all()
                         and sensor_description['complete']),
            'motion_frames': len(root), 'sensor_samples': len(force),
            'sensor_mounts': sensor_description['mounts']},
        'trajectory': {
            'pass': bool(np.isfinite(root_speed).all()
                         and np.isfinite(angular_step).all()),
            'root_speed_peak_m_s': float(root_speed.max(initial=0)),
            'joint_angular_speed_peak_rad_s': float(angular_step.max(initial=0))},
        'sensor': {
            'pass': bool(np.isfinite(force).all() and np.isfinite(gyro).all()),
            'specific_force_peak_m_s2': float(force.max(initial=0)),
            'angular_velocity_peak_rad_s': float(gyro.max(initial=0))},
        'work_grid_convergence': {
            'pass': bool(convergence_pass), **convergence},
    }
    warnings = []
    if checks['trajectory']['root_speed_peak_m_s'] > 12:
        warnings.append('root-speed-outlier')
    if checks['trajectory']['joint_angular_speed_peak_rad_s'] > 35:
        warnings.append('joint-speed-outlier')
    if checks['sensor']['specific_force_peak_m_s2'] > 100:
        warnings.append('specific-force-outlier')
    if checks['sensor']['angular_velocity_peak_rad_s'] > 25:
        warnings.append('angular-velocity-outlier')
    passed = all(check['pass'] for check in checks.values())
    risk = min(1., (checks['sensor']['specific_force_peak_m_s2'] / 100
                    + checks['sensor']['angular_velocity_peak_rad_s'] / 25
                    + .5 * len(warnings)) / 2)
    return {
        'schema': 'imu_motion_simulator.kinematic_qa.v1',
        'motion_id': motion_metadata['kind_metadata']['motion_id'],
        'sensor_id': sensor_metadata['kind_metadata']['sensor_id'],
        'passed': passed, 'risk_score': float(risk), 'warnings': warnings,
        'checks': checks,
        'scope': 'Automatic structural and numerical QA; human motion naturalness remains a review decision'}


def _write_array(path, array, dtype):
    values = np.asarray(array, dtype=np.dtype(dtype).newbyteorder('<'))
    path.write_bytes(values.tobytes(order='C'))
    return {'path': path.name, 'dtype': values.dtype.str,
            'shape': list(values.shape), 'sha256': sha256_file(path)}


def _model_payload(model, betas, destination):
    vertices, joints = shaped_rest(model, betas)
    weights = np.asarray(model['weights'], dtype=np.float64)
    top = np.argpartition(weights, -4, axis=1)[:, -4:]
    top_weight = np.take_along_axis(weights, top, axis=1)
    order = np.argsort(top_weight, axis=1)[:, ::-1]
    top = np.take_along_axis(top, order, axis=1)
    top_weight = np.take_along_axis(top_weight, order, axis=1)
    top_weight /= top_weight.sum(axis=1, keepdims=True)
    parents = np.asarray(model['kintree_table'])
    ids = [int(value) for value in parents[1]]
    lookup = {value: index for index, value in enumerate(ids)}
    parent_indices = np.asarray([-1] + [lookup[int(value)] for value in parents[0, 1:]],
                                dtype=np.int16)
    return {
        'vertices': _write_array(destination / 'vertices.bin', vertices, 'f4'),
        'faces': _write_array(destination / 'faces.bin', model['f'], 'u4'),
        'skin_indices': _write_array(destination / 'skin-indices.bin', top, 'u2'),
        'skin_weights': _write_array(destination / 'skin-weights.bin', top_weight, 'f4'),
        'rest_joints': _write_array(destination / 'rest-joints.bin', joints, 'f4'),
        'parents': parent_indices.tolist(),
    }


def build_bundle(motion, sensors, model_archive, layout_path, destination, *,
                 selection=None, convergence=None, model=None):
    motion, sensors, destination = map(Path, (motion, sensors, destination))
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    motion_description, motion_metadata, arrays = read_internal(motion, 'motion')
    sensor_description, sensor_metadata, sensor_arrays = read_internal(sensors, 'sensors')
    if motion_description.get('motion_contract_version') != 2 \
            or sensor_description.get('sensor_contract_version') != 2:
        raise ValueError('Review bundle requires v2 artifacts')
    selected = slice(None); selection_value = None
    if selection is not None:
        selected, selection_value = selection_slice(selection, motion)
    layout = load_layout(layout_path)
    if layout['layout_id'] != sensor_metadata['kind_metadata']['layout_id']:
        raise ValueError('Review layout differs from sensor artifact')
    if model is None:
        model = load_model_member(
            model_archive, motion_metadata['kind_metadata']['source_gender'])
    model_payload = _model_payload(model, arrays['betas'], destination)
    selected_arrays = {
        'root_position_m': arrays['root_position_m'][selected],
        'joint_local_quaternion_wxyz': arrays['joint_local_quaternion_wxyz'][selected],
    }
    motion_payload = {
        key: _write_array(destination / (key + '.bin'), value, 'f4')
        for key, value in selected_arrays.items()}
    sensor_payload = {
        'specific_force_m_s2': _write_array(
            destination / 'specific_force_m_s2.bin',
            sensor_arrays['specific_force_m_s2'], 'f4'),
        'angular_velocity_rad_s': _write_array(
            destination / 'angular_velocity_rad_s.bin',
            sensor_arrays['angular_velocity_rad_s'], 'f4'),
    }
    qa = automatic_qa(motion, sensors, selection=selection,
                      convergence=convergence)
    _atomic_json(destination / 'qa.json', qa)
    period = motion_metadata['clocks']['motion']
    start = 0 if selection_value is None else selection_value['start_frame']
    manifest = {
        'schema': 'imu_motion_simulator.threejs_review.v1',
        'review_id': str(uuid4()),
        'motion_id': motion_metadata['kind_metadata']['motion_id'],
        'motion_sha256': sha256_file(motion),
        'sensor_id': sensor_metadata['kind_metadata']['sensor_id'],
        'sensor_sha256': sha256_file(sensors),
        'selection': selection_value,
        'frame_count': len(selected_arrays['root_position_m']),
        'frame_period_s': period['numerator'] / period['denominator'],
        'source_start_frame': start,
        'joint_names': motion_metadata['kind_metadata']['joint_names'],
        'model': model_payload, 'motion': motion_payload,
        'sensors': sensor_payload,
        'sensor_rate_hz': 1 / (sensor_metadata['clocks']['sensor']['numerator']
                               / sensor_metadata['clocks']['sensor']['denominator']),
        'layout': layout,
        'surface_mode': 'smplh-linear-blend-skinning-without-pose-correctives-or-dmpl',
        'dynamic_shape': _dynamic_shape(motion_metadata),
        'exact_mp4_available_on_demand': True,
        'qa': qa,
        'recipe_sha256': _producer_hash(),
    }
    _atomic_json(destination / 'manifest.json', manifest)
    labels = [] if selection_value is None else selection_value['label_candidates']
    review = {
        'schema': 'imu_motion_simulator.review_revision.v2',
        'review_id': manifest['review_id'], 'revision': 1,
        'motion_id': manifest['motion_id'], 'sensor_id': manifest['sensor_id'],
        'decision': 'unreviewed', 'reviewer': None, 'reason': None,
        'labels': labels, 'automatic_qa_passed': qa['passed'],
        'quality_flags': []}
    _atomic_json(destination / 'review-r1.json', review)
    for name in ('index.html', 'app.js', 'three.module.js', 'three.core.min.js'):
        shutil.copy2(WEB / name, destination / name)
    return validate_bundle(destination)


def validate_bundle(path):
    path = Path(path); manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('schema') != 'imu_motion_simulator.threejs_review.v1':
        raise ValueError('Invalid review manifest')
    required = {'index.html', 'app.js', 'three.module.js', 'three.core.min.js', 'qa.json',
                'review-r1.json', 'manifest.json'}
    if not required <= {entry.name for entry in path.iterdir()}:
        raise ValueError('Review bundle is incomplete')
    payloads = list(manifest['model'].values()) + list(manifest['motion'].values()) \
        + list(manifest['sensors'].values())
    for item in payloads:
        if not isinstance(item, dict):
            continue
        target = path / item['path']
        if not target.is_file() or sha256_file(target) != item['sha256']:
            raise ValueError('Review payload hash mismatch: ' + item['path'])
    reviews = sorted(path.glob('review-r*.json'),
                     key=lambda item: int(item.stem.split('r')[-1]))
    revisions = [json.loads(item.read_text()) for item in reviews]
    if [item['revision'] for item in revisions] != list(range(1, len(revisions) + 1)):
        raise ValueError('Review revisions are not contiguous')
    fields_v1 = {'schema', 'review_id', 'revision', 'motion_id', 'sensor_id',
                 'decision', 'reviewer', 'reason', 'labels',
                 'automatic_qa_passed'}
    fields_v2 = fields_v1 | {'quality_flags'}
    for item in revisions:
        schema = item.get('schema')
        fields = fields_v2 if schema == 'imu_motion_simulator.review_revision.v2' \
            else fields_v1
        if set(item) != fields \
                or schema not in ('imu_motion_simulator.review_revision.v1',
                                  'imu_motion_simulator.review_revision.v2') \
                or item['review_id'] != manifest['review_id'] \
                or item['motion_id'] != manifest['motion_id'] \
                or item['sensor_id'] != manifest['sensor_id'] \
                or item['decision'] not in ('unreviewed', 'accepted', 'rejected') \
                or not isinstance(item['labels'], list):
            raise ValueError('Invalid review revision')
        for flag in item.get('quality_flags', []):
            _validate_quality_flag(flag, manifest)
    return {'valid': True, 'review_id': manifest['review_id'],
            'motion_id': manifest['motion_id'], 'frames': manifest['frame_count'],
            'latest_revision': revisions[-1]['revision'],
            'decision': revisions[-1]['decision'],
            'automatic_qa_passed': manifest['qa']['passed'],
            'quality_flags': revisions[-1].get('quality_flags', [])}


def _validate_quality_flag(flag, manifest):
    fields = {'code', 'scope', 'source_frame_range', 'joints',
              'disposition', 'observed_at_utc'}
    if not isinstance(flag, dict) or set(flag) != fields \
            or flag['code'] != 'source-fit-temporal-discontinuity' \
            or flag['scope'] != 'downloaded-amass-fit' \
            or flag['disposition'] != 'advisory' \
            or not isinstance(flag['observed_at_utc'], str) \
            or not isinstance(flag['joints'], list) \
            or not flag['joints'] \
            or any(joint not in manifest['joint_names'] for joint in flag['joints']):
        raise ValueError('Invalid review quality flag')
    try:
        observed = datetime.fromisoformat(flag['observed_at_utc'])
    except ValueError as error:
        raise ValueError('Invalid review quality flag timestamp') from error
    if observed.utcoffset() is None or observed.utcoffset().total_seconds() != 0:
        raise ValueError('Review quality flag timestamp must be UTC')
    frame_range = flag['source_frame_range']
    source_start = manifest['source_start_frame']
    source_stop = source_start + manifest['frame_count']
    if not isinstance(frame_range, list) or len(frame_range) != 2 \
            or any(type(value) is not int for value in frame_range) \
            or not source_start <= frame_range[0] < frame_range[1] <= source_stop:
        raise ValueError('Review quality flag is outside the reviewed source range')


def append_quality_flag(path, *, reviewer, reason, start_frame, stop_frame,
                        joints, code='source-fit-temporal-discontinuity'):
    """Append an immutable advisory observation without changing source motion."""
    path = Path(path)
    report = validate_bundle(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if not reviewer or not reason:
        raise ValueError('Quality flag requires reviewer and reason')
    flag = {
        'code': code,
        'scope': 'downloaded-amass-fit',
        'source_frame_range': [start_frame, stop_frame],
        'joints': list(joints),
        'disposition': 'advisory',
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    _validate_quality_flag(flag, manifest)
    latest_path = path / f"review-r{report['latest_revision']}.json"
    latest = json.loads(latest_path.read_text())
    flags = list(latest.get('quality_flags', []))
    if any((item['code'], item['source_frame_range'])
           == (flag['code'], flag['source_frame_range']) for item in flags):
        raise ValueError('Duplicate review quality flag')
    revision = report['latest_revision'] + 1
    value = dict(latest, schema='imu_motion_simulator.review_revision.v2',
                 revision=revision, reviewer=reviewer, reason=reason,
                 quality_flags=flags + [flag])
    _atomic_json(path / f'review-r{revision}.json', value)
    return validate_bundle(path)
