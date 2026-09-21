"""Record reproducible motion/IMU features and non-decision observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from imu_motion_simulator.contracts.common import get_json, sha256_file

SCHEMA = 'imu_motion_simulator.review_diagnostics.v1'


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n')


def _peak(values, start, stop):
    if start >= stop:
        return None
    section = values[start:stop]
    sample, mount = np.unravel_index(int(np.argmax(section)), section.shape)
    return {'sample': start + int(sample), 'mount': int(mount),
            'value': float(section[sample, mount])}


def _sensor_features(array, edge):
    magnitude = np.linalg.norm(array, axis=2)
    count = len(magnitude)
    first = _peak(magnitude, 0, edge)
    middle = _peak(magnitude, edge, count - edge)
    last = _peak(magnitude, count - edge, count)
    return {
        'first': first, 'interior': middle, 'last': last,
        'global': _peak(magnitude, 0, count),
        'first_to_interior_peak_ratio': (
            None if middle is None else first['value'] / max(middle['value'], 1e-12)),
        'last_to_interior_peak_ratio': (
            None if middle is None else last['value'] / max(middle['value'], 1e-12)),
    }


def calculate_features(root, quaternion, force, gyro, *, start_frame,
                       frame_period_s, sensor_rate_hz, joint_names,
                       boundary_guard_s=.2):
    """Describe numerical behavior without assigning a quality decision."""
    root = np.asarray(root, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64)
    force = np.asarray(force, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    if len(root) < 2 or quaternion.shape != (len(root), len(joint_names), 4) \
            or len(force) < 2 or gyro.shape != force.shape \
            or force.ndim != 3 or force.shape[2] != 3:
        raise ValueError('Inconsistent motion/sensor arrays')
    edge = min(max(1, round(boundary_guard_s * sensor_rate_hz)),
               (len(force) - 1) // 2)
    if edge < 1:
        raise ValueError('Too few sensor samples for a boundary window')
    step = np.linalg.norm(np.diff(root, axis=0), axis=1)
    norms = np.linalg.norm(quaternion, axis=2)
    cosine = np.sum(quaternion[1:] * quaternion[:-1], axis=2) \
        / (norms[1:] * norms[:-1])
    angle = 2 * np.arccos(np.clip(np.abs(cosine), 0, 1))
    frame, joint = np.unravel_index(int(np.argmax(angle)), angle.shape)
    seam_cosine = np.sum(quaternion[0] * quaternion[-1], axis=1) \
        / (norms[0] * norms[-1])
    return {
        'motion_frames': len(root), 'sensor_samples': len(force),
        'duration_s': (len(root) - 1) * frame_period_s,
        'source_start_frame': start_frame,
        'sensor_rate_hz': sensor_rate_hz,
        'boundary_guard_s': boundary_guard_s,
        'boundary_samples': edge,
        'root_net_displacement_m': float(np.linalg.norm(root[-1] - root[0])),
        'root_path_length_m': float(step.sum()),
        'root_largest_step': {
            'source_frame': start_frame + int(np.argmax(step)),
            'meters': float(step.max())},
        'joint_largest_step': {
            'source_frame': start_frame + int(frame),
            'joint': joint_names[joint],
            'degrees': float(np.degrees(angle[frame, joint]))},
        'joint_first_step_max_degrees': float(np.degrees(angle[0].max())),
        'joint_last_step_max_degrees': float(np.degrees(angle[-1].max())),
        'loop_seam_root_m': float(np.linalg.norm(root[-1] - root[0])),
        'loop_seam_joint_max_degrees': float(np.degrees(
            2 * np.arccos(np.clip(np.abs(seam_cosine), 0, 1))).max()),
        'specific_force_m_s2': _sensor_features(force, edge),
        'angular_velocity_rad_s': _sensor_features(gyro, edge),
    }


def _candidate_features(candidate, objects):
    paths = {role: Path(objects[digest]['local_path'])
             for role, digest in candidate['objects'].items()}
    selection = json.loads(paths['selection'].read_text())
    if selection['motion_sha256'] != candidate['objects']['motion']:
        raise ValueError('Selection/motion hash mismatch')
    start, stop = selection['start_frame'], selection['stop_frame']
    with h5py.File(paths['motion'], 'r') as motion, \
            h5py.File(paths['sensors'], 'r') as sensors:
        motion_meta = get_json(motion, 'metadata')
        sensor_meta = get_json(sensors, 'metadata')
        if motion_meta['kind_metadata']['motion_id'] != selection['motion_id'] \
                or sensor_meta['kind_metadata']['selection_id'] != selection['selection_id']:
            raise ValueError('Diagnostic input lineage mismatch')
        period = motion_meta['clocks']['motion']
        sensor_period = sensor_meta['clocks']['sensor']
        metrics = calculate_features(
            motion['data/root_position_m'][start:stop],
            motion['data/joint_local_quaternion_wxyz'][start:stop],
            sensors['data/specific_force_m_s2'][:],
            sensors['data/angular_velocity_rad_s'][:],
            start_frame=start,
            frame_period_s=period['numerator'] / period['denominator'],
            sensor_rate_hz=sensor_period['denominator'] / sensor_period['numerator'],
            joint_names=motion_meta['kind_metadata']['joint_names'])
    return {'candidate_id': candidate['candidate_id'],
            'source_dataset': candidate['source_dataset'],
            'source_member': candidate['source_member'],
            'input_objects': candidate['objects'], 'metrics': metrics}


def _observations(path, candidates):
    document = json.loads(Path(path).read_text())
    if set(document) != {'schema', 'observations'} \
            or document['schema'] != 'imu_motion_simulator.observation_input.v1':
        raise ValueError('Invalid observation input')
    available = {row['candidate_id']: row for row in candidates}
    expanded = []
    seen = set()
    for item in document['observations']:
        if set(item) != {'candidate_ids', 'region', 'feature', 'note'} \
                or item['region'] not in {'start', 'middle', 'end', 'whole'} \
                or not isinstance(item['candidate_ids'], list) \
                or not item['candidate_ids'] or not item['feature'] or not item['note']:
            raise ValueError('Invalid observation entry')
        for candidate_id in item['candidate_ids']:
            if candidate_id not in available:
                raise ValueError('Observation references unknown candidate: ' + candidate_id)
            key = (candidate_id, item['region'], item['feature'])
            if key in seen:
                raise ValueError('Duplicate observation: ' + candidate_id)
            seen.add(key)
            expanded.append({'candidate_id': candidate_id,
                             'source_dataset': available[candidate_id]['source_dataset'],
                             'input_objects': available[candidate_id]['objects'],
                             'region': item['region'],
                             'feature': item['feature'],
                             'note': item['note'],
                             'provenance': 'user observation, 2026-09-16',
                             'source_frame_range': None})
    return expanded


def build_diagnostics(corpus_path, observation_path, output, *,
                      sample_index_path=None):
    corpus_path, observation_path, output = (
        Path(value).resolve() for value in
        (corpus_path, observation_path, output))
    corpus = json.loads(corpus_path.read_text())
    if corpus['schema'] != 'imu_motion_simulator.candidate_corpus.v1':
        raise ValueError('Unsupported candidate corpus')
    candidates = sorted(corpus['candidates'], key=lambda row: row['candidate_id'])
    sample_sha256 = None
    if sample_index_path is not None:
        sample_index_path = Path(sample_index_path).resolve()
        sample = json.loads(sample_index_path.read_text())
        if sample['schema'] != 'imu_motion_simulator.review_sample_index.v1':
            raise ValueError('Unsupported review sample index')
        sample_ids = {row['candidate_id'] for row in sample['bundles']}
        selected = [row for row in candidates
                    if row['candidate_id'] in sample_ids]
        if len(selected) != len(sample_ids):
            raise ValueError('Review sample is absent from candidate corpus')
        sample_sha256 = sha256_file(sample_index_path)
    else:
        selected = candidates
    binding = {'schema': SCHEMA, 'corpus_sha256': sha256_file(corpus_path),
               'observation_input_sha256': sha256_file(observation_path),
               'code_sha256': sha256_file(__file__),
               'sample_index_sha256': sample_sha256,
               'candidate_count': len(selected)}
    if output.exists():
        report = json.loads((output / 'report.json').read_text())
        if report['binding'] != binding \
                or sha256_file(output / 'features.jsonl') != report['features_sha256'] \
                or sha256_file(output / 'observations.json') != report['observations_sha256']:
            raise ValueError('Existing diagnostic study has different inputs')
        return report
    partial = output.with_name(output.name + '.partial')
    partial.mkdir(parents=True, exist_ok=True)
    binding_path = partial / 'binding.json'
    if binding_path.exists():
        if json.loads(binding_path.read_text()) != binding:
            raise ValueError('Partial study has different inputs')
    else:
        _write_json(binding_path, binding)
    observations = _observations(observation_path, candidates)
    selected_ids = {row['candidate_id'] for row in selected}
    observations = [row for row in observations
                    if row['candidate_id'] in selected_ids]
    _write_json(partial / 'observations.json', {
        'schema': 'imu_motion_simulator.diagnostic_observations.v1',
        'binding_sha256': sha256_file(binding_path),
        'observations': observations})
    feature_path = partial / 'features.jsonl'
    existing = {}
    if feature_path.exists():
        for line in feature_path.read_text().splitlines():
            row = json.loads(line)
            if row['candidate_id'] in existing:
                raise ValueError('Duplicate partial feature row')
            existing[row['candidate_id']] = row
    objects = {item['sha256']: item for item in corpus['objects']}
    with feature_path.open('a') as handle:
        for candidate in selected:
            candidate_id = candidate['candidate_id']
            if candidate_id in existing:
                if existing[candidate_id]['input_objects'] != candidate['objects']:
                    raise ValueError('Partial feature input mismatch')
                continue
            feature = _candidate_features(candidate, objects)
            handle.write(json.dumps(feature, ensure_ascii=False,
                                    allow_nan=False) + '\n')
            handle.flush()
    report = {'schema': SCHEMA, 'binding': binding,
              'features_sha256': sha256_file(feature_path),
              'observations_sha256': sha256_file(partial / 'observations.json'),
              'features': len(selected), 'observations': len(observations),
              'decision_fields': False}
    _write_json(partial / 'report.json', report)
    partial.rename(output)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('observations', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--sample-index', type=Path)
    args = parser.parse_args()
    print(json.dumps(build_diagnostics(
        args.corpus, args.observations, args.output,
        sample_index_path=args.sample_index),
        indent=2))
