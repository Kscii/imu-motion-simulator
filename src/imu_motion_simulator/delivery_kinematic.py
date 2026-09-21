"""Accepted kinematic review export to the project HDF5 3.3 contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .contracts.common import COORDINATES, sha256_file
from .contracts.core import ANNOTATIONS, SEQUENCES
from .contracts.delivery import CATALOG, REPLAY, VERSIONS, table, write_delivery
from .contracts.internal import read_internal
from .motion.selection import selection_slice
from .review.bundle import validate_bundle
from .sensors.layout import load_layout


def _latest_accepted(review_bundle):
    review_bundle = Path(review_bundle)
    validate_bundle(review_bundle)
    rows = sorted(review_bundle.glob('review-r*.json'),
                  key=lambda path: int(path.stem.split('r')[-1]))
    if not rows:
        raise ValueError('Review bundle has no revisions')
    value = json.loads(rows[-1].read_text())
    if value.get('decision') != 'accepted' \
            or value.get('automatic_qa_passed') is not True \
            or not value.get('reviewer') \
            or not value.get('reason') or not value.get('labels'):
        raise ValueError('Latest review revision is not an accepted, QA-passed labeled decision')
    return rows[-1], value


def _dynamic_shape(metadata):
    return metadata['resolved_config'].get('dynamic_shape', {
        'source_available': True, 'effective_policy': 'source',
        'components': 8})


def _model_asset(model_archive, dmpl_archive, model_hash, dynamic_available):
    files, blobs = [], {}
    sources = [('smplh.tar.xz', Path(model_archive))]
    if dynamic_available:
        sources.append(('dmpls.tar.xz', Path(dmpl_archive)))
    for logical, path in sources:
        content = path.read_bytes(); digest = hashlib.sha256(content).hexdigest()
        blobs[digest] = content
        files.append({'logical_path': logical, 'sha256': digest,
                      'byte_length': len(content),
                      'media_type': 'application/x-xz',
                      'blob_path': '/assets/blobs/' + digest,
                      'external_ref': None})
    asset = {
        'asset_id': ('smplh-dmpl/' if dynamic_available else 'smplh/')
        + model_hash[:12], 'role': 'smplh_model',
        'revision': model_hash[:12], 'media_type': 'application/json',
        'entrypoint': 'smplh.tar.xz', 'files': files, 'dependencies': [],
        'license': {'id': 'source-terms-retained', 'source_url': None,
                    'attribution': ('SMPL+H and DMPL source packages'
                                    if dynamic_available else
                                    'SMPL+H source package'),
                    'distribution_scope': 'local-user-research-only',
                    'evidence_refs': ['local-source-archive']},
        'provenance': {'source_ids': (['extended-smplh', 'dmpl']
                                     if dynamic_available else
                                     ['extended-smplh']),
                       'authoring_tool_versions': {}, 'transformations': [],
                       'limitations': [
                           'Private local export; no redistribution decision']
                       + ([] if dynamic_available else [
                           'Source dynamic shape is unavailable; replay DMPL is disabled-zero'])}}
    return asset, blobs


def export_kinematic(motion, sensors, selection, review_bundle, layout_path,
                     output, *, dataset_id, model_archive=None,
                     dmpl_archive=None, include_replay=False, mp4=None):
    motion, sensors, output = map(Path, (motion, sensors, output))
    review_path, review = _latest_accepted(review_bundle)
    motion_description, motion_metadata, motion_arrays = read_internal(motion, 'motion')
    sensor_description, sensor_metadata, sensor_arrays = read_internal(sensors, 'sensors')
    selected, selection_value = selection_slice(selection, motion)
    if motion_description.get('motion_contract_version') != 2 \
            or sensor_description.get('sensor_contract_version') != 2 \
            or sensor_metadata['kind_metadata']['selection_id'] != selection_value['selection_id'] \
            or review['motion_id'] != motion_metadata['kind_metadata']['motion_id'] \
            or review['sensor_id'] != sensor_metadata['kind_metadata']['sensor_id']:
        raise ValueError('Export inputs do not share one accepted lineage')
    layout = load_layout(layout_path)
    if layout['layout_id'] != sensor_metadata['kind_metadata']['layout_id']:
        raise ValueError('Export layout differs from sensor layout')
    labels_by_code = {}
    for label in review['labels']:
        if not isinstance(label, dict) or not label.get('code'):
            raise ValueError('Accepted labels require a code')
        code = label['code']
        labels_by_code.setdefault(code, {
            'code': code, 'name': label.get('name', code),
            'is_fall': bool(label.get('is_fall', False)),
            'taxonomy_id': label.get('taxonomy_id', 'imu-motion-simulator-activity'),
            'taxonomy_version': label.get('taxonomy_version', '1.0.0'),
            'origin': label.get('origin', 'legacy-human'),
            'verification': label.get('verification', 'human'),
            'mapping_rule_id': label.get('mapping_rule_id'),
        })
    if len(labels_by_code) != 1:
        raise ValueError('Recording-level v1 export requires exactly one activity label')
    activity = next(iter(labels_by_code.values()))
    count, mounts = sensor_arrays['specific_force_m_s2'].shape[:2]
    samples = np.concatenate([
        np.c_[sensor_arrays['specific_force_m_s2'][:, index],
              sensor_arrays['angular_velocity_rad_s'][:, index]]
        for index in range(mounts)], axis=0).astype('<f4')
    info = motion_metadata['kind_metadata']
    participant = 'smplh-' + info['source_gender'] + '-' \
        + hashlib.sha256(motion_arrays['betas'].astype('<f8').tobytes()).hexdigest()[:12]
    sequences = []
    for index, mount in enumerate(layout['mounts']):
        sequences.append((index * count, (index + 1) * count,
                          info['source_member'], participant,
                          selection_value['selection_id'], mount['id'], activity['code'],
                          activity['is_fall'], 'recording', info['source_fps_hz']))
    sequences = table(sequences, SEQUENCES)
    annotations = table([], ANNOTATIONS)
    taxonomy_id = activity.get('taxonomy_id', 'imu-motion-simulator-activity')
    taxonomy_version = activity.get('taxonomy_version', '1.0.0')
    if not taxonomy_id or not taxonomy_version:
        raise ValueError('Accepted activity lacks taxonomy identity')
    labels = {
        'catalog': table([(taxonomy_id, taxonomy_version, activity['code'],
                           activity['name'], activity['is_fall'], True)], CATALOG),
        'sequence_versions': table([(index, taxonomy_id, taxonomy_version)
                                    for index in range(mounts)], VERSIONS)}
    replay = assets = blobs = None
    if include_replay:
        dynamic = _dynamic_shape(motion_metadata)
        if model_archive is None or (dynamic['source_available']
                                     and dmpl_archive is None):
            raise ValueError(
                'Self-contained replay requires SMPL+H and any source DMPL archive')
        asset, blobs = _model_asset(
            model_archive, dmpl_archive, info['model_sha256'],
            dynamic['source_available'])
        assets = [asset]
        replay_arrays = {
            'time_ns': motion_arrays['time_ns'][selected].astype('<i8'),
            'root_position_m': motion_arrays['root_position_m'][selected].astype('<f4'),
            'root_quaternion_wxyz': motion_arrays['root_quaternion_wxyz'][selected].astype('<f4'),
            'joint_local_quaternion_wxyz': motion_arrays['joint_local_quaternion_wxyz'][selected].astype('<f4'),
            'betas': motion_arrays['betas'].astype('<f4'),
            'dmpls': motion_arrays['dmpls'][selected].astype('<f4'),
        }
        period = motion_metadata['clocks']['motion']
        replay_metadata = {
            'replay_contract_version': (2 if dynamic['source_available'] else 3),
            'model_asset_id': asset['asset_id'],
            'model_family': 'smplh', 'joint_names': info['joint_names'],
            'coordinates': COORDINATES,
            'clock': {'original_rate_hz': info['source_fps_hz'],
                      'period_s': {'numerator': period['numerator'],
                                   'denominator': period['denominator']},
                      'origin': 'canonical motion source time'},
            'objects': [], 'source_motion_id': info['motion_id'],
            'representation': 'local-quaternion-wxyz'}
        if not dynamic['source_available']:
            replay_metadata['dynamic_shape'] = dynamic
        record_id = 'motion-' + info['motion_id']
        zero = int(replay_arrays['time_ns'][0])
        replay = {'index': table([(index, record_id, zero)
                                  for index in range(mounts)], REPLAY),
                  'records': {record_id: {'metadata': replay_metadata,
                                          'arrays': replay_arrays}}}
    media = None
    if mp4 is not None:
        mp4 = Path(mp4); sidecar = json.loads(mp4.with_suffix(mp4.suffix + '.json').read_text())
        if sidecar['motion_sha256'] != sha256_file(motion) \
                or sidecar['sensors_sha256'] != sha256_file(sensors):
            raise ValueError('MP4 sidecar differs from export lineage')
        duration = int(round(sidecar['frames'] / sidecar['fps'] * 1e9))
        sensor_end = int(sensor_arrays['time_ns'][-1])
        media = [{'sequence_index': 0, 'bytes': mp4.read_bytes(),
                  'timing': np.asarray([[0, 0], [sensor_end, sensor_end]], dtype='<i8'),
                  'media_duration_ns': duration,
                  'sample_zero_recording_time_ns': 0,
                  'sample_zero_media_time_ns': 0}]
    provenance = {
        'producer': {'name': 'imu_motion_simulator.delivery_kinematic',
                     'version': '1.0.0'},
        'input_files': [
            {'role': 'motion', 'sha256': sha256_file(motion)},
            {'role': 'sensors', 'sha256': sha256_file(sensors)},
            {'role': 'selection', 'sha256': sha256_file(selection)},
            {'role': 'review', 'sha256': sha256_file(review_path)}],
        'sequence_sources': [{
            'sequence_index': index, 'source_kind': 'synthetic',
            'group_keys': {'dataset': info['source_dataset'],
                           'member': info['source_member'],
                           'source_gender': info['source_gender']},
            'refs': {
                'motion': {'artifact_id': motion_description['artifact_id'],
                           'sha256': sha256_file(motion)},
                'sensors': {'artifact_id': sensor_description['artifact_id'],
                            'sha256': sha256_file(sensors)},
                'review': {'artifact_id': review['review_id'],
                           'sha256': sha256_file(review_path)},
                'activity_label': {
                    'taxonomy_id': taxonomy_id, 'taxonomy_version': taxonomy_version,
                    'code': activity['code'],
                    'origin': activity.get('origin', 'legacy-human'),
                    'verification': activity.get('verification', 'human'),
                    'mapping_rule_id': activity.get('mapping_rule_id')},
                'route': 'direct-mocap-kinematic',
                'model_sha256': info['model_sha256'],
                'source_quality_flags': review.get('quality_flags', [])}}
            for index in range(mounts)],
        'limitations': ['Kinematic motion reproduction; no contact forces or autonomous response',
                        'Ideal IMU unless the input sensor artifact declares calibrated variant']
        + ([] if _dynamic_shape(motion_metadata)['source_available'] else [
            'Source dynamic shape is unavailable; replay DMPL is disabled-zero'])}
    return write_delivery(
        output, samples=samples, sequences=sequences, annotations=annotations,
        dataset_id=dataset_id, labels=labels, media=media, replay=replay,
        assets=assets, blobs=blobs, provenance=provenance,
        evaluation_role='training_only')
