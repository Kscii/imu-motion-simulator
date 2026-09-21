"""The unchanged v3.2/v3.3 IMU core and its container-independent identity."""
from __future__ import annotations

import hashlib
import json
import h5py
import numpy as np

from .common import check_table, finite, json_load, require, sha_string, text

COLUMNS = ['acceleration_x_mps2', 'acceleration_y_mps2', 'acceleration_z_mps2',
           'angular_velocity_x_rad_s', 'angular_velocity_y_rad_s', 'angular_velocity_z_rad_s']
UNITS = ['m/s^2'] * 3 + ['rad/s'] * 3
SEQUENCES = [('sample_start', 'int64'), ('sample_stop', 'int64'), ('source_file', 'UTF-8'),
             ('participant_id', 'UTF-8'), ('recording_id', 'UTF-8'), ('body_location', 'UTF-8'),
             ('activity_code', 'UTF-8'), ('is_fall', 'bool'), ('supervision_kind', 'UTF-8'),
             ('source_sampling_rate_hz', 'float64')]
ANNOTATIONS = [('sequence_index', 'int32'), ('kind', 'UTF-8'), ('start_sample', 'int64'),
               ('stop_sample', 'int64'), ('code', 'UTF-8')]
KINDS = {'activity': 0, 'onset': 1, 'impact': 2, 'exclude': 3}


def local_annotations(annotations):
    grouped = {}
    for row in annotations:
        item = dict(kind=text(row['kind']), start_sample=int(row['start_sample']),
                    stop_sample=int(row['stop_sample']), code=text(row['code']))
        grouped.setdefault(int(row['sequence_index']), []).append(item)
    return grouped


def logical_content_sha256(samples, sequences, annotations, *, dataset_id, sampling_rate_hz=25.0):
    """Preserve the existing collector/benchmark ordered sequence-local algorithm.

    Conformance is checked against the independent existing implementation.
    HDF5 input is read in bounded sample chunks; global offsets are not identity.
    """
    digest = hashlib.sha256()
    grouped = local_annotations(annotations)
    for index, row in enumerate(sequences):
        start, stop = int(row['sample_start']), int(row['sample_stop'])
        labels = sorted(grouped.get(index, []), key=lambda item: (
            item['start_sample'], KINDS.get(item['kind'], 99), item['stop_sample'], item['code']))
        metadata = dict(dataset_id=dataset_id, source_file=text(row['source_file']),
                        participant_id=text(row['participant_id']), recording_id=text(row['recording_id']),
                        body_location=text(row['body_location']), activity=text(row['activity_code']),
                        is_fall=bool(row['is_fall']), sampling_rate_hz=float(sampling_rate_hz),
                        original_sampling_rate_hz=float(row['source_sampling_rate_hz']),
                        supervision_kind=text(row['supervision_kind']), annotations=labels)
        encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()
        digest.update(len(encoded).to_bytes(8, 'little'))
        digest.update(encoded)
        digest.update(np.asarray([stop - start, 6], dtype='<i8').tobytes())
        for offset in range(start, stop, 65536):
            digest.update(np.asarray(samples[offset:min(stop, offset + 65536)], dtype='<f4', order='C').tobytes())
    return digest.hexdigest()


def validate_annotations(sequences, annotations):
    previous = None
    for row in annotations:
        index, start, stop = int(row['sequence_index']), int(row['start_sample']), int(row['stop_sample'])
        kind, code = text(row['kind']), text(row['code'])
        require(0 <= index < len(sequences) and kind in KINDS and bool(code), 'invalid annotation identity')
        length = int(sequences[index]['sample_stop'] - sequences[index]['sample_start'])
        require(0 <= start < length, 'annotation start outside sequence')
        require(stop == start if kind in ('onset', 'impact') else start < stop <= length, 'annotation interval/point')
        if kind == 'exclude':
            require(code in ('sync_tap', 'other'), 'unknown exclude code')
        key = (index, start, KINDS[kind], stop, code)
        require(previous is None or previous <= key, 'annotations not deterministically sorted')
        previous = key
    grouped = local_annotations(annotations)
    for index, sequence in enumerate(sequences):
        if text(sequence['supervision_kind']) != 'temporal':
            continue
        rows = grouped.get(index, [])
        intervals = sorted((r['start_sample'], r['stop_sample']) for r in rows if r['kind'] in ('activity', 'exclude'))
        cursor = 0
        for start, stop in intervals:
            require(start == cursor, 'annotation coverage gap/overlap')
            cursor = stop
        require(cursor == int(sequence['sample_stop'] - sequence['sample_start']), 'incomplete annotation coverage')
        onsets = [r for r in rows if r['kind'] == 'onset']
        impacts = [r for r in rows if r['kind'] == 'impact']
        used_impacts, used_activities = set(), set()
        activities = [r for r in rows if r['kind'] == 'activity']
        for onset in onsets:
            matching = [i for i, activity in enumerate(activities) if activity['start_sample'] == onset['start_sample']
                        and activity['code'] == onset['code']]
            require(len(matching) == 1 and matching[0] not in used_activities, 'onset does not uniquely match activity')
            ai = matching[0]
            activity = activities[ai]
            found = [i for i, impact in enumerate(impacts) if impact['code'] == onset['code']
                     and activity['start_sample'] < impact['start_sample'] < activity['stop_sample']]
            require(len(found) == 1 and found[0] not in used_impacts, 'impact must be strictly inside one fall')
            used_activities.add(ai)
            used_impacts.add(found[0])
        require(len(used_impacts) == len(impacts), 'orphan impact')
        require(bool(sequence['is_fall']) == bool(onsets), 'sequence fall flag inconsistent with events')


def validate_core(handle, *, verify_hash=True, allow_provisional=False):
    for name in ('samples', 'sequences', 'annotations'):
        require(name in handle, f'missing core {name}')
    samples, sequences, annotations = (handle[name] for name in ('samples', 'sequences', 'annotations'))
    require(isinstance(samples, h5py.Dataset), 'samples must be a dataset')
    require(samples.ndim == 2 and samples.shape[1] == 6 and samples.dtype == np.dtype('float32'), 'samples dtype/shape')
    require(json_load(samples.attrs.get('columns', 'null')) == COLUMNS, 'sample columns')
    require(json_load(samples.attrs.get('units', 'null')) == UNITS, 'sample units')
    check_table(sequences, SEQUENCES)
    check_table(annotations, ANNOTATIONS)
    rows, labels = sequences[:], annotations[:]
    require(len(rows) > 0, 'empty sequences')
    cursor = 0
    for row in rows:
        start, stop = int(row['sample_start']), int(row['sample_stop'])
        require(start == cursor and stop - start >= 2, 'noncontiguous or too-short sequence')
        cursor = stop
        require(all(text(row[name]) for name, _ in SEQUENCES[2:7]), 'empty sequence identity')
        require(text(row['supervision_kind']) in ('recording', 'temporal'), 'unknown supervision')
        require(np.isfinite(row['source_sampling_rate_hz']) and row['source_sampling_rate_hz'] > 0, 'invalid source rate')
    require(cursor == len(samples), 'sequences do not exactly cover samples')
    for key, expected in [('sampling_rate_hz', 25.0), ('axis_frame', 'sensor_local'), ('hdf5_compatibility', '1.14')]:
        actual = handle.attrs.get(key)
        require((float(actual) if key == 'sampling_rate_hz' and actual is not None else text(actual)) == expected,
                f'invalid {key}')
    require(text(handle.attrs.get('dataset_id', '')), 'empty dataset id')
    roles = ('training_only', 'cross_validation', 'unverified_synthetic') if allow_provisional else ('training_only', 'cross_validation')
    require(text(handle.attrs.get('evaluation_role', '')) in roles, 'evaluation role')
    require(json_load(handle.attrs.get('feature_columns', 'null')) == COLUMNS, 'feature columns')
    for name, count in [('sample_count', len(samples)), ('sequence_count', len(rows)), ('annotation_count', len(labels))]:
        require(np.asarray(handle.attrs.get(name)).dtype == np.dtype('i8') and handle.attrs.get(name) == count, f'{name} dtype/mismatch')
    validate_annotations(rows, labels)
    expected = text(handle.attrs.get('logical_content_sha256', ''))
    require(sha_string(expected), 'invalid logical hash')
    if verify_hash:
        finite(samples)
        actual = logical_content_sha256(samples, rows, labels, dataset_id=text(handle.attrs['dataset_id']))
        require(expected == actual, 'logical core hash mismatch')
    return rows, labels
