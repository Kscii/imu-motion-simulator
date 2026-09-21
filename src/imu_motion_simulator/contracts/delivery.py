"""Immutable v3.2/v3.3 contract I/O; not a review service or publication API."""
from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4
from fractions import Fraction
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .assets import validate_assets, write_assets
from .common import (COORDINATES, atomic_h5, check_table, digest_dataset, fields_dtype,
                     finite, get_json, json_dump, local_tree, put_json, require,
                     sha256_file, sha_string, text, time_ns, uuid_string, window_rows)
from .core import (ANNOTATIONS, COLUMNS, SEQUENCES, UNITS, local_annotations,
                   logical_content_sha256, validate_core)

PROVISIONAL_INDEX = [('sequence_index', 'int32'), ('candidate_id', 'UTF-8'),
                     ('version_id', 'UTF-8'), ('commit_sha256', 'UTF-8'),
                     ('source_dataset', 'UTF-8')]
WEAK_LABEL_INDEX = [('sequence_index', 'int32'), ('state', 'UTF-8'),
                    ('code', 'UTF-8'), ('rule_id', 'UTF-8')]

CATALOG = [('taxonomy_id', 'UTF-8'), ('taxonomy_version', 'UTF-8'), ('code', 'UTF-8'),
           ('name', 'UTF-8'), ('is_fall', 'bool'), ('active', 'bool')]
VERSIONS = [('sequence_index', 'int32'), ('taxonomy_id', 'UTF-8'), ('taxonomy_version', 'UTF-8')]
MEDIA = [('sequence_index', 'int32'), ('byte_length', 'int64'), ('file_offset', 'int64'),
         ('sha256', 'UTF-8'), ('content_type', 'UTF-8'), ('container', 'UTF-8'),
         ('media_duration_ns', 'int64'), ('sample_zero_recording_time_ns', 'int64'),
         ('sample_zero_media_time_ns', 'int64')]
REPLAY = [('sequence_index', 'int32'), ('record_id', 'UTF-8'), ('sample_zero_replay_time_ns', 'int64')]


def table(rows, fields):
    return np.asarray([tuple(row) for row in rows], dtype=fields_dtype(fields))


def sparse_indices(rows, sequence_count):
    indices = [int(row['sequence_index']) for row in rows]
    require(indices and indices == sorted(set(indices)) and 0 <= indices[0] <= indices[-1] < sequence_count,
            'attachment indices must be a sorted unique nonempty subset')
    return indices


def validate_labels(handle, sequences, annotations):
    require(set(handle['labels']) == {'catalog', 'sequence_versions'}, 'label objects')
    check_table(handle['labels/catalog'], CATALOG)
    check_table(handle['labels/sequence_versions'], VERSIONS)
    catalog, versions = handle['labels/catalog'][:], handle['labels/sequence_versions'][:]
    lookup = {}
    for row in catalog:
        key = tuple(text(row[name]) for name in ('taxonomy_id', 'taxonomy_version', 'code'))
        require(all(key) and text(row['name']) and key not in lookup, 'empty/duplicate label')
        lookup[key] = bool(row['is_fall'])
    require([int(r['sequence_index']) for r in versions] == list(range(len(sequences))), 'taxonomy sequence coverage')
    grouped = local_annotations(annotations)
    for i, (sequence, version) in enumerate(zip(sequences, versions)):
        prefix = tuple(text(version[k]) for k in ('taxonomy_id', 'taxonomy_version'))
        require(all(prefix), 'empty taxonomy identity')
        labels = grouped.get(i, [])
        for row in labels:
            if row['kind'] == 'exclude':
                continue
            key = prefix + (row['code'],)
            require(key in lookup, 'unresolved taxonomy code')
            if row['kind'] in ('onset', 'impact'):
                require(lookup[key], 'fall event uses non-fall code')
            if row['kind'] == 'activity' and lookup[key]:
                onsets = [r for r in labels if r['kind'] == 'onset' and r['code'] == row['code'] and r['start_sample'] == row['start_sample']]
                require(len(onsets) == 1, 'fall activity without onset')
        if text(sequence['supervision_kind']) == 'recording':
            key = prefix + (text(sequence['activity_code']),)
            require(key in lookup and lookup[key] == bool(sequence['is_fall']), 'recording activity taxonomy mismatch')


def validate_media(handle, sequences):
    require(set(handle['media']) == {'index', 'videos', 'timing'}, 'media objects')
    check_table(handle['media/index'], MEDIA)
    rows = handle['media/index'][:]
    indices = sparse_indices(rows, len(sequences))
    require(set(handle['media/videos']) == set(handle['media/timing']) == {str(i) for i in indices}, 'media index/payload mismatch')
    ranges = []
    for row in rows:
        i = str(int(row['sequence_index']))
        video, timing = handle['media/videos/' + i], handle['media/timing/' + i]
        require(video.ndim == 1 and video.dtype == np.dtype('u1') and video.chunks is None and video.compression is None,
                'video must be contiguous uint8')
        require(video.id.get_create_plist().get_nfilters() == 0, 'video filters forbidden')
        offset = video.id.get_offset()
        require(offset is not None and int(row['file_offset']) == offset and int(row['byte_length']) == len(video), 'video physical offset/length mismatch')
        require(len(video) >= 12 and video[4:8].tobytes() == b'ftyp', 'invalid MP4 signature')
        require(text(row['content_type']) == 'video/mp4' and text(row['container']) == 'mp4', 'video type')
        require(digest_dataset(video) == text(row['sha256']), 'video hash mismatch')
        require(offset + len(video) <= Path(handle.filename).stat().st_size, 'video beyond file')
        ranges.append((offset, offset + len(video)))
        require(timing.ndim == 2 and timing.shape[1] == 2 and timing.dtype == np.dtype('i8') and len(timing) > 0, 'media timing shape/dtype')
        values = timing[:]
        require(np.all(values[1:] > values[:-1]) and np.all(values >= 0), 'media timing not increasing/nonnegative')
        duration = int(row['media_duration_ns'])
        require(duration > 0 and int(values[-1, 1]) <= duration, 'media timing beyond duration')
        zero, media_zero = int(row['sample_zero_recording_time_ns']), int(row['sample_zero_media_time_ns'])
        require(int(values[0, 0]) <= zero <= int(values[-1, 0]), 'sample zero outside timing')
        k = int(np.searchsorted(values[:, 0], zero))
        if int(values[k, 0]) == zero:
            mapped = Fraction(int(values[k, 1]))
        else:
            a, b = values[k - 1], values[k]
            mapped = int(a[1]) + Fraction(zero - int(a[0]), int(b[0]) - int(a[0])) * (int(b[1]) - int(a[1]))
        require(abs(mapped - media_zero) <= Fraction(1, 2), 'sample zero media mapping mismatch')
    ranges.sort()
    require(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])), 'overlapping video ranges')
    return indices


def validate_replay(handle, sequences, registry):
    require(set(handle['replay']) == {'index', 'records'}, 'replay objects')
    check_table(handle['replay/index'], REPLAY)
    rows = handle['replay/index'][:]
    sparse_indices(rows, len(sequences))
    ids = {text(row['record_id']) for row in rows}
    require(set(handle['replay/records']) == ids, 'replay record/index mismatch')
    for rid in ids:
        require(rid and '/' not in rid and rid not in ('.', '..'), 'invalid replay record id')
        record = handle['replay/records/' + rid]
        metadata = get_json(record, 'metadata')
        if metadata.get('replay_contract_version') in (2, 3):
            replay_version = metadata['replay_contract_version']
            required = {'replay_contract_version', 'model_asset_id', 'model_family',
                        'joint_names', 'coordinates', 'clock', 'objects',
                        'source_motion_id', 'representation'}
            if replay_version == 3:
                required.add('dynamic_shape')
            require(set(metadata) == required and metadata['model_family'] == 'smplh'
                    and metadata['representation'] == 'local-quaternion-wxyz'
                    and len(metadata['joint_names']) == 52
                    and metadata['coordinates'] == COORDINATES
                    and metadata['source_motion_id'], 'kinematic replay metadata fields')
            if replay_version == 3:
                dynamic = metadata['dynamic_shape']
                require(set(dynamic) == {'source_available', 'effective_policy',
                                         'components'}
                        and dynamic == {'source_available': False,
                                        'effective_policy': 'disabled-zero',
                                        'components': 8},
                        'kinematic replay-v3 dynamic-shape declaration')
            aid = metadata['model_asset_id']
            require(aid in registry and registry[aid][0]['role'] == 'smplh_model',
                    'missing/wrong SMPL+H replay asset')
            expected_model_files = ({'smplh.tar.xz', 'dmpls.tar.xz'}
                                    if replay_version == 2 else
                                    {'smplh.tar.xz'})
            require(set(registry[aid][1]) == expected_model_files,
                    'kinematic replay model asset files')
            require(metadata['objects'] == [], 'kinematic replay objects not supported')
            shapes = {
                'time_ns': (len(record['time_ns']),),
                'root_position_m': (len(record['time_ns']), 3),
                'root_quaternion_wxyz': (len(record['time_ns']), 4),
                'joint_local_quaternion_wxyz': (len(record['time_ns']), 52, 4),
                'betas': (16,), 'dmpls': (len(record['time_ns']), 8),
            }
            require(set(record) == set(shapes) | {'metadata'},
                    'kinematic replay arrays')
            for name, shape in shapes.items():
                ds = record[name]
                dtype = 'i8' if name == 'time_ns' else 'f4'
                require(ds.shape == shape and ds.dtype == np.dtype(dtype),
                        'kinematic replay shape/dtype: ' + name)
                finite(ds)
                if name in ('root_quaternion_wxyz',
                            'joint_local_quaternion_wxyz'):
                    for start in range(0, len(ds), 65536):
                        require(np.all(np.abs(np.linalg.norm(
                            ds[start:start + 65536], axis=-1) - 1) <= 1e-5),
                            'kinematic replay quaternion norm')
            if replay_version == 3:
                require(not np.any(record['dmpls'][:]),
                        'kinematic replay-v3 disabled DMPL must be zero')
            times = record['time_ns'][:]
            require(len(times) >= 2 and np.all(times[1:] > times[:-1]),
                    'kinematic replay clock order')
            clock = metadata['clock']; period = clock.get('period_s')
            require(set(clock) == {'original_rate_hz', 'period_s', 'origin'}
                    and set(period) == {'numerator', 'denominator'}
                    and clock['origin'], 'kinematic replay clock fields')
            expected = time_ns(np.arange(len(times)), period['numerator'],
                               period['denominator'])
            rate = period['denominator'] / period['numerator']
            require(np.isclose(clock['original_rate_hz'], rate, rtol=1e-12),
                    'kinematic replay original rate')
            # A cropped rational clock can differ from a freshly generated
            # zero-origin clock by one nanosecond at rounding boundaries.
            require(np.all(np.abs((times - times[0]) - expected) <= 1),
                    'kinematic replay accumulated clock drift')
            for row in rows:
                if text(row['record_id']) != rid:
                    continue
                sequence = sequences[int(row['sequence_index'])]
                zero = int(row['sample_zero_replay_time_ns'])
                end = zero + (int(sequence['sample_stop'] - sequence['sample_start']) - 1) * 40_000_000
                require(int(times[0]) <= zero <= end <= int(times[-1]),
                        'kinematic replay does not envelope IMU samples')
            continue
        require(set(metadata) == {'model_asset_id', 'scene_asset_id', 'visual_asset_id', 'binding_asset_id',
                                 'root_link', 'joint_names', 'joint_units', 'coordinates', 'clock', 'objects', 'source_episode_id'}, 'replay metadata fields')
        require(metadata['coordinates'] == COORDINATES and uuid_string(metadata['source_episode_id']), 'replay coordinates/episode')
        for field, role in [('model_asset_id', 'model_package'), ('scene_asset_id', 'scene'), ('visual_asset_id', 'visual'), ('binding_asset_id', 'binding')]:
            aid = metadata[field]
            if aid is not None:
                require(aid in registry and registry[aid][0]['role'] == role, 'missing/wrong replay asset')
            else:
                require(field in ('visual_asset_id', 'binding_asset_id'), 'missing required replay asset')
        require((metadata['visual_asset_id'] is None) == (metadata['binding_asset_id'] is None), 'visual/binding must be paired')
        # Skin binding acceptance is a separate capability, not silently inferred.
        require(metadata['visual_asset_id'] is None, 'visual binding validation not yet implemented')
        files = registry[metadata['model_asset_id']][1]
        urdf = ET.fromstring(handle[files['model.urdf']['blob_path']][:].tobytes())
        joints = [j for j in urdf.findall('joint') if j.attrib['type'] != 'fixed']
        require(metadata['joint_names'] == [j.attrib['name'] for j in joints], 'replay/source joint order mismatch')
        require(metadata['joint_units'] == [('m' if j.attrib['type'] == 'prismatic' else 'rad') for j in joints], 'replay joint units')
        roots = {link.attrib['name'] for link in urdf.findall('link')} - {j.find('child').attrib['link'] for j in urdf.findall('joint')}
        require(roots == {metadata['root_link']}, 'source root mismatch')
        objects = metadata['objects']
        require(len({o['object_id'] for o in objects}) == len(objects), 'duplicate replay object')
        for obj in objects:
            require(set(obj) == {'object_id', 'asset_id'} and obj['asset_id'] in registry, 'replay object asset')
        shapes = {'time_ns': (len(record['time_ns']),), 'root_position_m': (len(record['time_ns']), 3),
                  'root_quaternion_wxyz': (len(record['time_ns']), 4), 'joint_position': (len(record['time_ns']), len(joints))}
        if objects:
            shapes['object_pose_world'] = (len(record['time_ns']), len(objects), 7)
        require(set(record) == set(shapes) | {'metadata'}, 'replay arrays')
        for name, shape in shapes.items():
            ds = record[name]
            require(ds.shape == shape and ds.dtype == np.dtype('i8' if name == 'time_ns' else 'f4'), 'replay shape/dtype: ' + name)
            finite(ds)
            if name in ('root_quaternion_wxyz', 'object_pose_world'):
                for start in range(0, len(ds), 65536):
                    q = ds[start:start + 65536]
                    q = q if name == 'root_quaternion_wxyz' else q[..., 3:]
                    require(np.all(np.abs(np.linalg.norm(q, axis=-1) - 1) <= 1e-5), 'replay quaternion norm')
        times = record['time_ns'][:]
        require(len(times) >= 2 and np.all(times[1:] > times[:-1]), 'replay clock order')
        clock = metadata['clock']
        require(set(clock) == {'original_rate_hz', 'period_s', 'origin'} and clock['origin'], 'replay clock fields')
        if clock['period_s'] is None:
            require(clock['original_rate_hz'] is None, 'variable clock has nominal rate')
        else:
            p = clock['period_s']
            require(set(p) == {'numerator', 'denominator'}, 'replay period fields')
            expected = time_ns(np.arange(len(times)), p['numerator'], p['denominator'])
            rate = p['denominator'] / p['numerator']
            require(np.isclose(clock['original_rate_hz'], rate, rtol=1e-12), 'replay original rate')
            # Cropped records need not start on physical step zero. Differences
            # alone may differ by 1 ns because the original global steps round.
            diffs = np.diff(times)
            ideal = Fraction(p['numerator'] * 10**9, p['denominator'])
            require(np.all((diffs == ideal.numerator // ideal.denominator) | (diffs == -(-ideal.numerator // ideal.denominator))), 'replay skipped/duplicated physical frames')
            require(np.all(np.abs((times - times[0]) - expected) <= 1), 'replay accumulated clock drift')
        for row in rows:
            if text(row['record_id']) != rid:
                continue
            seq = sequences[int(row['sequence_index'])]
            zero = int(row['sample_zero_replay_time_ns'])
            end = zero + (int(seq['sample_stop'] - seq['sample_start']) - 1) * 40_000_000
            require(int(times[0]) <= zero <= end <= int(times[-1]), 'replay does not envelope IMU samples')


def validate_provenance(handle, sequences):
    require(set(handle['provenance']) == {'metadata'}, 'provenance objects')
    metadata = get_json(handle, 'provenance/metadata')
    require(set(metadata) == {'producer', 'input_files', 'sequence_sources', 'limitations'}, 'provenance fields')
    sources = metadata['sequence_sources']
    require([s['sequence_index'] for s in sources] == list(range(len(sequences))), 'provenance coverage')
    for source in sources:
        require(set(source) == {'sequence_index', 'source_kind', 'group_keys', 'refs'}, 'source provenance fields')
        require(source['source_kind'] in ('real', 'synthetic'), 'source kind')
        if source['source_kind'] == 'synthetic':
            require(text(handle.attrs['evaluation_role']) == 'training_only', 'synthetic evaluation leakage')
            require(source['group_keys'], 'missing synthetic leakage groups')
            refs = source['refs']
            physical = {'episode', 'sensors', 'review', 'route', 'model_manifest_sha256'}
            kinematic = {'motion', 'sensors', 'review', 'route', 'model_sha256'}
            expected = physical if physical <= set(refs) else kinematic
            require(expected <= set(refs), 'incomplete synthetic provenance')
            for kind in ('sensors', 'review',
                         'episode' if expected is physical else 'motion'):
                require(uuid_string(refs[kind]['artifact_id']) and sha_string(refs[kind]['sha256']), 'invalid synthetic reference')
            require(sha_string(refs['model_manifest_sha256' if expected is physical
                                    else 'model_sha256']), 'invalid model provenance')


def _validate(handle, *, full=True, filename=None):
    local_tree(handle)
    version = text(handle.attrs.get('imu_schema_version', ''))
    profile = text(handle.attrs.get('artifact_profile', ''))
    core = {'samples', 'sequences', 'annotations'}
    if version == '3.2.0':
        require(profile in ('training_dataset', 'client_delivery'), 'unknown v3.2 profile')
        require(set(handle) == (core if profile == 'training_dataset' else core | {'labels', 'media'}), 'v3.2 root objects')
        if profile == 'training_dataset':
            require(Path(filename or handle.filename).name == text(handle.attrs.get('dataset_id', '')) + '.h5', 'v3.2 training filename')
    elif version == '3.3.0':
        require(profile in ('imu_dataset', 'imu_dataset_provisional')
                and uuid_string(text(handle.attrs.get('artifact_id', ''))), 'v3.3 profile/identity')
        if profile == 'imu_dataset_provisional':
            require(set(handle) == core | {'candidate_index', 'weak_labels', 'provenance'},
                    'provisional root objects')
            require(text(handle.attrs.get('evaluation_role', '')) == 'unverified_synthetic',
                    'provisional evaluation role')
        else:
            require(core | {'labels'} <= set(handle) <= core | {'labels', 'media', 'replay', 'assets', 'provenance'}, 'v3.3 root objects')
            require('replay' not in handle or 'assets' in handle, 'replay requires assets')
    else:
        require(False, 'unsupported IMU version: ' + version)
    provisional = profile == 'imu_dataset_provisional'
    sequences, annotations = validate_core(handle, allow_provisional=provisional)
    if provisional:
        check_table(handle['candidate_index'], PROVISIONAL_INDEX)
        require(set(handle['weak_labels']) == {'index', 'source_candidates'},
                'provisional weak-label objects')
        check_table(handle['weak_labels/index'], WEAK_LABEL_INDEX)
        candidates = handle['candidate_index'][:]
        weak = handle['weak_labels/index'][:]
        require(len(candidates) == len(weak) == len(sequences), 'provisional sequence coverage')
        require(len(annotations) == 0 and all(text(row['activity_code']) == 'unverified'
                and not bool(row['is_fall']) for row in sequences),
                'provisional core cannot imply reviewed labels')
        for index, (candidate, label) in enumerate(zip(candidates, weak)):
            require(int(candidate['sequence_index']) == int(label['sequence_index']) == index,
                    'provisional index order')
            require(text(candidate['candidate_id']) and sha_string(text(candidate['version_id']))
                    and sha_string(text(candidate['commit_sha256']))
                    and text(candidate['source_dataset']), 'provisional candidate identity')
            state, code, rule = (text(label[name]) for name in ('state', 'code', 'rule_id'))
            require((state == 'weak' and bool(code) and bool(rule))
                    or (state == 'unresolved' and not code and not rule),
                    'provisional label state')
        sources = get_json(handle, 'weak_labels/source_candidates')
        identities = {(text(row['candidate_id']), text(row['version_id'])) for row in candidates}
        require(isinstance(sources, list) and len(sources) == len(identities)
                and {(row.get('candidate_id'), row.get('version_id')) for row in sources} == identities
                and all(isinstance(row.get('label_candidates'), list) for row in sources),
                'provisional raw label coverage')
        metadata = get_json(handle, 'provenance/metadata')
        require(metadata.get('schema') == 'imu_motion_simulator.provisional_provenance.v1'
                and sha_string(metadata.get('rules_sha256', ''))
                and isinstance(metadata.get('rules'), dict)
                and hashlib.sha256((json_dump(metadata['rules']) + '\n').encode()).hexdigest()
                == metadata['rules_sha256'],
                'provisional rule provenance')
        rule_lookup = {row['rule_id']: row['target_code']
                       for row in metadata['rules'].get('rules', [])}
        require(all(rule_lookup.get(text(row['rule_id'])) == text(row['code'])
                    for row in weak if text(row['state']) == 'weak'),
                'provisional weak label differs from frozen rule')
    if 'labels' in handle:
        validate_labels(handle, sequences, annotations)
    if full:
        if 'media' in handle:
            indices = validate_media(handle, sequences)
            if version == '3.2.0':
                require(indices == list(range(len(sequences))), 'v3.2 client requires all videos')
        registry = validate_assets(handle) if 'assets' in handle else {}
        if 'replay' in handle:
            validate_replay(handle, sequences, registry)
        if 'provenance' in handle and not provisional:
            validate_provenance(handle, sequences)
    return dict(version=version, profile=profile, sequences=len(sequences), samples=len(handle['samples']),
                logical_content_sha256=text(handle.attrs['logical_content_sha256']),
                capabilities=dict(core=True, video='media' in handle, replay='replay' in handle,
                                  review=False, scientific_derivation=False), validation='full' if full else 'core')


def validate_delivery(path, *, full=True, _filename=None):
    with h5py.File(path, 'r') as handle:
        return _validate(handle, full=full, filename=_filename)


def write_delivery(path, *, samples, sequences, annotations, dataset_id, labels=None,
                   version='3.3.0', profile=None, media=None, replay=None, assets=None,
                   blobs=None, provenance=None, evaluation_role='training_only'):
    """Low-level contract construction; caller must supply truthful review provenance.

    This API cannot authorize publication or establish human acceptance. All
    optional payloads must be omitted rather than passed as empty placeholders.
    """
    profile = profile or ('imu_dataset' if version == '3.3.0' else 'training_dataset')
    with atomic_h5(path, lambda tmp: validate_delivery(tmp, _filename=path)) as handle:
        handle.attrs.update(imu_schema_version=version, artifact_profile=profile,
                            dataset_id=dataset_id, sampling_rate_hz=np.float64(25), axis_frame='sensor_local',
                            hdf5_compatibility='1.14', evaluation_role=evaluation_role,
                            feature_columns=json_dump(COLUMNS), sequence_count=np.int64(len(sequences)),
                            sample_count=np.int64(len(samples)), annotation_count=np.int64(len(annotations)))
        if version == '3.3.0':
            handle.attrs['artifact_id'] = str(uuid4())
        # No coercion of the scientific core: reject incorrect input dtypes.
        ds = handle.create_dataset('samples', shape=samples.shape, dtype=samples.dtype, chunks=True)
        for start in range(0, len(samples), 65536):
            ds[start:start + 65536] = samples[start:start + 65536]
        ds.attrs.update(columns=json_dump(COLUMNS), units=json_dump(UNITS))
        handle.create_dataset('sequences', data=sequences)
        handle.create_dataset('annotations', data=annotations)
        handle.attrs['logical_content_sha256'] = logical_content_sha256(samples, sequences, annotations, dataset_id=dataset_id)
        if labels is not None:
            handle.create_dataset('labels/catalog', data=labels['catalog'])
            handle.create_dataset('labels/sequence_versions', data=labels['sequence_versions'])
        if media is not None:
            rows = []
            for item in media:
                i, payload = item['sequence_index'], item['bytes']
                video = handle.create_dataset(f'media/videos/{i}', data=np.frombuffer(payload, dtype='u1'))
                handle.create_dataset(f'media/timing/{i}', data=item['timing'])
                handle.flush()
                rows.append((i, len(payload), video.id.get_offset(), hashlib.sha256(payload).hexdigest(), 'video/mp4', 'mp4',
                             item['media_duration_ns'], item['sample_zero_recording_time_ns'], item['sample_zero_media_time_ns']))
            handle.create_dataset('media/index', data=table(rows, MEDIA))
        if assets is not None:
            write_assets(handle, assets, blobs or {})
        if replay is not None:
            handle.create_dataset('replay/index', data=replay['index'])
            for rid, item in replay['records'].items():
                require(rid and '/' not in rid and rid not in ('.', '..'), 'invalid replay id')
                put_json(handle, f'replay/records/{rid}/metadata', item['metadata'])
                for name, values in item['arrays'].items():
                    require('/' not in name, 'invalid replay array name')
                    handle.create_dataset(f'replay/records/{rid}/{name}', data=values, chunks=True)
        if provenance is not None:
            put_json(handle, 'provenance/metadata', provenance)
    return validate_delivery(path)


def migrate_v32(source, target, *, labels=None, provenance):
    """Create a new 3.3 file. Taxonomy and source provenance are never guessed."""
    validate_delivery(source)
    with h5py.File(source, 'r') as handle:
        require(text(handle.attrs['imu_schema_version']) == '3.2.0', 'migration requires v3.2 input')
        if 'labels' in handle:
            require(labels is None, 'client taxonomy must be preserved')
            labels = {name: handle['labels/' + name][:] for name in ('catalog', 'sequence_versions')}
        require(labels is not None and provenance is not None, 'migration requires trusted taxonomy and provenance')
        media = None
        if 'media' in handle:
            media = []
            for row in handle['media/index']:
                index = int(row['sequence_index'])
                media.append(dict(sequence_index=index, bytes=handle[f'media/videos/{index}'][:].tobytes(),
                                  timing=handle[f'media/timing/{index}'][:],
                                  **{key: int(row[key]) for key in ('media_duration_ns', 'sample_zero_recording_time_ns', 'sample_zero_media_time_ns')}))
        result = write_delivery(target, samples=handle['samples'], sequences=handle['sequences'][:], annotations=handle['annotations'][:],
                                dataset_id=text(handle.attrs['dataset_id']), labels=labels, media=media, provenance=provenance,
                                evaluation_role=text(handle.attrs['evaluation_role']))
        require(result['logical_content_sha256'] == text(handle.attrs['logical_content_sha256']), 'migration changed core identity')
        return result


class DeliveryReader:
    """Open once, validate once, then read bounded array windows without rescans."""
    def __init__(self, path, *, full=False):
        self.handle = h5py.File(path, 'r')
        try:
            self.description = _validate(self.handle, full=full)
        except BaseException:
            self.handle.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.handle.close()

    def samples(self, sequence_index, start=0, stop=None):
        require(type(sequence_index) is int and 0 <= sequence_index < len(self.handle['sequences']), 'sequence index')
        sequence = self.handle['sequences'][sequence_index]
        length = int(sequence['sample_stop'] - sequence['sample_start'])
        stop = length if stop is None else stop
        require(type(start) is int and type(stop) is int and 0 <= start <= stop <= length, 'sample range')
        offset = int(sequence['sample_start'])
        return self.handle['samples'][offset + start:offset + stop]

    def replay_window(self, record_id, start_ns, stop_ns):
        require(self.description['validation'] == 'full', 'replay reads require full validation')
        require('replay' in self.handle and record_id in self.handle['replay/records'], 'unknown replay id')
        record = self.handle['replay/records'][record_id]
        start, stop = window_rows(record['time_ns'], start_ns, stop_ns)
        return dict(row_start=start, row_stop=stop,
                    out_of_range=start_ns < int(record['time_ns'][0]) or stop_ns > int(record['time_ns'][-1]),
                    arrays={name: ds[start:stop] for name, ds in record.items() if name != 'metadata'})
