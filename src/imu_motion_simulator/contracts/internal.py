"""Strict immutable HDF5 contracts for simulator research artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np

from .assets import validate_assets, write_assets
from .common import (COORDINATES, atomic_h5, finite, get_json, local_tree,
                     put_json, relative_path, require, sha_string, text,
                     time_ns, uuid_string)

SUPPORTED_KINDS = ('source', 'motion', 'sensors')
METADATA_FIELDS = {'producer', 'created_at_utc', 'parents', 'completion', 'coordinates',
                   'clocks', 'arrays', 'resolved_config', 'provenance', 'kind_metadata'}

MOTION_ARRAYS = {
    'time_ns': ('i8', ('T',)), 'step_index': ('i8', ('T',)),
    'root_position_m': ('f8', ('T', 3)), 'root_quaternion_wxyz': ('f8', ('T', 4)),
    'joint_local_quaternion_wxyz': ('f8', ('T', 'J', 4)),
    'joint_world_position_m': ('f8', ('T', 'J', 3)), 'betas': ('f8', (16,)),
}
MOTION_V2_ARRAYS = {
    'time_ns': ('i8', ('T',)),
    'root_position_m': ('f8', ('T', 3)),
    'root_quaternion_wxyz': ('f8', ('T', 4)),
    'joint_local_quaternion_wxyz': ('f8', ('T', 52, 4)),
    'betas': ('f8', (16,)),
    'dmpls': ('f8', ('T', 8)),
    'valid': ('bool', ('T',)),
}
SENSOR_V2_ARRAYS = {
    'time_ns': ('i8', ('N',)),
    'specific_force_m_s2': ('f8', ('N', 'M', 3)),
    'angular_velocity_rad_s': ('f8', ('N', 'M', 3)),
    'valid': ('bool', ('N', 'M')),
}


def new_metadata(*, producer, kind_metadata, provenance, parents=None,
                 completion='completed', reason=None, clocks=None,
                 arrays=None, resolved_config=None):
    return dict(producer=producer, created_at_utc=datetime.now(timezone.utc).isoformat(),
                parents=list(parents or []), completion=dict(status=completion, reason=reason),
                coordinates=COORDINATES, clocks=dict(clocks or {}), arrays=dict(arrays or {}),
                resolved_config=dict(resolved_config or {}), provenance=provenance,
                kind_metadata=kind_metadata)


def source_metadata(*, producer, kind_metadata, provenance):
    return new_metadata(producer=producer, kind_metadata=kind_metadata, provenance=provenance)


def _metadata(handle, kind):
    local_tree(handle)
    require(text(handle.attrs.get('ims_schema_version', '')) == '1.0.0', 'unsupported internal schema')
    require(text(handle.attrs.get('artifact_kind', '')) == kind, 'internal artifact kind mismatch')
    require(uuid_string(text(handle.attrs.get('artifact_id', ''))), 'invalid artifact UUID')
    require({'metadata', 'data'} <= set(handle) <= {'metadata', 'data', 'assets'}, f'{kind} root objects')
    metadata = get_json(handle, 'metadata')
    require(set(metadata) == METADATA_FIELDS, 'internal metadata fields')
    producer = metadata['producer']
    require(set(producer) == {'name', 'version', 'code_sha256'} and producer['name'] and producer['version']
            and sha_string(producer['code_sha256']), 'producer identity')
    created = datetime.fromisoformat(metadata['created_at_utc'])
    require(created.utcoffset() is not None and created.utcoffset().total_seconds() == 0, 'audit time must be UTC')
    require(metadata['coordinates'] == COORDINATES, 'internal coordinate declaration')
    for parent in metadata['parents']:
        require(set(parent) == {'role', 'artifact_id', 'sha256'} and parent['role']
                and uuid_string(parent['artifact_id']) and sha_string(parent['sha256']), 'parent reference')
    completion = metadata['completion']
    require(set(completion) == {'status', 'reason'} and completion['status'] in ('completed', 'terminated', 'failed'), 'completion status')
    require(completion['status'] == 'completed' or completion['reason'], 'missing failure reason')
    require(all(isinstance(metadata[key], dict) for key in ('clocks', 'arrays', 'resolved_config', 'provenance', 'kind_metadata')), 'metadata object types')
    if 'assets' in handle:
        validate_assets(handle)
    return metadata


def _shape_matches(actual, spec, dimensions):
    if len(actual) != len(spec): return False
    for value, wanted in zip(actual, spec):
        if isinstance(wanted, int) and value != wanted: return False
        if isinstance(wanted, str):
            if wanted in dimensions and dimensions[wanted] != value: return False
            dimensions.setdefault(wanted, value)
    return True


def _validate_arrays(handle, metadata, specs):
    data = handle['data']; require(isinstance(data, h5py.Group), 'data must be a group')
    require(set(data) == set(specs), 'missing/unknown numerical arrays')
    require(set(metadata['arrays']) == set(specs), 'array declaration mismatch')
    dims = {}
    for name, (dtype, shape) in specs.items():
        ds = data[name]; require(isinstance(ds, h5py.Dataset), f'{name}: expected dataset')
        require(ds.dtype == np.dtype(dtype), f'{name}: dtype')
        require(_shape_matches(ds.shape, shape, dims), f'{name}: shape')
        declared = metadata['arrays'][name]
        require(set(declared) == {'dtype', 'shape'} and declared['dtype'] == np.dtype(dtype).name
                and declared['shape'] == list(ds.shape), f'{name}: array declaration')
        if ds.dtype.kind in 'fc': finite(ds)
    return dims


def _unit_quaternion(dataset, message):
    for start in range(0, len(dataset), 65536):
        require(np.all(np.abs(np.linalg.norm(dataset[start:start + 65536], axis=-1) - 1.) <= 1e-8), message)


def validate_source(path):
    with h5py.File(path, 'r') as handle:
        metadata = _metadata(handle, 'source')
        require(isinstance(handle['data'], h5py.Group) and not len(handle['data']), 'source stores originals by reference; numerical arrays belong to motion')
        require(metadata['arrays'] == metadata['clocks'] == metadata['resolved_config'] == {}, 'unexpected source numerical/config payload')
        require(set(metadata['provenance']) == {'source_refs', 'group_keys', 'evidence', 'limitations'}, 'source provenance fields')
        source = metadata['kind_metadata']
        require(set(source) == {'source_id', 'dataset_id', 'files', 'license', 'access_scope', 'audit', 'missing_information'}, 'source metadata fields')
        require(source['source_id'] and source['dataset_id'] and source['files'], 'empty source identity/files')
        paths = set()
        for file in source['files']:
            require(set(file) == {'logical_path', 'sha256', 'byte_length', 'media_type', 'source_uri'}, 'source file fields')
            relative_path(file['logical_path'])
            require(file['logical_path'] not in paths and sha_string(file['sha256']) and type(file['byte_length']) is int
                    and file['byte_length'] > 0 and file['media_type'], 'invalid source file')
            paths.add(file['logical_path'])
            require(isinstance(file['source_uri'], str) and '?' not in file['source_uri'] and '#' not in file['source_uri']
                    and '@' not in file['source_uri'], 'credential-bearing source URI')
        require(source['license'] and source['access_scope'] and isinstance(source['audit'], dict)
                and isinstance(source['missing_information'], list), 'source audit/license metadata')
        return dict(version='1.0.0', kind='source', artifact_id=text(handle.attrs['artifact_id']), files=len(source['files']),
                    execution_ready=False, capabilities={'source_audit': True, 'motion': False, 'simulation': False})


def validate_motion(path):
    with h5py.File(path, 'r') as handle:
        metadata = _metadata(handle, 'motion')
        info = metadata['kind_metadata']
        if info.get('motion_contract_version') == 2:
            dims = _validate_arrays(handle, metadata, MOTION_V2_ARRAYS)
            required = {'motion_contract_version', 'motion_id', 'source_dataset',
                        'source_member', 'source_gender', 'source_fps_hz',
                        'joint_names', 'model_family', 'model_sha256',
                        'original_archive_sha256'}
            require(set(info) == required, 'motion-v2 metadata fields')
            require(info['model_family'] == 'smplh' and len(info['joint_names']) == 52
                    and info['joint_names'][0] == 'pelvis', 'motion-v2 model semantics')
            require(info['source_gender'] in ('male', 'female', 'neutral')
                    and isinstance(info['source_fps_hz'], (int, float))
                    and np.isfinite(info['source_fps_hz']) and info['source_fps_hz'] > 0
                    and sha_string(info['model_sha256'])
                    and sha_string(info['original_archive_sha256']), 'motion-v2 identity')
            period = metadata['clocks'].get('motion')
            require(isinstance(period, dict) and set(period) == {
                'numerator', 'denominator', 'origin'}
                and type(period['numerator']) is int and type(period['denominator']) is int
                and period['numerator'] > 0 and period['denominator'] > 0
                and period['origin'] == 'source-frame-zero', 'motion-v2 clock')
            steps = np.arange(dims['T'], dtype=np.int64)
            require(dims['T'] >= 2 and np.array_equal(
                handle['data/time_ns'][:], time_ns(
                    steps, period['numerator'], period['denominator'])),
                    'motion-v2 time drift')
            _unit_quaternion(handle['data/root_quaternion_wxyz'],
                             'motion-v2 root quaternion')
            _unit_quaternion(handle['data/joint_local_quaternion_wxyz'],
                             'motion-v2 local quaternion')
            require(np.array_equal(handle['data/root_quaternion_wxyz'][:],
                                   handle['data/joint_local_quaternion_wxyz'][:, 0]),
                    'motion-v2 root/local mismatch')
            require(metadata['completion']['status'] == 'completed'
                    and bool(handle['data/valid'][:].all()),
                    'motion-v2 must be complete and valid')
            dynamic = metadata['resolved_config'].get('dynamic_shape')
            if dynamic is None:
                dynamic_available = True
            else:
                require(set(dynamic) == {
                    'source_available', 'effective_policy', 'components'}
                    and type(dynamic['source_available']) is bool
                    and dynamic['components'] == 8
                    and dynamic['effective_policy'] in {
                        'source', 'disabled-zero'}
                    and dynamic['source_available']
                    == (dynamic['effective_policy'] == 'source'),
                    'motion-v2 dynamic-shape declaration')
                dynamic_available = dynamic['source_available']
                if not dynamic_available:
                    require(not np.any(handle['data/dmpls'][:]),
                            'disabled motion-v2 DMPL must be zero')
            return dict(version='1.0.0', motion_contract_version=2, kind='motion',
                        artifact_id=text(handle.attrs['artifact_id']), frames=dims['T'],
                        joints=52, model_family='smplh', execution_ready=True,
                        dynamic_shape_available=dynamic_available)
        dims = _validate_arrays(handle, metadata, MOTION_ARRAYS)
        required = {'motion_id', 'source_member', 'source_gender', 'source_fps', 'joint_names', 'model_sha256', 'original_archive_sha256', 'dmpl_applied'}
        require(set(info) == required and info['source_fps'] == 120 and len(info['joint_names']) == dims['J'] == 52, 'motion metadata')
        require(info['source_gender'] in ('male', 'female', 'neutral') and not info['dmpl_applied'] and sha_string(info['model_sha256']) and sha_string(info['original_archive_sha256']), 'motion identity')
        step = handle['data/step_index'][:]
        require(metadata['clocks'].get('motion') == {'numerator': 1, 'denominator': 120, 'origin': 'source-frame-zero'}, 'motion clock')
        require(np.array_equal(step, np.arange(len(step), dtype=np.int64)) and np.array_equal(handle['data/time_ns'][:], time_ns(step, 1, 120)), 'motion time drift')
        _unit_quaternion(handle['data/root_quaternion_wxyz'], 'motion root quaternion')
        _unit_quaternion(handle['data/joint_local_quaternion_wxyz'], 'motion local quaternion')
        require(metadata['completion']['status'] == 'completed', 'motion must be complete')
        return dict(version='1.0.0', kind='motion', artifact_id=text(handle.attrs['artifact_id']), frames=dims['T'], joints=dims['J'], execution_ready=True)


def validate_sensors(path):
    with h5py.File(path, 'r') as handle:
        metadata = _metadata(handle, 'sensors')
        info = metadata['kind_metadata']
        require(info.get('sensor_contract_version') == 2,
                'only kinematic sensor contract v2 is supported')
        dims = _validate_arrays(handle, metadata, SENSOR_V2_ARRAYS)
        required = {'sensor_contract_version', 'sensor_id', 'motion_id',
                    'selection_id', 'layout_id', 'mount_ids', 'profile_id',
                    'variant'}
        require(set(info) == required and uuid_string(info['sensor_id'])
                and info['motion_id'] and info['layout_id']
                and isinstance(info['mount_ids'], list) and info['mount_ids']
                and len(info['mount_ids']) == dims['M']
                and len(set(info['mount_ids'])) == dims['M']
                and info['profile_id'] and info['variant'] in ('ideal', 'calibrated'),
                'sensor-v2 metadata')
        require(info['selection_id'] is None or isinstance(info['selection_id'], str),
                'sensor-v2 selection identity')
        period = metadata['clocks'].get('sensor')
        require(isinstance(period, dict) and set(period) == {
            'numerator', 'denominator', 'origin'}
            and type(period['numerator']) is int and type(period['denominator']) is int
            and period['numerator'] > 0 and period['denominator'] > 0
            and period['origin'] == 'motion-start', 'sensor-v2 clock')
        times = handle['data/time_ns'][:]
        require(dims['N'] >= 2 and np.array_equal(times, time_ns(
            np.arange(dims['N'], dtype=np.int64), period['numerator'],
            period['denominator'])), 'sensor-v2 time drift')
        return dict(version='1.0.0', sensor_contract_version=2, kind='sensors',
                    artifact_id=text(handle.attrs['artifact_id']),
                    motion_id=info['motion_id'], layout_id=info['layout_id'],
                    samples=dims['N'], mounts=dims['M'],
                    valid=bool(handle['data/valid'][:].all()),
                    complete=metadata['completion']['status'] == 'completed')


VALIDATORS = {'source': validate_source, 'motion': validate_motion,
              'sensors': validate_sensors}


def validate_internal(path, kind=None):
    if kind is None:
        with h5py.File(path, 'r') as handle: kind = text(handle.attrs.get('artifact_kind', ''))
    require(kind in VALIDATORS, 'internal kind not implemented: ' + str(kind))
    return VALIDATORS[kind](path)


def write_internal(path, kind, metadata, arrays, *, assets=None, blobs=None):
    require(kind in VALIDATORS and kind != 'source', 'unsupported numerical artifact kind')
    values = {name: np.asarray(value) for name, value in arrays.items()}; metadata = dict(metadata)
    metadata['arrays'] = {name: {'dtype': value.dtype.name, 'shape': list(value.shape)} for name, value in values.items()}
    with atomic_h5(path, VALIDATORS[kind]) as handle:
        handle.attrs.update(ims_schema_version='1.0.0', artifact_kind=kind, artifact_id=str(uuid4()))
        put_json(handle, 'metadata', metadata); data = handle.create_group('data')
        for name, value in values.items():
            data.create_dataset(name, data=value, chunks=value.ndim > 0, compression='gzip' if value.ndim > 0 else None, shuffle=value.ndim > 0)
        if assets is not None: write_assets(handle, assets, blobs or {})
    return VALIDATORS[kind](path)


def write_source(path, metadata, *, assets=None, blobs=None):
    with atomic_h5(path, validate_source) as handle:
        handle.attrs.update(ims_schema_version='1.0.0', artifact_kind='source', artifact_id=str(uuid4()))
        put_json(handle, 'metadata', metadata); handle.create_group('data')
        if assets is not None: write_assets(handle, assets, blobs or {})
    return validate_source(path)


def read_source(path):
    description = validate_source(path)
    with h5py.File(path, 'r') as handle: return description, get_json(handle, 'metadata')


def read_internal(path, kind=None):
    description = validate_internal(path, kind)
    with h5py.File(path, 'r') as handle:
        return description, get_json(handle, 'metadata'), {name: handle['data/' + name][:] for name in handle['data']}


def artifact_parent(path, role):
    from .common import sha256_file
    description = validate_internal(path)
    return {'role': role, 'artifact_id': description['artifact_id'], 'sha256': sha256_file(Path(path))}
