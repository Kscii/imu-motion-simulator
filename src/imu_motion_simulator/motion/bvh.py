"""Explicit diagnostic BVH-to-SMPL+H adapter for one representative source."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation

from ..contracts.common import sha256_file, time_ns
from ..contracts.internal import new_metadata, write_internal
from .smplh import SMPLH_JOINT_NAMES


MODEL_REST_TO_CANONICAL = np.asarray([
    [1., 0., 0.],
    [0., 0., -1.],
    [0., 1., 0.],
])


def _parse(raw):
    lines = raw.decode('utf-8', errors='strict').replace('\r', '').splitlines()
    try:
        motion_line = lines.index('MOTION')
    except ValueError as error:
        raise ValueError('BVH has no MOTION section') from error
    names, parents, offsets, channels = [], [], [], []
    stack, pending, end_site = [], None, False
    for line in lines[1:motion_line]:
        words = line.strip().split()
        if not words:
            continue
        if words[0] in ('ROOT', 'JOINT'):
            pending = len(names); names.append(words[1])
            parents.append(stack[-1] if stack else -1)
            offsets.append(None); channels.append([]); end_site = False
        elif words[:2] == ['End', 'Site']:
            pending = None; end_site = True
        elif words[0] == '{':
            stack.append(None if end_site else pending)
        elif words[0] == '}':
            if not stack:
                raise ValueError('BVH hierarchy brace mismatch')
            stack.pop(); end_site = False
        elif words[0] == 'OFFSET' and stack and stack[-1] is not None:
            offsets[stack[-1]] = [float(value) for value in words[1:4]]
        elif words[0] == 'CHANNELS' and stack and stack[-1] is not None:
            count = int(words[1]); channels[stack[-1]] = words[2:]
            if len(channels[stack[-1]]) != count:
                raise ValueError('BVH channel count mismatch')
    header = lines[motion_line + 1].split(':')
    timing = lines[motion_line + 2].split(':')
    if header[0].strip() != 'Frames' or timing[0].strip() != 'Frame Time':
        raise ValueError('BVH motion header')
    frames, frame_time = int(header[1]), float(timing[1])
    values = np.asarray([[float(value) for value in line.split()]
                         for line in lines[motion_line + 3:] if line.strip()],
                        dtype=np.float64)
    if values.shape != (frames, sum(map(len, channels))) \
            or not np.isfinite(values).all() or frame_time <= 0:
        raise ValueError('BVH motion array')
    if any(offset is None for offset in offsets):
        raise ValueError('BVH joint without offset')
    return names, np.asarray(parents), np.asarray(offsets), channels, frame_time, values


def _kinematics(names, parents, offsets, channels, values):
    count, joints = len(values), len(names)
    position = np.empty((count, joints, 3)); rotation = np.empty((count, joints, 3, 3))
    column = 0
    for joint in range(joints):
        spec = channels[joint]; data = values[:, column:column + len(spec)]; column += len(spec)
        translation = np.zeros((count, 3)); axes, angles = [], []
        for index, channel in enumerate(spec):
            if channel.endswith('position'):
                translation[:, 'XYZ'.index(channel[0])] = data[:, index]
            elif channel.endswith('rotation'):
                axes.append(channel[0]); angles.append(data[:, index])
            else:
                raise ValueError('Unsupported BVH channel: ' + channel)
        local = (Rotation.from_euler(''.join(axes), np.stack(angles, axis=1),
                                     degrees=True).as_matrix()
                 if axes else np.repeat(np.eye(3)[None], count, axis=0))
        parent = int(parents[joint])
        if parent < 0:
            position[:, joint] = offsets[joint] + translation
            rotation[:, joint] = local
        else:
            position[:, joint] = position[:, parent] + np.einsum(
                'tij,tj->ti', rotation[:, parent], offsets[joint] + translation)
            rotation[:, joint] = rotation[:, parent] @ local
    return position, rotation


def load_mapping(path):
    value = json.loads(Path(path).read_text())
    required = {'schema', 'adapter_id', 'description', 'position_scale_m',
                'world_from_source_matrix', 'joint_map'}
    if set(value) != required or value['schema'] != 'imu_motion_simulator.bvh_mapping.v1' \
            or not value['adapter_id'] or not 0 < value['position_scale_m'] <= 1:
        raise ValueError('Invalid BVH mapping')
    basis = np.asarray(value['world_from_source_matrix'], dtype=np.float64)
    if basis.shape != (3, 3) or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-8) \
            or np.linalg.det(basis) < .999999:
        raise ValueError('Invalid BVH world basis')
    if not isinstance(value['joint_map'], dict) or value['joint_map'].get('Hips') != 'pelvis' \
            or len(set(value['joint_map'].values())) != len(value['joint_map']):
        raise ValueError('Invalid BVH joint mapping')
    if not set(value['joint_map'].values()) <= set(SMPLH_JOINT_NAMES):
        raise ValueError('BVH mapping contains unknown SMPL+H joints')
    return value


def decode_bvh(bvh, model_archive, mapping_path, output, *, zip_member=None,
               source_dataset='BVH', source_gender='neutral'):
    bvh = Path(bvh)
    if zip_member is None:
        raw = bvh.read_bytes(); source_member = bvh.name
    else:
        with zipfile.ZipFile(bvh) as archive:
            raw = archive.read(zip_member)
        source_member = zip_member
    names, parents, offsets, channels, frame_time, values = _parse(raw)
    source_position, source_rotation = _kinematics(
        names, parents, offsets, channels, values)
    mapping = load_mapping(mapping_path); lookup = {name: i for i, name in enumerate(names)}
    missing = set(mapping['joint_map']) - set(names)
    if missing:
        raise ValueError('BVH source joints missing: ' + ', '.join(sorted(missing)))
    world = np.repeat(MODEL_REST_TO_CANONICAL[None, None],
                      len(values) * 52, axis=0).reshape(-1, 52, 3, 3)
    basis = np.asarray(mapping['world_from_source_matrix'])
    mapped_targets = set()
    for source_name, target_name in mapping['joint_map'].items():
        source = lookup[source_name]; target = SMPLH_JOINT_NAMES.index(target_name)
        first_inverse = source_rotation[0, source].T
        delta = source_rotation[:, source] @ first_inverse
        world[:, target] = (basis @ delta @ basis.T
                            @ MODEL_REST_TO_CANONICAL)
        mapped_targets.add(target)
    target_parents = [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
        13, 14, 16, 17, 18, 19,
        20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,
        21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50]
    local = np.repeat(np.eye(3)[None, None], len(values) * 52,
                      axis=0).reshape(-1, 52, 3, 3)
    local[:, 0] = world[:, 0]
    for joint in sorted(mapped_targets - {0}):
        parent = target_parents[joint]
        local[:, joint] = world[:, parent].transpose(0, 2, 1) @ world[:, joint]
    root_source = source_position[:, lookup['Hips']]
    root = (root_source - root_source[0]) * mapping['position_scale_m']
    root = np.einsum('ij,tj->ti', np.asarray(mapping['world_from_source_matrix']), root)
    quaternion = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_quat().reshape(-1, 52, 4)
    quaternion = quaternion[..., [3, 0, 1, 2]]
    period = Fraction(str(frame_time)).limit_denominator(1_000_000)
    step = np.arange(len(values), dtype=np.int64)
    gender = source_gender
    with tarfile.open(model_archive, 'r:xz') as archive:
        model_raw = archive.extractfile(f'{gender}/model.npz').read()
    with np.load(io.BytesIO(model_raw), allow_pickle=True,
                 encoding='latin1') as model_data:
        model = {name: model_data[name] for name in model_data.files}
    from .kinematics import shaped_rest
    rest_vertices, rest_joints = shaped_rest(model, np.zeros(16))
    canonical_vertices = rest_vertices @ MODEL_REST_TO_CANONICAL.T
    canonical_joints = rest_joints @ MODEL_REST_TO_CANONICAL.T
    pelvis_height = float(canonical_joints[0, 2]
                          - canonical_vertices[:, 2].min())
    if not .5 < pelvis_height < 1.5:
        raise ValueError('SMPL+H model is not Z-up after BVH mapping')
    root[:, 2] += pelvis_height
    model_hash = hashlib.sha256(model_raw).hexdigest()
    source_hash = sha256_file(bvh)
    motion_id = hashlib.sha256((source_hash + '\0' + source_member + '\0'
                                + sha256_file(mapping_path)).encode()).hexdigest()[:32]
    arrays = {
        'time_ns': time_ns(step, period.numerator, period.denominator),
        'root_position_m': root, 'root_quaternion_wxyz': quaternion[:, 0].copy(),
        'joint_local_quaternion_wxyz': quaternion,
        'betas': np.zeros(16), 'dmpls': np.zeros((len(values), 8)),
        'valid': np.ones(len(values), dtype=bool)}
    metadata = new_metadata(
        producer={'name': 'imu_motion_simulator.motion.bvh', 'version': '1.0.0',
                  'code_sha256': sha256_file(Path(__file__))},
        kind_metadata={'motion_contract_version': 2, 'motion_id': motion_id,
                       'source_dataset': source_dataset, 'source_member': source_member,
                       'source_gender': gender,
                       'source_fps_hz': period.denominator / period.numerator,
                       'joint_names': SMPLH_JOINT_NAMES, 'model_family': 'smplh',
                       'model_sha256': model_hash,
                       'original_archive_sha256': source_hash},
        provenance={'source_member_sha256': hashlib.sha256(raw).hexdigest(),
                    'adapter_id': mapping['adapter_id'],
                    'mapping_sha256': sha256_file(mapping_path),
                    'mapped_joints': mapping['joint_map'],
                    'limitations': [
                        'Representative diagnostic adapter; not approved for AMASS-scale production',
                        'First-frame source joint frames are normalized to the SMPL+H rest orientation',
                        'Unmapped fingers remain in the neutral pose and source body shape is unavailable']},
        clocks={'motion': {'numerator': period.numerator,
                           'denominator': period.denominator,
                           'origin': 'source-frame-zero'}},
        resolved_config={'mapping': mapping,
                         'model_rest_to_canonical_matrix':
                             MODEL_REST_TO_CANONICAL.tolist(),
                         'neutral_pelvis_height_m': pelvis_height,
                         'ground_policy': 'neutral-SMPL+H-rest-surface-at-z-zero',
                         'adapter_confidence': 'diagnostic-representative'})
    return write_internal(output, 'motion', metadata, arrays)
