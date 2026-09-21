"""Native and explicit Stage-II Extended SMPL+H decoding."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import tarfile
from fractions import Fraction

import numpy as np
from scipy.spatial.transform import Rotation

from ..contracts.common import sha256_file, time_ns
from ..contracts.internal import artifact_parent, new_metadata, write_internal


SMPLH_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
] + [f'{side}_{finger}{joint}' for side in ('left', 'right')
     for finger in ('index', 'middle', 'pinky', 'ring', 'thumb') for joint in (1, 2, 3)]
AMASS_FIELDS = {'poses', 'trans', 'betas', 'dmpls', 'gender',
                'mocap_framerate'}
STAGEII_FIELDS = {'poses', 'trans', 'betas', 'gender', 'mocap_frame_rate',
                  'surface_model_type', 'num_betas'}
NATIVE_ADAPTER = 'amass-smplh-g-v1'
STAGEII_ADAPTER = 'stageii-smplh-v1'


def normalize_amass_gender(value):
    """Normalize the string and byte scalar encodings found across AMASS."""
    scalar = np.asarray(value).item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode('ascii')
    return str(scalar).strip().lower()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _producer():
    return {'name': 'imu_motion_simulator.motion.smplh', 'version': '1.0.0',
            'code_sha256': sha256_file(Path(__file__))}


def _load_npz_member(archive, member, mode, *, allow_pickle):
    with tarfile.open(archive, mode) as handle:
        found = handle.getmember(member)
        raw = handle.extractfile(found).read()
    return raw, _load_npz_bytes(raw, allow_pickle=allow_pickle)


def _load_npz_bytes(raw, *, allow_pickle):
    with np.load(io.BytesIO(raw), allow_pickle=allow_pickle, encoding='latin1') as data:
        return {name: data[name] for name in data.files}


def _load_amass_bytes(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        fields = set(data.files)
        if not AMASS_FIELDS <= fields:
            raise ValueError('Missing required AMASS member fields')
        return {name: data[name] for name in AMASS_FIELDS}, fields


def _load_stageii_bytes(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        fields = set(data.files)
        if not STAGEII_FIELDS <= fields:
            raise ValueError('Missing required Stage-II SMPL+H member fields')
        source = {name: data[name] for name in STAGEII_FIELDS}
        poses = np.asarray(source['poses'])
        frames = len(poses) if poses.ndim == 2 else 0
        split = {'root_orient': (frames, 3), 'pose_body': (frames, 63),
                 'pose_hand': (frames, 90)}
        if set(split) <= fields:
            parts = []
            for name, shape in split.items():
                value = np.asarray(data[name])
                if value.shape != shape:
                    raise ValueError('Unexpected Stage-II pose split: ' + name)
                parts.append(value)
            if poses.shape != (frames, 156) \
                    or not np.allclose(poses, np.concatenate(parts, axis=1),
                                       rtol=0, atol=1e-8):
                raise ValueError('Stage-II poses differ from official parameter split')
        return source, fields


def _load_motion_bytes(raw, adapter_id):
    if adapter_id == NATIVE_ADAPTER:
        source, fields = _load_amass_bytes(raw)
        return source, fields, True
    if adapter_id == STAGEII_ADAPTER:
        source, fields = _load_stageii_bytes(raw)
        source = dict(source)
        source['mocap_framerate'] = source.pop('mocap_frame_rate')
        source['dmpls'] = np.zeros((len(source['poses']), 8), dtype=np.float64)
        return source, fields, False
    raise ValueError('Unsupported SMPL+H source adapter: ' + str(adapter_id))


def _parents(kintree):
    tree = np.asarray(kintree); ids = [int(value) for value in tree[1]]
    lookup = {value: index for index, value in enumerate(ids)}
    return np.asarray([-1] + [lookup[int(value)] for value in tree[0, 1:]], dtype=np.int64)


def _model_tensors(model, *, shapedirs=None):
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError('SMPL+H decoding requires the locked motion dependency group') from error
    regressor = model['J_regressor']
    if hasattr(regressor, 'toarray'): regressor = regressor.toarray()
    posedirs = np.asarray(model['posedirs'])
    if posedirs.shape != (6890, 3, 459): raise ValueError('Unexpected Extended SMPL+H posedirs')
    def tensor(value): return torch.as_tensor(np.asarray(value), dtype=torch.float64)
    return torch, dict(
        v_template=tensor(model['v_template']),
        shapedirs=tensor(model['shapedirs'] if shapedirs is None else shapedirs),
        posedirs=tensor(posedirs.reshape(-1, 459).T), J_regressor=tensor(regressor),
        parents=torch.as_tensor(_parents(model['kintree_table'])),
        lbs_weights=tensor(model['weights']),
    )


def _basis(joints):
    count = min(30, len(joints))
    left = np.mean((joints[:count, 1] - joints[:count, 2])[:, :2], axis=0)
    if np.linalg.norm(left) < 1e-8: raise ValueError('Cannot establish source left axis')
    left = np.r_[left / np.linalg.norm(left), 0.]
    up = np.array([0., 0., 1.]); forward = np.cross(left, up)
    basis = np.stack([forward, left, up])
    if np.linalg.det(basis) < .999999: raise ValueError('Invalid source-to-canonical basis')
    return basis


def _decode(torch, lbs_args, poses, betas, translations, chunk_frames):
    from smplx.lbs import lbs
    vertices, joints = [], []
    with torch.no_grad():
        for start in range(0, len(poses), chunk_frames):
            stop = min(len(poses), start + chunk_frames); n = stop - start
            shape = torch.as_tensor(np.repeat(betas[None], n, axis=0), dtype=torch.float64)
            pose = torch.as_tensor(poses[start:stop], dtype=torch.float64)
            v, j = lbs(shape, pose, **lbs_args)
            offset = translations[start:stop, None]
            vertices.append(v.detach().cpu().numpy() + offset)
            joints.append(j.detach().cpu().numpy() + offset)
    return np.concatenate(vertices), np.concatenate(joints)


def _clock_period(fps):
    rate = Fraction(str(float(fps))).limit_denominator(1_000_000)
    return rate.denominator, rate.numerator


def decode_amass_member(amass_archive, model_archive, member, output, *, chunk_frames=128,
                        source_artifact=None, source_dataset=None,
                        original_archive_sha256=None, source_bytes=None,
                        model_cache=None, adapter_id=NATIVE_ADAPTER):
    """Decode one immutable AMASS member without extracting either source archive."""
    amass_archive, model_archive, output = map(Path, (amass_archive, model_archive, output))
    if source_bytes is None:
        with tarfile.open(amass_archive, 'r:bz2') as handle:
            found = handle.getmember(member)
            raw = handle.extractfile(found).read()
    else:
        raw = bytes(source_bytes)
    source, source_fields, dmpl_available = _load_motion_bytes(raw, adapter_id)
    gender = normalize_amass_gender(source['gender'])
    fps = float(source['mocap_framerate'])
    poses = np.asarray(source['poses'], dtype=np.float64); trans = np.asarray(source['trans'], dtype=np.float64)
    betas = np.asarray(source['betas'], dtype=np.float64)
    if gender not in ('male', 'female', 'neutral') or not np.isfinite(fps) or fps <= 0 \
            or poses.ndim != 2 or poses.shape[1] != 156:
        raise ValueError('Expected AMASS SMPL+H G motion')
    if trans.shape != (len(poses), 3) or betas.shape != (16,) or source['dmpls'].shape != (len(poses), 8):
        raise ValueError('Unexpected AMASS SMPL+H array shape')
    dmpls = np.asarray(source['dmpls'], dtype=np.float64)
    if adapter_id == STAGEII_ADAPTER:
        if str(np.asarray(source['surface_model_type']).item()) != 'smplh' \
                or int(np.asarray(source['num_betas']).item()) != 16:
            raise ValueError('Expected 16-beta Stage-II SMPL+H motion')
    if len(poses) < 2 or not all(np.isfinite(value).all()
                                 for value in (poses, trans, betas, dmpls)):
        raise ValueError('Invalid AMASS numeric values')
    model_member = f'{gender}/model.npz'
    cached = None if model_cache is None else model_cache.get(gender)
    if cached is None:
        model_raw, model = _load_npz_member(
            model_archive, model_member, 'r:xz', allow_pickle=True)
        if np.asarray(model['shapedirs']).shape != (6890, 3, 16) \
                or np.asarray(model['weights']).shape != (6890, 52):
            raise ValueError('Expected native 16-beta Extended SMPL+H model')
        torch, lbs_args = _model_tensors(model)
        cached = (model_raw, model, torch, lbs_args)
        if model_cache is not None:
            model_cache[gender] = cached
    else:
        model_raw, model, torch, lbs_args = cached
    calibration_vertices, calibration_joints = _decode(
        torch, lbs_args, poses[:min(30, len(poses))], betas,
        trans[:min(30, len(poses))], chunk_frames)
    basis = _basis(calibration_joints)
    local = Rotation.from_rotvec(poses.reshape(-1, 3)).as_quat().reshape(len(poses), 52, 4)
    local[:, 0] = Rotation.from_matrix(basis @ Rotation.from_rotvec(poses[:, :3]).as_matrix()).as_quat()
    local = local[..., [3, 0, 1, 2]]
    # Decode a short calibration window with the full surface, then use exact
    # SMPL+H joint FK for all remaining frames.  This avoids materializing a
    # T x 6890 surface merely to establish a floor plane during corpus import.
    from .kinematics import joint_transforms, shaped_rest
    _, rest_joints = shaped_rest(model, betas)
    root = np.einsum('ij,tj->ti', basis, trans + rest_joints[0])
    all_joints, _ = joint_transforms(model, betas, root, local)
    calibration_vertices = np.einsum('ij,tvj->tvi', basis, calibration_vertices)
    calibration_joints = np.einsum('ij,tkj->tki', basis, calibration_joints)
    support = [7, 8, 10, 11]
    sole_offset = np.median(
        calibration_vertices[:, :, 2].min(axis=1)
        - calibration_joints[:, support, 2].min(axis=1))
    estimated_surface_minimum = all_joints[:, support, 2].min(axis=1) + sole_offset
    ground = float(np.quantile(estimated_surface_minimum, .01))
    origin = root[0].copy(); origin[2] = ground
    root -= origin
    root_quaternion = local[:, 0].copy()
    step = np.arange(len(poses), dtype=np.int64)
    numerator, denominator = _clock_period(fps)
    model_hash = _sha(model_raw)
    archive_hash = original_archive_sha256 or sha256_file(amass_archive)
    identity = archive_hash + '\0' + member + '\0' + model_hash
    if adapter_id != NATIVE_ADAPTER:
        identity += '\0' + adapter_id
    motion_id = hashlib.sha256(identity.encode()).hexdigest()[:32]
    arrays = dict(time_ns=time_ns(step, numerator, denominator), root_position_m=root,
                  root_quaternion_wxyz=root_quaternion,
                  joint_local_quaternion_wxyz=local.astype(np.float64),
                  betas=betas, dmpls=dmpls,
                  valid=np.ones(len(poses), dtype=bool))
    parents = [] if source_artifact is None else [artifact_parent(source_artifact, 'source')]
    metadata = new_metadata(
        producer=_producer(),
        parents=parents,
        kind_metadata=dict(motion_contract_version=2, motion_id=motion_id,
                           source_dataset=source_dataset or member.split('/', 1)[0],
                           source_member=member, source_gender=gender,
                           source_fps_hz=fps, joint_names=SMPLH_JOINT_NAMES,
                           model_family='smplh', model_sha256=model_hash,
                           original_archive_sha256=archive_hash),
        provenance=dict(source_member_sha256=_sha(raw), model_member=model_member,
                        source_adapter=adapter_id,
                        ignored_source_fields=sorted(
                            source_fields - (AMASS_FIELDS if dmpl_available
                                             else STAGEII_FIELDS)),
                        official_parameter_split={'root_orient': [0, 3], 'pose_body': [3, 66], 'pose_hand': [66, 156]},
                        calibration_evidence='AMASS Extended SMPL+H official loader semantics; Z-up confirmed across decoded body extents',
                        limitations=([
                            'DMPL coefficients are retained; each consumer must declare bone-rigid or surface-DMPL use']
                            if dmpl_available else [
                            'Source has no DMPL coefficients; the canonical array is an explicit disabled-zero placeholder'])
                            + ['Canonical heading is derived from the first 30 decoded hip frames']),
        clocks={'motion': {'numerator': numerator, 'denominator': denominator,
                           'origin': 'source-frame-zero'}},
        resolved_config=dict(source_world='AMASS decoded Z-up', canonical_world='RH-Xforward-Yleft-Zup',
                             world_from_source_matrix=basis.tolist(), origin_canonical_m=origin.tolist(),
                             ground_policy='one-percentile estimated surface minimum from exact joint FK and calibrated sole offset',
                             sole_offset_m=float(sole_offset), shape_directions=16,
                             dynamic_directions=8,
                             dynamic_shape={
                                 'source_available': dmpl_available,
                                 'effective_policy': ('source' if dmpl_available
                                                      else 'disabled-zero'),
                                 'components': 8},
                             pose_directions=459,
                             surface_policy='consumer-selected', chunk_frames=chunk_frames))
    return write_internal(output, 'motion', metadata, arrays)
