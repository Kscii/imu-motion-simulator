"""SMPL+H kinematics and on-demand surface reconstruction."""
from __future__ import annotations

import io
from pathlib import Path
import tarfile

import numpy as np
from scipy.spatial.transform import Rotation

from .smplh import SMPLH_JOINT_NAMES, _model_tensors, _parents


def load_model_member(model_archive, gender):
    """Read one licensed model member without extracting the source archive."""
    model_archive = Path(model_archive)
    with tarfile.open(model_archive, 'r:xz') as handle:
        raw = handle.extractfile(f'{gender}/model.npz').read()
    with np.load(io.BytesIO(raw), allow_pickle=True, encoding='latin1') as data:
        return {name: data[name] for name in data.files}


def load_dmpl_basis(dmpl_archive, gender):
    with tarfile.open(Path(dmpl_archive), 'r:xz') as handle:
        raw = handle.extractfile(f'{gender}/model.npz').read()
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        basis = np.asarray(data['eigvec'], dtype=np.float64)
    if basis.shape != (6890, 3, 8) or not np.isfinite(basis).all():
        raise ValueError('Unexpected DMPL basis')
    return basis


def shaped_rest(model, betas):
    shapedirs = np.asarray(model['shapedirs'], dtype=np.float64)
    template = np.asarray(model['v_template'], dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)
    if shapedirs.shape != (6890, 3, 16) or betas.shape != (16,):
        raise ValueError('Expected Extended SMPL+H 16-beta shape')
    vertices = template + np.einsum('vci,i->vc', shapedirs, betas)
    regressor = model['J_regressor']
    if hasattr(regressor, 'toarray'):
        regressor = regressor.toarray()
    joints = np.asarray(regressor, dtype=np.float64) @ vertices
    if joints.shape != (52, 3):
        raise ValueError('Unexpected SMPL+H joint regressor')
    return vertices, joints


def joint_transforms(model, betas, root_position_m,
                     joint_local_quaternion_wxyz):
    """Return world joint positions and rotations for canonical motion-v2."""
    _, rest = shaped_rest(model, betas)
    parents = _parents(model['kintree_table'])
    local = np.asarray(joint_local_quaternion_wxyz, dtype=np.float64)
    root = np.asarray(root_position_m, dtype=np.float64)
    if local.ndim != 3 or local.shape[1:] != (52, 4) \
            or root.shape != (len(local), 3):
        raise ValueError('Canonical SMPL+H pose shape')
    local_rotation = Rotation.from_quat(
        local[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(-1, 52, 3, 3)
    position = np.empty((len(local), 52, 3), dtype=np.float64)
    world_rotation = np.empty((len(local), 52, 3, 3), dtype=np.float64)
    position[:, 0] = root
    world_rotation[:, 0] = local_rotation[:, 0]
    for joint in range(1, 52):
        parent = int(parents[joint])
        offset = rest[joint] - rest[parent]
        position[:, joint] = position[:, parent] + np.einsum(
            'tij,j->ti', world_rotation[:, parent], offset)
        world_rotation[:, joint] = np.einsum(
            'tij,tjk->tik', world_rotation[:, parent], local_rotation[:, joint])
    return position, world_rotation


def surface_sequence(model, dmpl_basis, arrays, *, chunk_frames=32,
                     use_dmpl=True):
    """Reconstruct exact SMPL+H LBS surface frames on demand.

    The model's pose correctives and the optional per-frame DMPL coefficients
    are applied.  Callers choose short ranges because the returned surface is
    intentionally not stored in every canonical motion artifact.
    """
    poses = Rotation.from_quat(
        arrays['joint_local_quaternion_wxyz'][..., [1, 2, 3, 0]].reshape(-1, 4)
    ).as_rotvec().reshape(len(arrays['time_ns']), 156)
    betas = np.asarray(arrays['betas'], dtype=np.float64)
    shapedirs = np.asarray(model['shapedirs'], dtype=np.float64)
    if use_dmpl:
        dmpls = np.asarray(arrays['dmpls'], dtype=np.float64)
        shapedirs = np.concatenate((shapedirs, np.asarray(dmpl_basis)), axis=2)
    torch, lbs_args = _model_tensors(model, shapedirs=shapedirs)
    from smplx.lbs import lbs

    result = []
    with torch.no_grad():
        for start in range(0, len(poses), chunk_frames):
            stop = min(len(poses), start + chunk_frames); count = stop - start
            shape = np.repeat(betas[None], count, axis=0)
            if use_dmpl:
                shape = np.concatenate((shape, dmpls[start:stop]), axis=1)
            vertices, joints = lbs(
                torch.as_tensor(shape, dtype=torch.float64),
                torch.as_tensor(poses[start:stop], dtype=torch.float64),
                **lbs_args)
            vertices = vertices.detach().cpu().numpy()
            joints = joints.detach().cpu().numpy()
            translation = arrays['root_position_m'][start:stop] - joints[:, 0]
            result.append(vertices + translation[:, None])
    return np.concatenate(result)


def joint_index(name):
    try:
        return SMPLH_JOINT_NAMES.index(name)
    except ValueError as error:
        raise ValueError('Unknown SMPL+H joint: ' + str(name)) from error
