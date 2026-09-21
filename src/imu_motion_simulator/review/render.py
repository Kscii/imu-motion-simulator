"""On-demand exact SMPL+H, DMPL and pose-corrective MP4 rendering."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from ..contracts.common import sha256_file
from ..contracts.internal import read_internal
from ..motion.kinematics import (joint_index, joint_transforms, load_dmpl_basis,
                                 load_model_member, surface_sequence)
from ..motion.selection import selection_slice
from ..sensors.layout import load_layout


def _dynamic_shape(metadata):
    return metadata['resolved_config'].get('dynamic_shape', {
        'source_available': True, 'effective_policy': 'source',
        'components': 8})


def prepare_render_cache(motion, sensors, model_archive, dmpl_archive, layout_path,
                         output, *, selection=None, fps=30):
    motion, sensors, output = map(Path, (motion, sensors, output))
    description, metadata, arrays = read_internal(motion, 'motion')
    sensor_description, sensor_metadata, _ = read_internal(sensors, 'sensors')
    if description.get('motion_contract_version') != 2 \
            or sensor_description.get('sensor_contract_version') != 2 \
            or sensor_metadata['kind_metadata']['motion_id'] \
            != metadata['kind_metadata']['motion_id']:
        raise ValueError('Render inputs are not one canonical motion/sensor pair')
    selected = slice(None); selection_value = None
    if selection is not None:
        selected, selection_value = selection_slice(selection, motion)
        if sensor_metadata['kind_metadata']['selection_id'] \
                != selection_value['selection_id']:
            raise ValueError('Render selection differs from sensor selection')
    selected_arrays = {
        key: (value[selected] if value.ndim and len(value) == description['frames']
              else value) for key, value in arrays.items()}
    source_period = metadata['clocks']['motion']
    source_fps = source_period['denominator'] / source_period['numerator']
    if type(fps) is not int or fps <= 0 or fps > source_fps:
        raise ValueError('Invalid render frame rate')
    indices = np.unique(np.rint(np.arange(
        0, (len(selected_arrays['time_ns']) - 1) / source_fps, 1 / fps)
        * source_fps).astype(np.int64))
    render_arrays = {
        key: (value[indices] if value.ndim and len(value) == len(selected_arrays['time_ns'])
              else value) for key, value in selected_arrays.items()}
    gender = metadata['kind_metadata']['source_gender']
    model = load_model_member(model_archive, gender)
    dynamic = _dynamic_shape(metadata)
    if dynamic['source_available']:
        if dmpl_archive is None:
            raise ValueError('This motion requires a DMPL archive for exact rendering')
        dmpl = load_dmpl_basis(dmpl_archive, gender)
    else:
        dmpl = None
    vertices = surface_sequence(
        model, dmpl, render_arrays,
        use_dmpl=dynamic['source_available'])
    joint_position, joint_rotation = joint_transforms(
        model, render_arrays['betas'], render_arrays['root_position_m'],
        render_arrays['joint_local_quaternion_wxyz'])
    layout = load_layout(layout_path)
    if layout['layout_id'] != sensor_metadata['kind_metadata']['layout_id']:
        raise ValueError('Render layout differs from sensor layout')
    sensor_position, sensor_quaternion = [], []
    for mount in layout['mounts']:
        joint = joint_index(mount['joint'])
        offset = np.asarray(mount['position_joint_m'], dtype=np.float64)
        local_rotation = Rotation.from_quat(np.asarray(
            mount['quaternion_joint_from_sensor_wxyz'])[[1, 2, 3, 0]]).as_matrix()
        rotation = joint_rotation[:, joint] @ local_rotation
        sensor_position.append(joint_position[:, joint] + np.einsum(
            'tij,j->ti', joint_rotation[:, joint], offset))
        sensor_quaternion.append(Rotation.from_matrix(rotation).as_quat()[..., [3, 0, 1, 2]])
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(
        output, vertices=vertices.astype(np.float32),
        faces=np.asarray(model['f'], dtype=np.int32),
        sensor_position_m=np.stack(sensor_position, axis=1).astype(np.float32),
        sensor_quaternion_wxyz=np.stack(sensor_quaternion, axis=1).astype(np.float32),
        frame_time_ns=render_arrays['time_ns'].astype(np.int64))
    return {
        'motion_sha256': sha256_file(motion),
        'sensors_sha256': sha256_file(sensors),
        'model_archive_sha256': sha256_file(model_archive),
        'dmpl_archive_sha256': (sha256_file(dmpl_archive)
                                if dynamic['source_available'] else None),
        'layout_sha256': sha256_file(layout_path),
        'selection_sha256': None if selection is None else sha256_file(selection),
        'cache_sha256': sha256_file(output), 'frames': len(vertices), 'fps': fps,
        'dmpl_applied': dynamic['source_available'],
        'dynamic_shape': dynamic,
        'surface': ('SMPL+H shape, pose correctives and DMPL'
                    if dynamic['source_available'] else
                    'SMPL+H shape and pose correctives; source DMPL unavailable'),
        'sensor_pose_source': 'same canonical motion and named layout as sensors artifact'}


def render_mp4(motion, sensors, model_archive, dmpl_archive, layout_path,
               blender, output, *, selection=None, fps=30, width=1280,
               height=720):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    script = Path(__file__).resolve().parents[3] / 'tools/render_smplh_motion.py'
    with tempfile.TemporaryDirectory(prefix='imu-render-') as directory:
        cache = Path(directory) / 'render.npz'
        report = prepare_render_cache(
            motion, sensors, model_archive, dmpl_archive, layout_path, cache,
            selection=selection, fps=fps)
        subprocess.run([
            str(Path(blender).resolve()), '--background', '--python', str(script),
            '--', str(cache), str(output), str(fps), str(width), str(height)],
            check=True)
    if not output.is_file() or output.stat().st_size < 12:
        raise RuntimeError('Blender did not create an MP4')
    report.update(output=str(output), output_sha256=sha256_file(output),
                  output_bytes=output.stat().st_size)
    metadata_path = output.with_suffix(output.suffix + '.json')
    metadata_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
