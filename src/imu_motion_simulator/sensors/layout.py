"""Strict named sensor layouts and processing profiles."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..motion.kinematics import joint_index


def load_layout(path):
    value = json.loads(Path(path).read_text())
    required = {'schema', 'layout_id', 'description', 'status', 'mounts'}
    if set(value) != required or value['schema'] != 'imu_motion_simulator.sensor_layout.v1' \
            or not value['layout_id'] or value['status'] not in ('production', 'provisional') \
            or not isinstance(value['mounts'], list) or not value['mounts']:
        raise ValueError('Invalid sensor layout')
    ids = set()
    for mount in value['mounts']:
        if set(mount) != {'id', 'joint', 'position_joint_m',
                         'quaternion_joint_from_sensor_wxyz'} \
                or not mount['id'] or mount['id'] in ids:
            raise ValueError('Invalid or duplicate sensor mount')
        joint_index(mount['joint'])
        position = np.asarray(mount['position_joint_m'], dtype=np.float64)
        quaternion = np.asarray(mount['quaternion_joint_from_sensor_wxyz'],
                                dtype=np.float64)
        if position.shape != (3,) or quaternion.shape != (4,) \
                or not np.isfinite(position).all() or not np.isfinite(quaternion).all() \
                or abs(np.linalg.norm(quaternion) - 1) > 1e-8:
            raise ValueError('Invalid sensor mount transform')
        ids.add(mount['id'])
    return value


def load_profile(path, *, calibrated=False):
    value = json.loads(Path(path).read_text())
    allowed_schemas = ({'imu_motion_simulator.calibration_profile.v1'}
                       if calibrated else {
                           'imu_motion_simulator.ideal_imu_profile.v1',
                           'imu_motion_simulator.ideal_imu_profile.v2',
                           'imu_motion_simulator.ideal_imu_profile.v3'})
    if value.get('schema') not in allowed_schemas or not value.get('profile_id'):
        raise ValueError('Invalid sensor profile')
    if calibrated:
        required = {'schema', 'profile_id', 'description', 'acceleration',
                    'angular_velocity'}
        if set(value) != required:
            raise ValueError('Invalid calibration profile fields')
        for key in ('acceleration', 'angular_velocity'):
            spec = value[key]
            if set(spec) != {'bias', 'scale', 'noise_std'}:
                raise ValueError('Invalid calibration component')
            for field in ('bias', 'scale', 'noise_std'):
                array = np.asarray(spec[field], dtype=np.float64)
                if array.shape != (3,) or not np.isfinite(array).all():
                    raise ValueError('Invalid calibration vector')
            if np.any(np.asarray(spec['scale']) <= 0) \
                    or np.any(np.asarray(spec['noise_std']) < 0):
                raise ValueError('Invalid calibration scale/noise')
    else:
        common = {'profile_id', 'description', 'output_hz', 'work_hz',
                  'lowpass_hz', 'gravity_m_s2'}
        required = ({'schema', *common}
                    if value['schema'] == 'imu_motion_simulator.ideal_imu_profile.v1'
                    else {'schema', *common, 'boundary_guard_s'}
                    if value['schema'] == 'imu_motion_simulator.ideal_imu_profile.v2'
                    else {'schema', *common, 'boundary_guard_s',
                          'work_grid_policy'}
                    if value['schema'] == 'imu_motion_simulator.ideal_imu_profile.v3'
                    else set())
        if not required or set(value) != required:
            raise ValueError('Invalid ideal profile fields')
        if (type(value['output_hz']) is not int or type(value['work_hz']) is not int
                or value['output_hz'] <= 0 or value['work_hz'] < 4 * value['output_hz']
                or not 0 < value['lowpass_hz'] < value['output_hz'] / 2
                or not 9 <= value['gravity_m_s2'] <= 10):
            raise ValueError('Invalid ideal profile rates')
        if value['schema'].endswith(('.v2', '.v3')):
            guard = value['boundary_guard_s']
            samples = guard * value['output_hz']
            if type(guard) not in (int, float) or not 0 <= guard <= 1 \
                    or abs(samples - round(samples)) > 1e-9:
                raise ValueError('Invalid ideal profile boundary guard')
        if value['schema'].endswith('.v3') \
                and value['work_grid_policy'] != 'source-rate-integer-multiple':
            raise ValueError('Invalid ideal profile work-grid policy')
    return value
