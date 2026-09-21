import hashlib

import h5py
import numpy as np
import pytest

from imu_motion_simulator.contracts.common import ContractError, time_ns
from imu_motion_simulator.contracts.internal import (
    new_metadata, validate_internal, validate_motion, write_internal)


SHA = hashlib.sha256(b'test-producer').hexdigest()


def metadata(kind_metadata, clocks, completion='completed', reason=None):
    return new_metadata(
        producer={'name': 'test', 'version': '1', 'code_sha256': SHA},
        kind_metadata=kind_metadata, provenance={'test': True}, clocks=clocks,
        completion=completion, reason=reason,
        resolved_config={'gravity_m_s2': 9.81},
    )


def motion_values(n=4):
    step = np.arange(n, dtype=np.int64); quaternion = np.zeros((n, 52, 4)); quaternion[..., 0] = 1
    arrays = {
        'time_ns': time_ns(step, 1, 120), 'step_index': step,
        'root_position_m': np.zeros((n, 3)),
        'root_quaternion_wxyz': quaternion[:, 0].copy(),
        'joint_local_quaternion_wxyz': quaternion,
        'joint_world_position_m': np.zeros((n, 52, 3)), 'betas': np.zeros(16),
    }
    info = {'motion_id': 'fixture-motion', 'source_member': 'fixture.npz', 'source_gender': 'male',
            'source_fps': 120, 'joint_names': [f'j{i}' for i in range(52)],
            'model_sha256': SHA, 'original_archive_sha256': SHA, 'dmpl_applied': False}
    return metadata(info, {'motion': {'numerator': 1, 'denominator': 120, 'origin': 'source-frame-zero'}}), arrays


def test_motion_contract_writes_and_rejects_time_drift(tmp_path):
    meta, arrays = motion_values(); path = tmp_path / 'motion.h5'
    report = write_internal(path, 'motion', meta, arrays)
    assert report == {'version': '1.0.0', 'kind': 'motion', 'artifact_id': report['artifact_id'],
                      'frames': 4, 'joints': 52, 'execution_ready': True}
    assert validate_internal(path)['kind'] == 'motion'
    with h5py.File(path, 'r+') as handle: handle['data/time_ns'][2] += 1
    with pytest.raises(ContractError, match='time drift'): validate_motion(path)


def test_atomic_writer_refuses_overwrite(tmp_path):
    meta, arrays = motion_values(); path = tmp_path / 'motion.h5'
    write_internal(path, 'motion', meta, arrays)
    with pytest.raises(ContractError, match='already exists'):
        write_internal(path, 'motion', meta, arrays)
