import json
from pathlib import Path

import numpy as np
import pytest

from imu_motion_simulator.validation.review_diagnostics import (
    _observations,
    calculate_features,
)


def test_diagnostics_keep_boundary_peaks_and_source_frame_locations():
    root = np.zeros((10, 3))
    root[-1, 0] = .02
    quaternion = np.zeros((10, 2, 4))
    quaternion[..., 0] = 1
    quaternion[-1, 1] = [0, 1, 0, 0]
    force = np.zeros((6, 1, 3))
    force[:, 0, 2] = [60, 10, 10, 10, 10, 120]
    gyro = np.zeros_like(force)
    gyro[:, 0, 0] = [3, 1, 1, 1, 1, 4]
    result = calculate_features(
        root, quaternion, force, gyro, start_frame=30,
        frame_period_s=.01, sensor_rate_hz=25,
        joint_names=['pelvis', 'spine3'])
    assert result['boundary_samples'] == 2
    assert result['specific_force_m_s2']['first']['value'] == 60
    assert result['specific_force_m_s2']['last']['value'] == 120
    assert result['specific_force_m_s2']['interior']['value'] == 10
    assert result['joint_largest_step']['source_frame'] == 38
    assert result['joint_largest_step']['joint'] == 'spine3'
    assert result['loop_seam_root_m'] == pytest.approx(.02)
    assert 'decision' not in result


def test_observation_expansion_has_no_review_decision(tmp_path):
    path = Path(tmp_path) / 'observations.json'
    path.write_text(json.dumps({
        'schema': 'imu_motion_simulator.observation_input.v1',
        'observations': [{
            'candidate_ids': ['clip-a', 'clip-b'],
            'region': 'start', 'feature': 'visual_twitch',
            'note': 'Observed at start, exact frame unknown.'}]}))
    candidates = [
        {'candidate_id': candidate_id, 'source_dataset': 'fixture',
         'objects': {'motion': digest}}
        for candidate_id, digest in [('clip-a', 'a'), ('clip-b', 'b')]]
    result = _observations(path, candidates)
    assert [row['candidate_id'] for row in result] == ['clip-a', 'clip-b']
    assert all(row['source_frame_range'] is None for row in result)
    assert all('decision' not in row and 'reviewer' not in row for row in result)
    document = json.loads(path.read_text())
    document['observations'][0]['candidate_ids'].append('missing')
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='unknown candidate'):
        _observations(path, candidates)
