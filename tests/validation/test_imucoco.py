import io
import zipfile

import numpy as np

from imu_motion_simulator.validation.imucoco import benchmark_imucoco


def _npz(**values):
    stream = io.BytesIO(); np.savez(stream, **values)
    return stream.getvalue()


def test_imucoco_identity_pose_matches_calibrated_orientation(tmp_path):
    archive_path = tmp_path / 'imucoco.zip'
    frames = 9
    devices = np.asarray([
        'wrist', 'pocket', 'ear', 'a1', 'a2', 'a3', 'a4', 'a5'])
    pose = np.broadcast_to(np.eye(3), (frames, 24, 3, 3)).copy()
    quaternion = np.zeros((frames, 4)); quaternion[:, 0] = 1
    imu = {'devices': devices, 'n_frames': np.asarray(frames)}
    calibration = {'R_nav2model': np.eye(3)}
    for device in devices:
        imu[device + '_quaternion'] = quaternion
        calibration[device + '_R_bone2sensor'] = np.eye(3)
    with zipfile.ZipFile(archive_path, 'w') as archive:
        archive.writestr(
            'release/participant_info.csv',
            'participant_id,dominant_hand\nP01,right\n')
        stem = 'release/P01/Upper_Walking'
        archive.writestr(stem + '.npz', _npz(
            pose_local=pose, trans=np.zeros((frames, 3))))
        archive.writestr(stem + '_imu.npz', _npz(**imu))
        archive.writestr(
            stem + '_calibration_lab.npz', _npz(**calibration))
        bad_quaternion = quaternion.copy(); bad_quaternion[3] = 0
        bad_imu = dict(imu, wrist_quaternion=bad_quaternion)
        bad_stem = 'release/P01/Lower_Running'
        archive.writestr(bad_stem + '.npz', _npz(
            pose_local=pose, trans=np.zeros((frames, 3))))
        archive.writestr(bad_stem + '_imu.npz', _npz(**bad_imu))
        archive.writestr(bad_stem + '_calibration_lab.npz', _npz(**calibration))
    output = tmp_path / 'report.json'
    report = benchmark_imucoco(
        archive_path, output, takes=[('P01', 'Upper', 'Walking')])
    assert output.is_file() and len(report['takes']) == 1
    assert report['aggregate']['orientation']['mean_geodesic_deg'] == 0
    assert report['aggregate'][
        'angular_velocity_from_orientation_rad_s']['mae_vector_norm'] == 0
    assert report['aggregate']['raw_acceleration']['status'] == 'blocked'
    assert report['grouped']['participant']['P01'][
        'orientation']['mean_geodesic_deg'] == 0
    assert set(report['grouped']['device']) == set(devices)
    assert report['time_offset_diagnostic'][
        'mean_geodesic_deg_by_lag_frames']['0'] == 0
    all_report = benchmark_imucoco(
        archive_path, tmp_path / 'all.json', all_takes=True)
    assert all_report['requested_takes'] == 2
    assert all_report['completed_takes'] == 1
    assert all_report['failed_takes'][0]['error'] == 'IMUCoCo quaternion norm'
