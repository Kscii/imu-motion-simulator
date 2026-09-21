import hashlib
import io
import json
import sqlite3
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from imu_motion_simulator.contracts.common import get_json, sha256_file, time_ns
from imu_motion_simulator.contracts.delivery import validate_delivery
from imu_motion_simulator.contracts.internal import (
    new_metadata,
    read_internal,
    validate_internal,
    write_internal,
)
from imu_motion_simulator.delivery_kinematic import export_kinematic
from imu_motion_simulator.motion.selection import write_selection
from imu_motion_simulator.motion.smplh import STAGEII_ADAPTER
from imu_motion_simulator.pipeline.amass_plan import (
    build_amass_catalog,
    build_amass_plan,
)
from imu_motion_simulator.pipeline.kinematic import _sensor_task, run_kinematic_pipeline
from imu_motion_simulator.pipeline.plan import load_plan
from imu_motion_simulator.production import JobState, load_job_config, run_job
from imu_motion_simulator.publication import LocalPublicationStore
from imu_motion_simulator.review import (
    append_quality_flag,
    automatic_qa,
    build_bundle,
    validate_bundle,
)
from imu_motion_simulator.sensors import (
    convergence_report,
    derive_calibrated,
    derive_ideal,
)
from imu_motion_simulator.sensors.convergence import (
    convergence_recipe_sha256,
    upgrade_legacy_convergence,
)
from imu_motion_simulator.sensors.derive import _resolved_work_hz, sensor_recipe_sha256

ROOT = Path(__file__).resolve().parents[2]
SHA = hashlib.sha256(b'kinematic-fixture').hexdigest()


def test_provisional_core_export_keeps_weak_labels_separate(tmp_path, monkeypatch):
    from imu_motion_simulator import provisional
    from imu_motion_simulator.publication import _json_bytes
    from imu_motion_simulator.sensors.layout import load_layout

    model = tmp_path / 'smplh.tar.xz'; model_archive(model)
    source = tmp_path / 'motion.h5'; motion(source)
    selection = tmp_path / 'selection.json'
    source_labels = [{'origin': 'babel-1.0', 'kind': 'recording-candidate',
                      'code': 'walk', 'categories': ['walk']}]
    write_selection(selection, source, label_candidates=source_labels)
    layout, profile, _ = configs(tmp_path)
    sensors = tmp_path / 'sensors.h5'
    derive_ideal(source, model, layout, profile, sensors, selection=selection)
    commit = {
        'candidate_id': 'clip-one', 'version_id': 'a' * 64,
        'source_dataset': 'fixture',
        'label_candidates': source_labels,
    }
    digest = hashlib.sha256(_json_bytes(commit)).hexdigest()
    store = LocalPublicationStore(tmp_path / 'store')
    monkeypatch.setattr(provisional, '_store', lambda config: store)
    monkeypatch.setattr(provisional, '_published', lambda config, remote: [(commit, digest)])
    monkeypatch.setattr(provisional, '_layout', lambda config: load_layout(layout))
    monkeypatch.setattr(provisional, '_artifact', lambda config, remote, row, role, temp: {
        'motion': source, 'selection': selection, 'sensors': sensors}[role])
    config = {'publication': {'target': 'dev', 'run_id': 'pilot'},
              'output': str(tmp_path / 'job')}
    (tmp_path / 'job').mkdir()
    with sqlite3.connect(tmp_path / 'job/production.sqlite3') as db:
        db.execute('CREATE TABLE job (id INTEGER, status TEXT)')
        db.execute('CREATE TABLE clips (status TEXT)')
        db.execute('INSERT INTO job VALUES (1, ?)', ('running',))
        db.execute('INSERT INTO clips VALUES (?)', ('published',))
    result = provisional.export_provisional(config)
    assert result['candidate_count'] == 1
    assert result['duration_s'] > 0
    assert result['duration_s'] == pytest.approx(
        result['sample_count'] / 25 / result['sequence_count'])
    assert result['weak_count'] == 1 and result['unresolved_count'] == 0
    path = tmp_path / 'job/provisional' / result['export_id'] / result['h5']['filename']
    assert validate_delivery(path)['profile'] == 'imu_dataset_provisional'
    with h5py.File(path) as handle:
        assert set(handle) == {'samples', 'sequences', 'annotations',
                               'candidate_index', 'weak_labels', 'provenance'}
        assert handle['sequences'][0]['activity_code'].decode() == 'unverified'
        assert handle['sequences'][0]['is_fall'] == False
        assert handle['weak_labels/index'][0]['code'].decode() == 'walking'
        assert handle['candidate_index'][0]['candidate_id'].decode() == 'clip-one'
        assert get_json(handle, 'weak_labels/source_candidates')[0][
            'label_candidates'] == commit['label_candidates']
    assert provisional.export_provisional(config) == result
    with sqlite3.connect(tmp_path / 'job/production.sqlite3') as db:
        db.execute("UPDATE job SET status='complete' WHERE id=1")
    final = provisional.export_provisional(config)
    assert final['coverage'] == 'complete'
    assert final['export_id'] != result['export_id']
    assert final['input_commits_sha256'] == result['input_commits_sha256']
    local_rules = provisional.default_rules()
    local_rules['revision'] = 2
    local_rules['rules'].append({
        'rule_id': 'local-fixture-walk-v1', 'origin': 'babel-1.0',
        'source_value': 'catset:walk', 'source_dataset': 'fixture',
        'target_code': 'walking_local', 'is_fall': False})
    relabeled = provisional.export_provisional(config, rules=local_rules)
    assert relabeled['export_id'] != final['export_id']
    assert relabeled['weak_count'] == 1
    relabeled_path = (tmp_path / 'job/provisional' / relabeled['export_id']
                      / relabeled['h5']['filename'])
    assert validate_delivery(relabeled_path)['profile'] == 'imu_dataset_provisional'
    with h5py.File(relabeled_path) as handle:
        assert handle['weak_labels/index'][0]['code'].decode() == 'walking_local'
    from imu_motion_simulator.local_relabel import relabel_local_core

    final_path = (tmp_path / 'job/provisional' / final['export_id']
                  / final['h5']['filename'])
    local = relabel_local_core(final_path, local_rules, tmp_path / 'local-core',
                               expected_candidates=1)
    assert local['cloud_published'] is False
    assert local['weak_count'] == 1 and local['unresolved_count'] == 0
    assert validate_delivery(local['h5']['path'])['profile'] == 'imu_dataset_provisional'
    with h5py.File(final_path) as original, h5py.File(local['h5']['path']) as variant:
        np.testing.assert_array_equal(original['samples'][:], variant['samples'][:])
        assert variant['weak_labels/index'][0]['code'].decode() == 'walking_local'


def test_provisional_prod_union_needs_both_jobs_and_full_cloud_audit(tmp_path, monkeypatch):
    from imu_motion_simulator import provisional
    from imu_motion_simulator.production.config import config_digest
    from imu_motion_simulator.publication import _json_bytes
    from imu_motion_simulator.sensors.layout import load_layout

    model = tmp_path / 'smplh.tar.xz'; model_archive(model)
    motion_path = tmp_path / 'motion.h5'; motion(motion_path)
    selection = tmp_path / 'selection.json'
    labels = [{'origin': 'babel-1.0', 'kind': 'recording-candidate',
               'code': 'walk', 'categories': ['walk']}]
    write_selection(selection, motion_path, label_candidates=labels)
    layout, profile, _ = configs(tmp_path)
    sensors = tmp_path / 'sensors.h5'
    derive_ideal(motion_path, model, layout, profile, sensors, selection=selection)
    publication = {'target': 'prod', 'run_id': None}
    native = {'output': str(tmp_path / 'native'), 'publication': publication,
              'policy_sha256': 'a' * 64}
    stageii = {'output': str(tmp_path / 'stageii'), 'publication': publication,
               'policy_sha256': 'a' * 64}
    for job in (native, stageii):
        root = Path(job['output']); root.mkdir()
        with sqlite3.connect(root / 'production.sqlite3') as db:
            db.execute('CREATE TABLE job (id INTEGER, status TEXT)')
            db.execute('CREATE TABLE clips (status TEXT)')
            db.execute("INSERT INTO job VALUES (1, 'complete')")
            db.execute("INSERT INTO clips VALUES ('published')")
    commits = {
        native['output']: {'candidate_id': 'z-native', 'version_id': 'a' * 64,
                           'source_dataset': 'Native', 'label_candidates': labels},
        stageii['output']: {'candidate_id': 'a-stageii', 'version_id': 'b' * 64,
                            'source_dataset': 'StageII', 'label_candidates': labels},
    }
    store = LocalPublicationStore(tmp_path / 'store')
    monkeypatch.setattr(provisional, '_store', lambda job: store)
    monkeypatch.setattr(provisional, '_published', lambda job, remote: [
        (commits[job['output']], hashlib.sha256(
            _json_bytes(commits[job['output']])).hexdigest())])
    monkeypatch.setattr(provisional, '_layout', lambda job: load_layout(layout))
    monkeypatch.setattr(provisional, '_artifact', lambda job, remote, row, role, temp: {
        'motion': motion_path, 'selection': selection, 'sensors': sensors}[role])

    partial = provisional.export_provisional(native, additional_configs=[stageii])
    assert partial['coverage'] == 'partial'
    audit = {'schema': 'imu_motion_simulator.prod_chain_audit.v1',
             'prefix': 'synthetic-motion/prod/v1', 'phase': 'complete',
             'all_cloud': {'full_reference_audit': True,
                           'cloud_commits': 2, 'cloud_feeds': 2}}
    for name, job in (('native', native), ('stageii', stageii)):
        audit[name] = {'report_path': str(Path(job['output']) / 'production-report.json'),
                       'planned_clips': 1,
                       'summary': {'status': 'complete',
                                   'config_sha256': config_digest(job),
                                   'counts': {'published': 1}}}
    audit_path = tmp_path / 'audit.json'
    audit['all_cloud']['cloud_feeds'] = 1
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='frozen candidate union'):
        provisional.export_provisional(native, additional_configs=[stageii],
                                       audit_report=audit_path)
    audit['all_cloud']['cloud_feeds'] = 2
    audit_path.write_text(json.dumps(audit))
    full = provisional.export_provisional(native, additional_configs=[stageii],
                                          audit_report=audit_path)
    assert full['coverage'] == full['source_job_status'] == 'complete'
    assert full['candidate_count'] == full['sequence_count'] == 2
    assert full['export_id'] != partial['export_id']
    path = tmp_path / 'native/provisional' / full['export_id'] / full['h5']['filename']
    assert validate_delivery(path)['profile'] == 'imu_dataset_provisional'
    with h5py.File(path) as handle:
        assert [item.decode() for item in handle['candidate_index']['candidate_id']] \
            == ['a-stageii', 'z-native']
    assert provisional.export_provisional(native, additional_configs=[stageii],
                                          audit_report=audit_path) == full
    with pytest.raises(ValueError, match='share one publication target'):
        provisional.export_provisional(native, additional_configs=[{
            **stageii, 'publication': {'target': 'dev', 'run_id': 'other'}}])
    with pytest.raises(ValueError, match='different machine QA policies'):
        provisional.export_provisional(native, additional_configs=[{
            **stageii, 'policy_sha256': 'b' * 64}])
    commits[stageii['output']]['candidate_id'] = 'z-native'
    with pytest.raises(ValueError, match='duplicate candidate IDs'):
        provisional.export_provisional(native, additional_configs=[stageii])


def test_provisional_weak_rules_reject_conflict_and_fall():
    from imu_motion_simulator.provisional import (
        default_rules,
        validate_rules,
        weak_label,
    )

    rules = default_rules()
    commit = {'source_dataset': 'ACCAD', 'label_candidates': [
        {'origin': 'babel-1.0', 'kind': 'recording-candidate',
         'code': 'walk', 'categories': ['walk']},
        {'origin': 'babel-1.0', 'kind': 'temporal-candidate',
         'code': 'transition', 'categories': ['transition']} ]}
    assert weak_label(commit, rules) == ('unresolved', '', '')
    rules['rules'][0]['is_fall'] = True
    with pytest.raises(ValueError, match='fall'):
        validate_rules(rules)


def test_provisional_dataset_specific_rule_overrides_global():
    from imu_motion_simulator.provisional import (
        default_rules,
        validate_rules,
        weak_label,
    )

    rules = default_rules()
    rules['rules'].append({'rule_id': 'accad-walk-v1', 'origin': 'babel-1.0',
                           'source_value': 'walk', 'source_dataset': 'ACCAD',
                           'target_code': 'walking_special', 'is_fall': False})
    validate_rules(rules)
    commit = {'source_dataset': 'ACCAD', 'label_candidates': [
        {'origin': 'babel-1.0', 'kind': 'recording-candidate',
         'code': 'walk', 'categories': ['walk']}]}
    assert weak_label(commit, rules) == ('weak', 'walking_special', 'accad-walk-v1')
    assert weak_label({**commit, 'source_dataset': 'EKUT'}, rules) == (
        'weak', 'walking', 'babel-walk-v1')


def test_provisional_freezes_only_published_machine_pass_with_verified_qa(tmp_path):
    from imu_motion_simulator.production.config import config_digest
    from imu_motion_simulator.provisional import _published

    root = tmp_path / 'job'; root.mkdir()
    config = {'output': str(root),
              'publication': {'target': 'dev', 'run_id': 'pilot'}}
    prefix = 'synthetic-motion/dev/pilot/v1'
    store = LocalPublicationStore(tmp_path / 'store')
    qa = tmp_path / 'manifest.json'
    qa.write_text(json.dumps({'qa': {'passed': True}}))
    qa_key = f'{prefix}/previews/clip-one/{"a" * 64}/manifest.json'
    store.put_file(qa, qa_key, sha256_file(qa))
    commit = {'schema': 'imu_motion_simulator.candidate_commit.v1',
              'candidate_id': 'clip-one', 'version_id': 'a' * 64,
              'bundle_files': [{'name': 'manifest.json', 'key': qa_key,
                                'sha256': sha256_file(qa),
                                'byte_length': qa.stat().st_size}]}
    commit_key = f'{prefix}/candidates/clip-one/{"a" * 64}.json'
    store.put_json(commit_key, commit)
    with sqlite3.connect(root / 'production.sqlite3') as db:
        db.execute('CREATE TABLE job (id INTEGER, config_sha256 TEXT)')
        db.execute('CREATE TABLE clips (clip_id TEXT, commit_key TEXT, status TEXT)')
        db.execute('INSERT INTO job VALUES (1, ?)', (config_digest(config),))
        db.execute('INSERT INTO clips VALUES (?, ?, ?)', ('clip-one', commit_key, 'published'))
        db.execute('INSERT INTO clips VALUES (?, ?, ?)', ('incomplete', None, 'failed'))
    assert len(_published(config, store)) == 1
    store._path(qa_key).write_text(json.dumps({'qa': {'passed': False}}))
    with pytest.raises(ValueError, match='machine QA'):
        _published(config, store)


def model_archive(path):
    vertices = np.zeros((6890, 3), dtype=np.float64)
    vertices[:52, 2] = np.arange(52) * .02
    vertices[1, 0], vertices[2, 0] = -.1, .1
    regressor = np.zeros((52, 6890), dtype=np.float64)
    regressor[np.arange(52), np.arange(52)] = 1
    weights = np.zeros((6890, 52), dtype=np.float64); weights[:, 0] = 1
    tree = np.vstack((np.r_[0, np.arange(51)], np.arange(52))).astype(np.int64)
    stream = io.BytesIO()
    np.savez(stream, v_template=vertices,
             shapedirs=np.zeros((6890, 3, 16)), J_regressor=regressor,
             weights=weights, kintree_table=tree,
             posedirs=np.zeros((6890, 3, 459)),
             f=np.asarray([[0, 1, 2]], dtype=np.uint32))
    with tarfile.open(path, 'w:xz') as archive:
        payload = stream.getvalue(); info = tarfile.TarInfo('male/model.npz')
        info.size = len(payload); archive.addfile(info, io.BytesIO(payload))


def dmpl_archive(path):
    stream = io.BytesIO(); np.savez(stream, eigvec=np.zeros((6890, 3, 8)))
    with tarfile.open(path, 'w:xz') as archive:
        payload = stream.getvalue(); info = tarfile.TarInfo('male/model.npz')
        info.size = len(payload); archive.addfile(info, io.BytesIO(payload))


def amass_archive(path):
    with tarfile.open(path, 'w:bz2') as archive:
        for index, frames in enumerate((31, 61), start=1):
            stream = io.BytesIO()
            translation = np.zeros((frames, 3), dtype=np.float64)
            translation[:, 0] = np.linspace(0, index * .1, frames)
            values = {'poses': np.zeros((frames, 156)), 'trans': translation,
                      'betas': np.zeros(16), 'dmpls': np.zeros((frames, 8)),
                      'gender': np.asarray('male'),
                      'mocap_framerate': np.asarray(30.)}
            if index == 2:
                values.update(marker_data=np.zeros((frames, 3, 3)),
                              marker_labels=np.asarray(['a', 'b', 'c']))
            np.savez(stream, **values)
            payload = stream.getvalue()
            info = tarfile.TarInfo(f'Fixture/S1/motion{index}_poses.npz')
            info.size = len(payload); archive.addfile(info, io.BytesIO(payload))


def stageii_archive(path):
    frames = 61
    root = np.zeros((frames, 3)); body = np.zeros((frames, 63))
    hand = np.zeros((frames, 90)); stream = io.BytesIO()
    np.savez(
        stream, poses=np.concatenate((root, body, hand), axis=1),
        trans=np.zeros((frames, 3)), betas=np.zeros(16),
        gender=np.asarray('male'), mocap_frame_rate=np.asarray(120.),
        surface_model_type=np.asarray('smplh'), num_betas=np.asarray(16),
        root_orient=root, pose_body=body, pose_hand=hand,
        markers=np.zeros((frames, 3, 3)))
    with tarfile.open(path, 'w:bz2') as archive:
        payload = stream.getvalue()
        info = tarfile.TarInfo('SOMA/S1/walk_take01_stageii.npz')
        info.size = len(payload); archive.addfile(info, io.BytesIO(payload))


def motion(path, frames=121, *, dynamic_shape=None):
    quaternion = np.zeros((frames, 52, 4)); quaternion[..., 0] = 1
    arrays = {
        'time_ns': time_ns(np.arange(frames), 1, 120),
        'root_position_m': np.zeros((frames, 3)),
        'root_quaternion_wxyz': quaternion[:, 0].copy(),
        'joint_local_quaternion_wxyz': quaternion,
        'betas': np.zeros(16), 'dmpls': np.zeros((frames, 8)),
        'valid': np.ones(frames, dtype=bool)}
    info = {
        'motion_contract_version': 2, 'motion_id': 'fixture-motion',
        'source_dataset': 'fixture', 'source_member': 'fixture/member.npz',
        'source_gender': 'male', 'source_fps_hz': 120.,
        'joint_names': [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
            + [f'{side}_{finger}{joint}' for side in ('left', 'right')
               for finger in ('index', 'middle', 'pinky', 'ring', 'thumb')
               for joint in (1, 2, 3)],
        'model_family': 'smplh', 'model_sha256': SHA,
        'original_archive_sha256': SHA}
    metadata = new_metadata(
        producer={'name': 'fixture', 'version': '1', 'code_sha256': SHA},
        kind_metadata=info, provenance={'fixture': True},
        clocks={'motion': {'numerator': 1, 'denominator': 120,
                           'origin': 'source-frame-zero'}},
        resolved_config={} if dynamic_shape is None else {
            'dynamic_shape': dynamic_shape})
    return write_internal(path, 'motion', metadata, arrays)


def configs(directory):
    layout = directory / 'layout.json'
    layout.write_text(json.dumps({
        'schema': 'imu_motion_simulator.sensor_layout.v1', 'layout_id': 'test-1',
        'description': 'test', 'status': 'production',
        'mounts': [{'id': 'pelvis', 'joint': 'pelvis',
                    'position_joint_m': [0, 0, 0],
                    'quaternion_joint_from_sensor_wxyz': [1, 0, 0, 0]}]}))
    profile = directory / 'profile.json'
    profile.write_text(json.dumps({
        'schema': 'imu_motion_simulator.ideal_imu_profile.v1',
        'profile_id': 'test-25hz', 'description': 'test', 'output_hz': 25,
        'work_hz': 200, 'lowpass_hz': 10., 'gravity_m_s2': 9.81}))
    calibration = directory / 'calibration.json'
    calibration.write_text(json.dumps({
        'schema': 'imu_motion_simulator.calibration_profile.v1',
        'profile_id': 'test-calibration', 'description': 'test',
        'acceleration': {'bias': [0, 0, 0], 'scale': [1, 1, 1],
                         'noise_std': [.01, .01, .01]},
        'angular_velocity': {'bias': [0, 0, 0], 'scale': [1, 1, 1],
                             'noise_std': [.001, .001, .001]}}))
    return layout, profile, calibration


def test_clip_worker_matches_serial_sensor_arrays_and_qa(tmp_path):
    model = tmp_path / 'smplh.tar.xz'; model_archive(model)
    source = tmp_path / 'motion.h5'; motion(source, frames=121)
    selection = tmp_path / 'selection.json'
    write_selection(selection, source, start_frame=0, stop_frame=121,
                    label_candidates=[])
    layout, profile, _ = configs(tmp_path)
    serial = tmp_path / 'serial.h5'
    derive_ideal(source, model, layout, profile, serial,
                 selection=selection)
    expected_convergence = convergence_report(
        source, model, layout, profile, selection=selection)
    expected_qa = automatic_qa(source, serial, selection=selection,
                               convergence=expected_convergence)
    requests = [
        ('clip', source, selection, tmp_path / f'worker-{index}.h5',
         model, layout, profile, sha256_file(model), None, None, None,
         sensor_recipe_sha256(), convergence_recipe_sha256())
        for index in (1, 2)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_sensor_task, requests))
    _, _, expected_arrays = read_internal(serial, 'sensors')
    for (_, result), request in zip(results, requests):
        _, _, actual_arrays = read_internal(request[3], 'sensors')
        for name in ('specific_force_m_s2', 'angular_velocity_rad_s'):
            np.testing.assert_array_equal(expected_arrays[name],
                                          actual_arrays[name])
        assert result['convergence'] == expected_convergence
        assert result['automatic_qa']['checks'] == expected_qa['checks']
        assert result['automatic_qa']['passed'] == expected_qa['passed']


def test_motion_v2_drives_multi_stage_kinematic_outputs(tmp_path):
    model = tmp_path / 'smplh.tar.xz'; model_archive(model)
    source = tmp_path / 'motion.h5'; report = motion(source)
    assert report['motion_contract_version'] == 2
    selection = tmp_path / 'selection.json'
    write_selection(selection, source, start_frame=0, stop_frame=121,
                    label_candidates=[{'code': 'stand', 'name': 'Standing',
                                       'is_fall': False, 'origin': 'test'}])
    layout, profile, calibration = configs(tmp_path)
    sensors = tmp_path / 'sensors.h5'
    sensor_report = derive_ideal(source, model, layout, profile, sensors,
                                 selection=selection)
    assert sensor_report['mounts'] == 1 and sensor_report['samples'] == 26
    _, _, values = read_internal(sensors, 'sensors')
    np.testing.assert_allclose(values['specific_force_m_s2'][..., 2], 9.81,
                               atol=1e-9)
    np.testing.assert_allclose(values['angular_velocity_rad_s'], 0, atol=1e-12)
    convergence = convergence_report(source, model, layout, profile,
                                     selection=selection)
    assert convergence['acceleration_rms_m_s2'] < 1e-10
    assert convergence['boundary_guard_s'] == 0
    assert convergence['boundary_guard_complete'] is True
    legacy_convergence = dict(convergence)
    legacy_convergence.pop('boundary_guard_complete')
    assert upgrade_legacy_convergence(
        legacy_convergence)['boundary_guard_complete'] is True
    assert upgrade_legacy_convergence({'work_hz': 240}) is None
    assert _resolved_work_hz({
        'work_hz': 240,
        'work_grid_policy': 'source-rate-integer-multiple'}, {
            'kind_metadata': {'source_fps_hz': 100.}}) == 300
    missing_convergence = automatic_qa(
        source, sensors, selection=selection)
    assert missing_convergence['passed'] is False
    assert missing_convergence['checks']['work_grid_convergence']['pass'] is False
    review = tmp_path / 'review'
    built = build_bundle(source, sensors, model, layout, review,
                         selection=selection, convergence=convergence)
    assert built['valid'] is True and validate_bundle(review)['frames'] == 121
    assert (review / 'three.module.js').stat().st_size > 100_000

    with h5py.File(sensors, 'r+') as handle:
        handle['data/specific_force_m_s2'][0, 0, 0] = 101.
    warning = automatic_qa(
        source, sensors, selection=selection, convergence=convergence)
    assert warning['passed'] is True
    assert warning['warnings'] == ['specific-force-outlier']

    guarded_profile = tmp_path / 'guarded-profile.json'
    guarded_profile.write_text(json.dumps({
        'schema': 'imu_motion_simulator.ideal_imu_profile.v2',
        'profile_id': 'guarded-test', 'description': 'test',
        'output_hz': 25, 'work_hz': 200, 'lowpass_hz': 10.,
        'gravity_m_s2': 9.81, 'boundary_guard_s': .2}))
    short_motion = tmp_path / 'short-motion.h5'; motion(short_motion, frames=21)
    short_sensors = tmp_path / 'short-sensors.h5'
    derive_ideal(short_motion, model, layout, guarded_profile, short_sensors)
    short_convergence = convergence_report(
        short_motion, model, layout, guarded_profile)
    assert short_convergence['boundary_guard_complete'] is False
    assert short_convergence['evaluated_samples'] > 0
    assert automatic_qa(
        short_motion, short_sensors,
        convergence=short_convergence)['passed'] is False

    calibrated_one = tmp_path / 'calibrated-one.h5'
    calibrated_two = tmp_path / 'calibrated-two.h5'
    derive_calibrated(sensors, calibration, calibrated_one, seed=7)
    derive_calibrated(sensors, calibration, calibrated_two, seed=7)
    one = read_internal(calibrated_one, 'sensors')[2]
    two = read_internal(calibrated_two, 'sensors')[2]
    np.testing.assert_array_equal(one['specific_force_m_s2'],
                                  two['specific_force_m_s2'])


def test_accepted_review_exports_hdf5_33_with_quaternion_replay(tmp_path, monkeypatch):
    model = tmp_path / 'smplh.tar.xz'; model_archive(model)
    dmpl = tmp_path / 'dmpls.tar.xz'; dmpl_archive(dmpl)
    source = tmp_path / 'motion.h5'; motion(source)
    selection = tmp_path / 'selection.json'
    write_selection(selection, source, label_candidates=[
        {'code': 'stand', 'name': 'Standing', 'is_fall': False,
         'origin': 'test'}])
    layout, profile, _ = configs(tmp_path)
    sensors = tmp_path / 'sensors.h5'
    derive_ideal(source, model, layout, profile, sensors, selection=selection)
    review = tmp_path / 'review'; build_bundle(
        source, sensors, model, layout, review, selection=selection,
        convergence=convergence_report(source, model, layout, profile,
                                       selection=selection))
    first = json.loads((review / 'review-r1.json').read_text())
    accepted = dict(first, revision=2, decision='accepted', reviewer='fixture',
                    reason='fixture acceptance', labels=[
                        {'code': 'stand', 'name': 'Standing', 'is_fall': False,
                         'taxonomy_id': 'motion-actions',
                         'taxonomy_version': '1.0.0.motion-r2',
                         'origin': 'auto', 'verification': 'rule',
                         'mapping_rule_id': 'fixture-rule'}])
    (review / 'review-r2.json').write_text(json.dumps(accepted))
    output = tmp_path / 'delivery.h5'
    result = export_kinematic(
        source, sensors, selection, review, layout, output,
        dataset_id='fixture-kinematic', model_archive=model,
        dmpl_archive=dmpl, include_replay=True)
    assert result['version'] == '3.3.0' and result['capabilities']['replay'] is True
    assert validate_delivery(output)['samples'] == 26
    with h5py.File(output, 'r') as handle:
        catalog = handle['labels/catalog'][0]
        assert catalog['taxonomy_id'].decode().rstrip('\x00') == 'motion-actions'
        assert catalog['taxonomy_version'].decode().rstrip('\x00') == '1.0.0.motion-r2'
        assert handle['replay/records/motion-fixture-motion/joint_local_quaternion_wxyz'].shape == (121, 52, 4)
        assert get_json(handle, 'replay/records/motion-fixture-motion/metadata')[
            'replay_contract_version'] == 2

    # A quality pass and an independently frozen label must produce the same
    # taxonomy and replay payload through the queued snapshot worker.
    from imu_motion_simulator.publication import LocalPublicationStore, _json_bytes
    from imu_motion_simulator.snapshot_worker import run_snapshot
    store = LocalPublicationStore(tmp_path / 'published')
    prefix = 'synthetic-motion/dev/pilot/v1'
    candidate_id, version_id = 'clip-one', 'a' * 64
    objects = []
    for role, path in [('motion', source), ('sensors', sensors), ('selection', selection)]:
        digest = sha256_file(path)
        key = f'{prefix}/objects/{digest}/{role}'
        store.put_file(path, key, digest)
        objects.append({'role': role, 'key': key, 'sha256': digest,
                        'byte_length': path.stat().st_size})
    bundle_files = []
    for path in sorted(review.iterdir()):
        if not path.is_file():
            continue
        digest = sha256_file(path)
        key = f'{prefix}/previews/{candidate_id}/{version_id}/{path.name}'
        store.put_file(path, key, digest)
        bundle_files.append({'name': path.name, 'key': key, 'sha256': digest,
                             'byte_length': path.stat().st_size})
    commit = {'candidate_id': candidate_id, 'version_id': version_id,
              'objects': objects, 'bundle_files': bundle_files}
    commit_key = f'{prefix}/candidates/{candidate_id}/{version_id}.json'
    store.put_json(commit_key, commit)
    commit_hash = hashlib.sha256(_json_bytes(commit)).hexdigest()
    quality = {'candidate_id': candidate_id, 'version_id': version_id,
               'candidate_commit_sha256': commit_hash, 'decision': 'pass',
               'reviewer': 'fixture', 'labels': []}
    quality_key = f'{prefix}/reviews/{candidate_id}/{version_id}/revisions/r1.json'
    store.put_json(quality_key, quality)
    frozen_label = {'candidate_id': candidate_id, 'version_id': version_id,
                    'candidate_commit_sha256': commit_hash, 'label': accepted['labels'][0]}
    label_key = f'{prefix}/labels/{candidate_id}/{version_id}/resolutions/frozen.json'
    store.put_json(label_key, frozen_label)
    entries = [{'candidate_id': candidate_id, 'version_id': version_id,
                'commit_key': commit_key, 'commit_sha256': commit_hash,
                'review_key': quality_key,
                'review_sha256': hashlib.sha256(_json_bytes(quality)).hexdigest(),
                'label_key': label_key,
                'label_sha256': hashlib.sha256(_json_bytes(frozen_label)).hexdigest()}]
    second_id, second_version = 'clip-two', 'c' * 64
    second_commit = {**commit, 'candidate_id': second_id,
                     'version_id': second_version}
    second_commit_key = f'{prefix}/candidates/{second_id}/{second_version}.json'
    store.put_json(second_commit_key, second_commit)
    second_commit_hash = hashlib.sha256(_json_bytes(second_commit)).hexdigest()
    second_quality = {**quality, 'candidate_id': second_id,
                      'version_id': second_version,
                      'candidate_commit_sha256': second_commit_hash}
    second_quality_key = (f'{prefix}/reviews/{second_id}/{second_version}'
                          '/revisions/r1.json')
    store.put_json(second_quality_key, second_quality)
    second_label = {**frozen_label, 'candidate_id': second_id,
                    'version_id': second_version,
                    'candidate_commit_sha256': second_commit_hash}
    second_label_key = (f'{prefix}/labels/{second_id}/{second_version}'
                        '/resolutions/frozen.json')
    store.put_json(second_label_key, second_label)
    entries.append({
        'candidate_id': second_id, 'version_id': second_version,
        'commit_key': second_commit_key, 'commit_sha256': second_commit_hash,
        'review_key': second_quality_key,
        'review_sha256': hashlib.sha256(_json_bytes(second_quality)).hexdigest(),
        'label_key': second_label_key,
        'label_sha256': hashlib.sha256(_json_bytes(second_label)).hexdigest(),
    })
    snapshot_id = 'synthetic-' + 'b' * 32
    store.put_json(f'{prefix}/snapshots/requests/{snapshot_id}.json', {
        'schema': 'imu_motion_simulator.snapshot_intent.v1', 'snapshot_id': snapshot_id,
        'entries': entries})
    worker_result = run_snapshot(
        store, 'pilot', snapshot_id, tmp_path / 'snapshot-output',
        model_archive=model, dmpl_archive=dmpl, layout=layout)
    assert worker_result['state'] == 'complete'
    assert worker_result['candidate_count'] == 2
    assert len(worker_result['shards']) == 1
    assert [row['candidate_id'] for row in worker_result['shards'][0]['candidates']] == [
        candidate_id, second_id]
    shard = tmp_path / 'snapshot-output/shard-0001.h5'
    assert validate_delivery(shard)['samples'] == 52
    with h5py.File(shard, 'r') as handle:
        assert len(handle['sequences']) == 2
        assert len(handle['labels/catalog']) == 1
        assert len(get_json(handle, 'assets/index')) == 1
        assert len(handle['assets/blobs']) == 2
        assert len(handle['replay/index']) == 2
        assert len(set(handle['replay/records'])) == 2
        assert [row['sequence_index'] for row in
                get_json(handle, 'provenance/metadata')['sequence_sources']] == [0, 1]
    assert run_snapshot(
        store, 'pilot', snapshot_id, tmp_path / 'snapshot-output',
        model_archive=model, dmpl_archive=dmpl, layout=layout) == worker_result
    from imu_motion_simulator.snapshot_pack import pack_clip_deliveries
    assert not list((tmp_path / 'snapshot-output/clip-deliveries').glob('*.h5'))
    assert (tmp_path / 'snapshot-output/checkpoints/shard-0001.json').is_file()
    clip_paths = []
    for index in (1, 2):
        clip = tmp_path / f'split-clip-{index}.h5'
        export_kinematic(source, sensors, selection, review, layout, clip,
                         dataset_id=snapshot_id, model_archive=model,
                         dmpl_archive=dmpl, include_replay=True)
        clip_paths.append(clip)
    split = pack_clip_deliveries(
        clip_paths, tmp_path / 'split-output', dataset_id=snapshot_id,
        max_shard_bytes=max(path.stat().st_size for path in clip_paths) + 1024)
    assert len(split) == 2
    assert [validate_delivery(path)['samples'] for path, _ in split] == [26, 26]

    from imu_motion_simulator import snapshot_worker
    split_id = 'synthetic-' + 'd' * 32
    store.put_json(f'{prefix}/snapshots/requests/{split_id}.json', {
        'schema': 'imu_motion_simulator.snapshot_intent.v1',
        'snapshot_id': split_id, 'entries': entries})
    monkeypatch.setattr(snapshot_worker, 'MAX_SHARD_BYTES',
                        max(path.stat().st_size for path in clip_paths) * 2)

    class FailSecondUpload:
        def __getattr__(self, name):
            return getattr(store, name)

        def put_file(self, path, key, digest):
            if key.endswith('/shards/shard-0002.h5'):
                raise OSError('injected second shard upload failure')
            return store.put_file(path, key, digest)

    split_output = tmp_path / 'streaming-snapshot'
    with pytest.raises(OSError, match='injected second shard'):
        run_snapshot(FailSecondUpload(), 'pilot', split_id, split_output,
                     model_archive=model, dmpl_archive=dmpl, layout=layout)
    assert (split_output / 'checkpoints/shard-0001.json').is_file()
    assert not (split_output / 'clip-deliveries/clip-0001.h5').exists()
    assert not (split_output / 'clip-deliveries/clip-0002.h5').exists()
    assert (split_output / 'shard-0002.h5').exists()

    original_export = snapshot_worker.export_kinematic

    def do_not_rebuild_first(*args, **kwargs):
        if Path(args[5]).name == 'clip-0001.h5':
            raise AssertionError('Completed shard was rebuilt')
        return original_export(*args, **kwargs)

    monkeypatch.setattr(snapshot_worker, 'export_kinematic', do_not_rebuild_first)
    resumed = run_snapshot(store, 'pilot', split_id, split_output,
                           model_archive=model, dmpl_archive=dmpl, layout=layout)
    assert resumed['candidate_count'] == 2
    assert len(resumed['shards']) == 2
    assert not list((split_output / 'clip-deliveries').glob('*.h5'))
    assert [validate_delivery(split_output / f'shard-{index:04}.h5')['samples']
            for index in (1, 2)] == [26, 26]

    monkeypatch.setattr(snapshot_worker, 'export_kinematic', original_export)
    crash_id = 'synthetic-' + 'e' * 32
    store.put_json(f'{prefix}/snapshots/requests/{crash_id}.json', {
        'schema': 'imu_motion_simulator.snapshot_intent.v1',
        'snapshot_id': crash_id, 'entries': entries})
    original_checkpoint = snapshot_worker._write_checkpoint
    crashed = False

    def fail_after_upload(*args, **kwargs):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise OSError('injected checkpoint failure after shard upload')
        return original_checkpoint(*args, **kwargs)

    monkeypatch.setattr(snapshot_worker, '_write_checkpoint', fail_after_upload)
    crash_output = tmp_path / 'uploaded-without-checkpoint'
    with pytest.raises(OSError, match='injected checkpoint failure'):
        run_snapshot(store, 'pilot', crash_id, crash_output,
                     model_archive=model, dmpl_archive=dmpl, layout=layout)
    assert (crash_output / 'shard-0001.h5').is_file()
    assert not (crash_output / 'checkpoints/shard-0001.json').exists()
    monkeypatch.setattr(snapshot_worker, '_write_checkpoint', original_checkpoint)
    recovered = run_snapshot(store, 'pilot', crash_id, crash_output,
                             model_archive=model, dmpl_archive=dmpl, layout=layout)
    assert recovered['candidate_count'] == 2
    assert len(recovered['shards']) == 2


def test_stageii_review_flag_and_replay_v3_preserve_missing_dmpl(tmp_path):
    model = tmp_path / 'smplh.tar.xz'; model_archive(model)
    dynamic = {'source_available': False,
               'effective_policy': 'disabled-zero', 'components': 8}
    source = tmp_path / 'stageii-motion.h5'; motion(
        source, dynamic_shape=dynamic)
    selection = tmp_path / 'selection.json'
    write_selection(selection, source, label_candidates=[
        {'code': 'walk', 'name': 'Walk', 'is_fall': False,
         'origin': 'stageii-source-member'}])
    layout, profile, _ = configs(tmp_path)
    sensors = tmp_path / 'sensors.h5'; derive_ideal(
        source, model, layout, profile, sensors, selection=selection)
    review = tmp_path / 'review'; build_bundle(
        source, sensors, model, layout, review, selection=selection,
        convergence=convergence_report(
            source, model, layout, profile, selection=selection))
    flagged = append_quality_flag(
        review, reviewer='fixture-observer', reason='visible one-frame jump',
        start_frame=10, stop_frame=16, joints=['spine2'])
    assert flagged['quality_flags'][0]['disposition'] == 'advisory'
    latest = json.loads((review / 'review-r2.json').read_text())
    accepted = dict(latest, revision=3, decision='accepted',
                    reviewer='fixture-reviewer', reason='accepted with source flag')
    (review / 'review-r3.json').write_text(json.dumps(accepted))
    output = tmp_path / 'stageii-delivery.h5'
    export_kinematic(
        source, sensors, selection, review, layout, output,
        dataset_id='fixture-stageii', model_archive=model,
        include_replay=True)
    with h5py.File(output, 'r') as handle:
        record = 'replay/records/motion-fixture-motion'
        metadata = get_json(handle, record + '/metadata')
        assert metadata['replay_contract_version'] == 3
        assert metadata['dynamic_shape'] == dynamic
        assert not np.any(handle[record + '/dmpls'][:])
        assets = get_json(handle, 'assets/index')
        assert [item['logical_path'] for item in assets[0]['files']] == [
            'smplh.tar.xz']
        provenance = get_json(handle, 'provenance/metadata')
        refs = provenance['sequence_sources'][0]['refs']
        assert refs['source_quality_flags'][0]['joints'] == ['spine2']
    assert validate_delivery(output)['capabilities']['replay'] is True


def test_stageii_exact_render_omits_dmpl_archive(tmp_path, monkeypatch):
    model = tmp_path / 'smplh.tar.xz'; model_archive(model)
    source = tmp_path / 'stageii-motion.h5'; motion(source, frames=31,
        dynamic_shape={'source_available': False,
                       'effective_policy': 'disabled-zero', 'components': 8})
    layout, profile, _ = configs(tmp_path)
    sensors = tmp_path / 'sensors.h5'; derive_ideal(
        source, model, layout, profile, sensors)
    from imu_motion_simulator.review import render
    observed = {}

    def surface(model_value, dmpl_basis, arrays, *, use_dmpl, **kwargs):
        observed.update(dmpl_basis=dmpl_basis, use_dmpl=use_dmpl)
        return np.zeros((len(arrays['time_ns']), 6890, 3))

    monkeypatch.setattr(render, 'surface_sequence', surface)
    report = render.prepare_render_cache(
        source, sensors, model, None, layout, tmp_path / 'render.npz', fps=30)
    assert observed == {'dmpl_basis': None, 'use_dmpl': False}
    assert report['dmpl_applied'] is False
    assert report['dmpl_archive_sha256'] is None


def test_disabled_dynamic_shape_rejects_nonzero_dmpl(tmp_path):
    source = tmp_path / 'stageii-motion.h5'; motion(source,
        dynamic_shape={'source_available': False,
                       'effective_policy': 'disabled-zero', 'components': 8})
    with h5py.File(source, 'r+') as handle:
        handle['data/dmpls'][0, 0] = 1.
    with pytest.raises(ValueError, match='disabled.*DMPL'):
        validate_internal(source, 'motion')


def test_unreviewed_bundle_cannot_export(tmp_path):
    model = tmp_path / 'smplh.tar.xz'; model_archive(model)
    source = tmp_path / 'motion.h5'; motion(source)
    selection = tmp_path / 'selection.json'; write_selection(
        selection, source, label_candidates=[{'code': 'stand'}])
    layout, profile, _ = configs(tmp_path)
    sensors = tmp_path / 'sensors.h5'; derive_ideal(
        source, model, layout, profile, sensors, selection=selection)
    review = tmp_path / 'review'; build_bundle(
        source, sensors, model, layout, review, selection=selection,
        convergence=convergence_report(source, model, layout, profile,
                                       selection=selection))
    with pytest.raises(ValueError, match='accepted'):
        export_kinematic(source, sensors, selection, review, layout,
                         tmp_path / 'delivery.h5', dataset_id='blocked')


def test_full_pipeline_streams_archive_and_reuses_loaded_model(tmp_path,
                                                               monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('smplx')
    library = tmp_path / 'library'; checkout = tmp_path / 'checkout'
    amass = library / 'datasets/amass/fixture.tar.bz2'
    smplh = library / 'models/smplh.tar.xz'
    dmpl = library / 'models/dmpls.tar.xz'
    amass.parent.mkdir(parents=True); smplh.parent.mkdir(parents=True)
    amass_archive(amass); model_archive(smplh); dmpl_archive(dmpl)
    checkout.mkdir(); layout, profile, _ = configs(checkout)
    plan_path = tmp_path / 'plan.json'
    build_amass_plan(
        library, plan_path, study_id='fixture-full-v1',
        source_dataset='Fixture',
        amass_archive='datasets/amass/fixture.tar.bz2',
        smplh_archive='models/smplh.tar.xz',
        dmpl_archive='models/dmpls.tar.xz',
        layout=layout.relative_to(checkout), profile=profile.relative_to(checkout))

    import imu_motion_simulator.pipeline.kinematic as pipeline
    original_decode = pipeline.decode_amass_member
    caches = []

    def observed_decode(*args, **kwargs):
        assert isinstance(kwargs.get('source_bytes'), bytes)
        caches.append(kwargs.get('model_cache'))
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(pipeline, 'decode_amass_member', observed_decode)
    sensor_report = run_kinematic_pipeline(
        plan_path, load_plan(plan_path), library, checkout,
        tmp_path / 'output', through='sensors')
    persisted = json.loads(
        (tmp_path / 'output/study-report.json').read_text())
    assert persisted == sensor_report
    assert persisted['stage'] == 'sensors'

    report = run_kinematic_pipeline(
        plan_path, load_plan(plan_path), library, checkout,
        tmp_path / 'output', through='review')
    assert len(caches) == 2 and caches[0] is caches[1]
    assert report['clips_ready'] == 2
    assert report['automatic_qa_complete'] is True
    assert report['automatic_qa_passed'] is True


def test_stageii_pipeline_declares_disabled_dynamic_shape(tmp_path):
    pytest.importorskip('torch'); pytest.importorskip('smplx')
    library = tmp_path / 'library'; checkout = tmp_path / 'checkout'
    source_archive = library / 'datasets/amass/SOMA.tar.bz2'
    model = library / 'models/smplh.tar.xz'
    source_archive.parent.mkdir(parents=True); model.parent.mkdir(parents=True)
    stageii_archive(source_archive); model_archive(model)
    checkout.mkdir(); layout, profile, _ = configs(checkout)
    plan_path = tmp_path / 'stageii-plan.json'
    build_amass_plan(
        library, plan_path, study_id='stageii-thin-v1', source_dataset='SOMA',
        amass_archive='datasets/amass/SOMA.tar.bz2',
        smplh_archive='models/smplh.tar.xz',
        dmpl_archive='models/not-required.tar.xz',
        layout=layout.relative_to(checkout), profile=profile.relative_to(checkout),
        adapter_id=STAGEII_ADAPTER)
    output = tmp_path / 'stageii-output'
    report = run_kinematic_pipeline(
        plan_path, load_plan(plan_path), library, checkout, output,
        through='review')
    record = next(iter(report['clips'].values()))
    description, metadata, arrays = read_internal(
        output / record['motion'], 'motion')
    assert description['dynamic_shape_available'] is False
    assert metadata['resolved_config']['dynamic_shape'] == {
        'source_available': False, 'effective_policy': 'disabled-zero',
        'components': 8}
    assert not np.any(arrays['dmpls'])
    manifest = json.loads((output / record['review'] / 'manifest.json').read_text())
    assert manifest['dynamic_shape']['source_available'] is False


def test_production_streams_and_resumes_without_duplicate_publication(tmp_path):
    pytest.importorskip('torch'); pytest.importorskip('smplx')
    library = tmp_path / 'library'; checkout = tmp_path / 'checkout'
    archive = library / 'datasets/amass/Fixture.tar.bz2'
    model = library / 'models/smplh.tar.xz'
    dmpl = library / 'models/dmpls.tar.xz'
    archive.parent.mkdir(parents=True); model.parent.mkdir(parents=True)
    amass_archive(archive); model_archive(model); dmpl_archive(dmpl)
    checkout.mkdir(); layout, profile, _ = configs(checkout)
    catalog = tmp_path / 'catalog'
    build_amass_catalog(
        library, catalog, amass_directory='datasets/amass',
        smplh_archive='models/smplh.tar.xz',
        dmpl_archive='models/dmpls.tar.xz',
        layout=layout.relative_to(checkout),
        profile=profile.relative_to(checkout))
    config_path = tmp_path / 'production.yaml'
    config_path.write_text(yaml.safe_dump({
        'schema': 'imu_motion_simulator.production_job.v1',
        'library_root': str(library), 'catalog': str(catalog),
        'checkout': str(checkout), 'output': str(tmp_path / 'output'),
        'policy': str(ROOT / 'configs/quality/machine-review-v1.json'),
        'sensor_workers': 1, 'sources': ['Fixture'],
        'clips_per_source': None,
        'publication': {'target': 'prod', 'backend': 'local',
                        'bucket': None, 'project': None,
                        'root': str(tmp_path / 'object-store'), 'run_id': None},
    }))
    config = load_job_config(config_path)
    seen = []
    def progress(summary):
        if summary['counts'].get('published') == 1 and not seen:
            commits = LocalPublicationStore(tmp_path / 'object-store').list_keys(
                'synthetic-motion/prod/v1/candidates/')
            assert len(commits) == 1
            seen.append(commits[0])
            JobState(config).request('pause')

    paused = run_job(config, on_progress=progress)
    assert seen and paused['status'] == 'paused'
    assert paused['counts'].get('published', 0) >= 1
    resumed = run_job(config)
    assert resumed['status'] == 'complete'
    assert resumed['counts'] == {'published': 2}
    assert resumed['sources']['Fixture']['status'] == 'complete'
    assert {'archive-hash', 'model-hash', 'archive-stream-next'} <= {
        row['phase'] for row in resumed['source_phase_metrics']}
    assert {'sensor-compute', 'sensor-dispatch-wait',
            'sensor-return-and-collection'} <= {
        row['phase'] for row in resumed['phase_metrics']}
    store = LocalPublicationStore(tmp_path / 'object-store')
    commits = store.list_keys('synthetic-motion/prod/v1/candidates/')
    feeds = store.list_keys('synthetic-motion/prod/v1/index-feed/')
    assert len(commits) == len(feeds) == 2
    with pytest.raises(ValueError, match='cannot be restarted'):
        run_job(config)
    assert store.list_keys('synthetic-motion/prod/v1/index-feed/') == feeds
