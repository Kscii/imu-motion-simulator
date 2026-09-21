from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

import numpy as np
import pytest

from imu_motion_simulator.labels.babel import (candidate_from_index,
                                                candidates_for_member,
                                                load_index)
from imu_motion_simulator.pipeline.amass_plan import (build_amass_catalog,
                                                      build_amass_plan)
from imu_motion_simulator.pipeline.amass_batch import (audit_amass_batch,
                                                       run_amass_catalog)
from imu_motion_simulator.pipeline.kinematic import _human_review_queue
from imu_motion_simulator.pipeline.plan import load_plan
from imu_motion_simulator.motion.smplh import STAGEII_ADAPTER


def _motion(frames, gender='male', fps=60., *, extra=False):
    stream = io.BytesIO()
    values = dict(poses=np.zeros((frames, 156)), trans=np.zeros((frames, 3)),
                  betas=np.zeros(16), dmpls=np.zeros((frames, 8)),
                  gender=np.asarray(gender), mocap_framerate=np.asarray(fps))
    if extra:
        values.update(marker_data=np.zeros((frames, 3, 3)),
                      marker_labels=np.asarray(['a', 'b', 'c']))
    np.savez(stream, **values)
    return stream.getvalue()


def _stageii_motion(frames, *, betas=16, action='walk'):
    stream = io.BytesIO()
    root = np.zeros((frames, 3)); body = np.zeros((frames, 63))
    hand = np.zeros((frames, 90))
    np.savez(
        stream, poses=np.concatenate((root, body, hand), axis=1),
        trans=np.zeros((frames, 3)), betas=np.zeros(betas),
        gender=np.asarray('male'), mocap_frame_rate=np.asarray(120.),
        surface_model_type=np.asarray('smplh'), num_betas=np.asarray(betas),
        root_orient=root, pose_body=body, pose_hand=hand,
        marker_labels=np.asarray([action]))
    return stream.getvalue()


def _archives(root):
    amass = root / 'datasets/amass/smplh-g/example.tar.bz2'
    amass.parent.mkdir(parents=True)
    with tarfile.open(amass, 'w:bz2') as archive:
        for name, content in [('Example/S1/walk_poses.npz',
                               _motion(121, gender=b'male')),
                              ('Example/S2/turn_poses.npz',
                               _motion(61, 'female', 30., extra=True))]:
            info = tarfile.TarInfo(name); info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    babel = root / 'datasets/babel/babel.zip'; babel.parent.mkdir(parents=True)
    primary = {'7': {'babel_sid': 7, 'dur': 2.,
                     'feat_p': 'Example/S1/walk_poses.npz',
                     'seq_ann': {'labels': [{'raw_label': 'walk',
                                             'proc_label': 'walk',
                                             'act_cat': ['walk']}]}}}
    extra = {'7': {'babel_sid': 7, 'dur': 2.,
                   'feat_p': 'Example/S1/walk_poses.npz',
                   'seq_anns': [{'labels': [{'raw_label': 'move forward',
                                             'proc_label': 'walk',
                                             'act_cat': ['walk']}]}]}}
    with zipfile.ZipFile(babel, 'w') as archive:
        archive.writestr('babel_v1.0_release/train.json', json.dumps(primary))
        archive.writestr('babel_v1.0_release/extra_train.json', json.dumps(extra))
    return amass, babel


def test_babel_primary_and_extra_annotations_merge(tmp_path):
    _, babel = _archives(tmp_path)
    candidate = candidates_for_member(babel, 'Example/S1/walk_poses.npz')
    assert candidate['babel_sid'] == 7
    assert candidate['splits'] == ['extra_train', 'train']
    assert len(candidate['labels']) == 2
    assert load_index(babel)['Example/S1/walk_poses.npz'] == candidate
    doubled = {'Example/Example/S1/walk_poses.npz': candidate}
    assert candidate_from_index(doubled, 'Example/S1/walk_poses.npz') == candidate
    prefixed = {'BMLrub/BioMotionLab_NTroje/rub001/walk_poses.npz': candidate}
    assert candidate_from_index(
        prefixed, 'BioMotionLab_NTroje/rub001/walk_poses.npz',
        source_dataset='BMLrub') == candidate


def test_amass_inventory_freezes_a_valid_full_corpus_plan(tmp_path):
    _, babel = _archives(tmp_path)
    output = tmp_path / 'plan.json'
    report = build_amass_plan(
        tmp_path, output, study_id='example-full-v1', source_dataset='Example',
        amass_archive='datasets/amass/smplh-g/example.tar.bz2',
        smplh_archive='models/smplh/original/smplh.tar.xz',
        dmpl_archive='models/dmpl/original/dmpls.tar.xz',
        babel_archive='datasets/babel/babel.zip',
        layout='configs/sensors/layouts/chest-1.json',
        profile='configs/sensors/profiles/ideal-25hz-v1.json')
    plan = load_plan(output)
    assert report['clips'] == 2 and report['babel_matched_clips'] == 1
    assert abs(report['source_duration_hours'] - 4 / 3600) < 1e-12
    assert [clip['expected_frames'] for clip in plan['clips']] == [121, 61]
    assert plan['clips'][0]['label_candidates'][0]['code'] == 'walk'


def test_amass_catalog_freezes_every_source_and_aggregate(tmp_path):
    amass, babel = _archives(tmp_path)
    second = amass.with_name('Second.tar.bz2')
    with tarfile.open(second, 'w:bz2') as archive:
        for name, payload in [
                ('Second/S1/rest_poses.npz', _motion(31)),
                ('Second/S1/calibration_poses.npz', _motion(1))]:
            info = tarfile.TarInfo(name)
            info.size = len(payload); archive.addfile(info, io.BytesIO(payload))
    stageii = amass.with_name('StageII.tar.bz2')
    with tarfile.open(stageii, 'w:bz2') as archive:
        stream = io.BytesIO()
        np.savez(stream, poses=np.zeros((31, 156)), trans=np.zeros((31, 3)),
                 betas=np.zeros(300), gender=np.asarray('male'),
                 mocap_frame_rate=np.asarray(30.))
        payload = stream.getvalue()
        info = tarfile.TarInfo('StageII/S1/rest_stageii.npz')
        info.size = len(payload); archive.addfile(info, io.BytesIO(payload))
    output = tmp_path / 'catalog'
    report = build_amass_catalog(
        tmp_path, output, amass_directory='datasets/amass/smplh-g',
        smplh_archive='models/smplh/original/smplh.tar.xz',
        dmpl_archive='models/dmpl/original/dmpls.tar.xz',
        babel_archive=babel.relative_to(tmp_path),
        layout='configs/sensors/layouts/chest-1.json',
        profile='configs/sensors/profiles/ideal-25hz-v2.json')
    catalog = json.loads((output / 'catalog.json').read_text())
    assert report['source_archives'] == 3
    assert report['plan_ready_archives'] == 2
    assert report['adapter_required_archives'] == 1
    assert report['excluded_members'] == 1
    assert report['clips'] == 3
    assert catalog['clips'] == 3 and catalog['babel_matched_clips'] == 1
    assert [item['source_dataset'] for item in catalog['sources']] == [
        'Second', 'StageII', 'example']
    for item in catalog['sources']:
        if item['status'] == 'plan-ready':
            assert load_plan(output / item['plan'])['clips']
        else:
            assert item['reason'].startswith(
                'Stage-II AMASS member requires an explicit adapter:')
    second_item = next(item for item in catalog['sources']
                       if item['source_dataset'] == 'Second')
    assert second_item['excluded_members'] == [{
        'source_member': 'Second/S1/calibration_poses.npz',
        'reason': 'fewer-than-two-frames'}]


def test_amass_catalog_rejects_changed_resumable_inputs(tmp_path):
    _archives(tmp_path)
    partial = tmp_path / 'catalog.partial'; partial.mkdir()
    (partial / 'catalog-state.json').write_text(json.dumps({
        'schema': 'imu_motion_simulator.amass_catalog_state.v1',
        'config': {}, 'completed': {}}))
    import pytest
    with pytest.raises(ValueError, match='different inputs'):
        build_amass_catalog(
            tmp_path, tmp_path / 'catalog',
            amass_directory='datasets/amass/smplh-g',
            smplh_archive='models/smplh/original/smplh.tar.xz',
            dmpl_archive='models/dmpl/original/dmpls.tar.xz',
            layout='configs/sensors/layouts/chest-1.json',
            profile='configs/sensors/profiles/ideal-25hz-v2.json')


def test_stageii_adapter_freezes_soma_and_grab_without_dmpl(tmp_path):
    directory = tmp_path / 'datasets/amass/smplh-g'; directory.mkdir(parents=True)
    members = {
        'SOMA': ('SOMA/S1/walk_take01_stageii.npz', _stageii_motion(121)),
        'GRAB': ('GRAB/S1/mug_lift_01_stageii.npz', _stageii_motion(61)),
    }
    for source, (member, payload) in members.items():
        with tarfile.open(directory / f'{source}.tar.bz2', 'w:bz2') as archive:
            info = tarfile.TarInfo(member); info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    output = tmp_path / 'stageii-catalog'
    report = build_amass_catalog(
        tmp_path, output, amass_directory='datasets/amass/smplh-g',
        smplh_archive='models/smplh/original/smplh.tar.xz',
        dmpl_archive='models/dmpl/original/dmpls.tar.xz',
        layout='configs/sensors/layouts/chest-1.json',
        profile='configs/sensors/profiles/ideal-25hz-v3.json',
        include_sources=['SOMA', 'GRAB'],
        source_adapters={'SOMA': STAGEII_ADAPTER,
                         'GRAB': STAGEII_ADAPTER})
    assert report['plan_ready_archives'] == 2 and report['clips'] == 2
    catalog = json.loads((output / 'catalog.json').read_text())
    assert catalog['schema'] == 'imu_motion_simulator.amass_catalog.v2'
    plans = {item['source_dataset']: load_plan(output / item['plan'])
             for item in catalog['sources']}
    assert all(plan['schema'] == 'imu_motion_simulator.kinematic_plan.v2'
               for plan in plans.values())
    assert all(plan['source_adapter']['dynamic_shape'] == 'disabled-zero'
               for plan in plans.values())
    assert plans['SOMA']['clips'][0]['label_candidates'][0]['code'] == 'walk'
    assert plans['GRAB']['clips'][0]['label_candidates'][0]['code'] == 'lift'


def test_stageii_adapter_rejects_non_16_beta_surface(tmp_path):
    archive_path = tmp_path / 'invalid.tar.bz2'
    payload = _stageii_motion(31, betas=300)
    with tarfile.open(archive_path, 'w:bz2') as archive:
        info = tarfile.TarInfo('MOYO/S1/walk_stageii.npz')
        info.size = len(payload); archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match='16-beta'):
        build_amass_plan(
            tmp_path, tmp_path / 'invalid-plan.json', study_id='invalid',
            source_dataset='MOYO', amass_archive='invalid.tar.bz2',
            smplh_archive='models/smplh/original/smplh.tar.xz',
            dmpl_archive='models/dmpl/original/dmpls.tar.xz',
            layout='configs/sensors/layouts/chest-1.json',
            profile='configs/sensors/profiles/ideal-25hz-v3.json',
            adapter_id=STAGEII_ADAPTER)


def test_amass_batch_skips_prior_evidence_and_resumes_completed_sources(
        tmp_path, monkeypatch):
    amass, babel = _archives(tmp_path)
    second = amass.with_name('Second.tar.bz2')
    with tarfile.open(second, 'w:bz2') as archive:
        payload = _motion(31)
        info = tarfile.TarInfo('Second/S1/rest_poses.npz')
        info.size = len(payload); archive.addfile(info, io.BytesIO(payload))
    catalog = tmp_path / 'catalog'
    build_amass_catalog(
        tmp_path, catalog, amass_directory='datasets/amass/smplh-g',
        smplh_archive='models/smplh/original/smplh.tar.xz',
        dmpl_archive='models/dmpl/original/dmpls.tar.xz',
        babel_archive=babel.relative_to(tmp_path),
        layout='configs/sensors/layouts/chest-1.json',
        profile='configs/sensors/profiles/ideal-25hz-v3.json')
    prior = tmp_path / 'prior-second'; (prior / 'run').mkdir(parents=True)
    shutil.copyfile(catalog / 'plans/Second.plan.json', prior / 'plan.json')
    prior_plan = json.loads((prior / 'plan.json').read_text())
    prior_clips = {
        clip['id']: {'automatic_qa': {
            'passed': True, 'warnings': [],
            'checks': {'structure': {'pass': True}}}}
        for clip in prior_plan['clips']}
    (prior / 'run/study-report.json').write_text(json.dumps({
        'study_id': 'prior-second', 'mode': 'kinematic', 'stage': 'review',
        'clips_planned': 1, 'clips_ready': 1,
        'automatic_qa_complete': True, 'automatic_qa_passed': True,
        'human_review': 'unreviewed', 'reviewed_clips': 0,
        'accepted_clips': 0, 'human_review_queue': [],
        'clips': prior_clips}))
    calls = []

    def fake_run(plan, library_root, workspace, checkout, output, *, through):
        calls.append(Path(plan).parent.name)
        plan_value = json.loads(Path(plan).read_text())
        clips = {
            clip['id']: {'automatic_qa': {
                'passed': True, 'warnings': [],
                'checks': {'structure': {'pass': True}}}}
            for clip in plan_value['clips']}
        report = {
            'study_id': calls[-1], 'mode': 'kinematic', 'stage': through,
            'clips_planned': 2, 'clips_ready': 2,
            'automatic_qa_complete': True, 'automatic_qa_passed': True,
            'human_review': 'unreviewed', 'reviewed_clips': 0,
            'accepted_clips': 0, 'human_review_queue': [{}],
            'clips': clips}
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / 'study-report.json').write_text(json.dumps(report))
        return report

    monkeypatch.setattr('imu_motion_simulator.pipeline.run.run_pipeline',
                        fake_run)
    output = tmp_path / 'batch'
    first = run_amass_catalog(
        catalog, output, library_root=tmp_path, checkout=tmp_path,
        workspace=tmp_path, reuse_sources={'Second': prior})
    second_report = run_amass_catalog(
        catalog, output, library_root=tmp_path, checkout=tmp_path,
        workspace=tmp_path, reuse_sources={'Second': prior})
    assert calls == ['example']
    assert first == second_report
    assert first['counts'] == {'reused-existing-study': 1, 'complete': 1}
    state = json.loads((output / 'batch-state.json').read_text())
    assert state['sources']['example']['report']['human_review_queue_size'] == 1
    assert state['sources']['Second']['evidence_profile'].endswith(
        'ideal-25hz-v3.json')
    audit = audit_amass_batch(catalog, output)
    assert audit['production_complete'] is True
    assert audit['totals']['clips'] == 3
    assert audit['totals']['clips_passed'] == 3

    (output / 'sources/example/run/study-report.json').unlink()
    recovered = run_amass_catalog(
        catalog, output, library_root=tmp_path, checkout=tmp_path,
        workspace=tmp_path, reuse_sources={'Second': prior})
    assert calls == ['example', 'example']
    assert recovered['counts'] == {
        'reused-existing-study': 1, 'complete': 1}
    assert audit_amass_batch(catalog, output)['production_complete'] is True


def test_amass_batch_rejects_invalid_worker_count(tmp_path):
    import pytest
    with pytest.raises(ValueError, match='workers'):
        run_amass_catalog(
            tmp_path / 'missing', tmp_path / 'output', library_root=tmp_path,
            checkout=tmp_path, workspace=tmp_path, workers=0)


def test_human_queue_prioritizes_uncertain_labels_and_stratifies_normal_clips():
    def clip(identifier, labels):
        return {'id': identifier, 'source_dataset': 'Example',
                'source_member': f'Example/S1/{identifier}_poses.npz',
                'label_candidates': labels}
    plan = {'clips': [clip('unlabeled', []),
                      clip('walk-low', [{'code': 'walk'}]),
                      clip('walk-high', [{'code': 'walk'}])]}
    state = {'clips': {
        'unlabeled': {'automatic_qa': {'passed': True, 'warnings': [],
                                       'risk_score': .1}},
        'walk-low': {'automatic_qa': {'passed': True, 'warnings': [],
                                      'risk_score': .2}},
        'walk-high': {'automatic_qa': {'passed': True, 'warnings': [],
                                       'risk_score': .7}}}}
    queue = _human_review_queue(plan, state)
    assert [row['clip_id'] for row in queue] == ['unlabeled', 'walk-high']
    assert queue[0]['reasons'] == ['label-unresolved']
    assert queue[1]['reasons'] == ['stratified-sample']
