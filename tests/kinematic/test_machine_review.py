import json
from pathlib import Path

from imu_motion_simulator.contracts.common import sha256_file
from imu_motion_simulator.pipeline.machine_review import (
    build_machine_corpus, evaluate_machine_qa, load_machine_policy,
    merge_candidate_corpora)


ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / 'configs/quality/machine-review-v1.json'


def _qa(passed=True, *, warnings=()):
    return {
        'warnings': list(warnings), 'risk_score': .5,
        'checks': {
            'structure': {'pass': passed},
            'trajectory': {'pass': True},
            'sensor': {'pass': True},
            # The machine policy recomputes this check from raw values.
            'work_grid_convergence': {
                'pass': False, 'boundary_guard_complete': True,
                'acceleration_rms_m_s2': .01,
                'angular_velocity_rms_rad_s': .001}}}


def _batch(tmp_path, passed):
    catalog = tmp_path / 'catalog'; (catalog / 'plans').mkdir(parents=True)
    clips = []
    for index in range(len(passed)):
        clips.append({
            'id': f'clip-{index}', 'source_dataset': 'Source',
            'source_member': f'Source/S1/clip-{index}_poses.npz',
            'expected_gender': 'male', 'expected_frames': 61,
            'frame_range': [0, 61],
            'label_candidates': [{'code': 'walk', 'name': 'Walk'}]})
    plan = {
        'schema': 'imu_motion_simulator.kinematic_plan.v1',
        'study_id': 'fixture-source', 'description': 'fixture',
        'inputs': {'amass_archive': 'amass.tar.bz2',
                   'smplh_archive': 'smplh.tar.xz',
                   'dmpl_archive': 'dmpls.tar.xz'},
        'sensor': {'layout': 'layout.json', 'profile': 'profile.json'},
        'review': {'primary': 'threejs', 'mp4': 'on-demand',
                   'policy': 'automatic-all-risk-stratified-human'},
        'clips': clips}
    plan_path = catalog / 'plans/Source.plan.json'
    plan_path.write_text(json.dumps(plan))
    catalog_value = {
        'schema': 'imu_motion_simulator.amass_catalog.v1',
        'clips': len(clips), 'sources': [{
            'source_dataset': 'Source', 'status': 'plan-ready',
            'plan': 'plans/Source.plan.json',
            'plan_sha256': sha256_file(plan_path), 'clips': len(clips)}]}
    catalog_path = catalog / 'catalog.json'
    catalog_path.write_text(json.dumps(catalog_value))
    batch = tmp_path / 'batch'
    run = batch / 'sources/Source/run'; run.mkdir(parents=True)
    records = {}
    for index, (clip, accepted) in enumerate(zip(clips, passed)):
        qa = _qa(accepted, warnings=(
            'joint-speed-outlier',) if index == 0 else ())
        qa['checks']['work_grid_convergence'].pop(
            'boundary_guard_complete')
        record = {
            'automatic_qa': qa,
            'convergence': {
                'work_hz': 240., 'comparison_work_hz': 480.,
                'acceleration_rms_m_s2': .01,
                'acceleration_peak_m_s2': .02,
                'angular_velocity_rms_rad_s': .001,
                'angular_velocity_peak_rad_s': .002,
                'boundary_guard_s': .2, 'boundary_guard_samples': 5,
                'evaluated_samples': 10}}
        for role, suffix in (
                ('motion', '.motion.h5'), ('sensors', '.sensors.h5'),
                ('selection', '.selection.json')):
            path = run / role / (clip['id'] + suffix)
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f'{role}-{index}'.encode())
            record[role] = str(path.relative_to(run))
            record[role + '_sha256'] = sha256_file(path)
        records[clip['id']] = record
    report = {'clips_planned': len(clips), 'clips_ready': len(clips),
              'clips': records}
    (run / 'study-report.json').write_text(json.dumps(report))
    state = {
        'schema': 'imu_motion_simulator.amass_batch_state.v1',
        'config': {'catalog_sha256': sha256_file(catalog_path)},
        'sources': {'Source': {
            'status': 'complete', 'plan_sha256': sha256_file(plan_path)}}}
    (batch / 'batch-state.json').write_text(json.dumps(state))
    return catalog, batch


def test_machine_policy_uses_raw_convergence_and_ignores_warnings():
    policy = load_machine_policy(POLICY)
    result = evaluate_machine_qa(
        _qa(True, warnings=('joint-speed-outlier',)), policy)
    assert result['passed'] is True
    assert result['warnings'] == ['joint-speed-outlier']


def test_machine_corpus_publishes_exact_threshold_without_failed_payloads(
        tmp_path):
    catalog, batch = _batch(tmp_path, [True, True, True, True, False])
    output = tmp_path / 'candidate'
    report = build_machine_corpus(catalog, batch, output, POLICY)
    assert report['statistics']['candidate_rate'] == .8
    assert report['statistics']['publishable'] is True
    corpus = json.loads((output / 'candidate-corpus.json').read_text())
    assert len(corpus['candidates']) == 4
    assert len(corpus['objects']) == 12
    assert corpus['candidates'][0]['warning_flags'] == [
        'joint-speed-outlier']
    assert corpus['excluded'][0]['candidate_id'] == 'clip-4'


def test_machine_corpus_reports_low_rate_without_quarantining(tmp_path):
    catalog, batch = _batch(tmp_path, [True, True, True, False, False])
    output = tmp_path / 'candidate'
    report = build_machine_corpus(catalog, batch, output, POLICY)
    assert report['quarantined_sources'] == []
    assert report['statistics']['candidate_rate'] == .6
    assert report['statistics']['published_candidates'] == 3
    assert report['statistics']['publishable'] is True
    assert len(json.loads((output / 'candidate-corpus.json').read_text())['candidates']) == 3


def test_machine_corpus_only_quarantines_explicitly_paused_source(tmp_path):
    catalog, batch = _batch(tmp_path, [True, True, True, False, False])
    policy = json.loads(POLICY.read_text())
    policy['paused_sources'] = ['Source']
    policy_path = tmp_path / 'paused-policy.json'
    policy_path.write_text(json.dumps(policy))
    output = tmp_path / 'candidate'
    report = build_machine_corpus(catalog, batch, output, policy_path)
    assert report['quarantined_sources'] == ['Source']
    assert report['statistics']['published_candidates'] == 0
    assert report['statistics']['publishable'] is True
    corpus = json.loads((output / 'candidate-corpus.json').read_text())
    assert len(corpus['excluded']) == 5
    assert all('source-quarantined' in item['reasons'] for item in corpus['excluded'])


def test_machine_corpora_merge_without_copying_payloads(tmp_path):
    first_root = tmp_path / 'one'; first_root.mkdir()
    catalog, batch = _batch(first_root, [True, True, True, True, False])
    first = first_root / 'candidate'
    build_machine_corpus(catalog, batch, first, POLICY)
    value = json.loads((first / 'candidate-corpus.json').read_text())
    for item in value['candidates']:
        item['candidate_id'] = 'two-' + item['candidate_id']
        item['source_dataset'] = 'Second'
    for item in value['excluded']:
        item['candidate_id'] = 'two-' + item['candidate_id']
        item['source_dataset'] = 'Second'
    second = tmp_path / 'second'; second.mkdir()
    (second / 'candidate-corpus.json').write_text(json.dumps(value))
    output = tmp_path / 'merged'
    report = merge_candidate_corpora([first, second], output)
    assert report['candidates'] == 8 and report['total_clips'] == 10
    merged = json.loads((output / 'candidate-corpus.json').read_text())
    assert len(merged['objects']) == 12
