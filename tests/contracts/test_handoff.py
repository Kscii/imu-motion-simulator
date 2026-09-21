import json

from imu_motion_simulator.contracts.handoff import (
    build_handoff_fixture, load_review_history,
    validate_snapshot_manifest)


def test_handoff_fixture_filters_reject_and_keeps_replay_only_hdf33(tmp_path):
    output = tmp_path / 'handoff'
    result = build_handoff_fixture(output)
    assert result['valid'] is True
    assert result['included'] == result['excluded'] == 1
    snapshot_path = output / 'snapshot/snapshot.json'
    snapshot = json.loads(snapshot_path.read_text())
    validated = validate_snapshot_manifest(
        snapshot, root=snapshot_path.parent)
    assert validated == {
        'valid': True, 'snapshot_id': 'handoff-fixture-snapshot-v1',
        'candidates': 1, 'shards': 1}
    assert snapshot['included_candidates'] == ['fixture-pass']
    assert snapshot['excluded_candidates'] == [{
        'candidate_id': 'fixture-reject', 'reason': 'human-reject'}]
    latest, paths = load_review_history(output / 'reviews/fixture-pass')
    assert latest['decision'] == 'pass' and len(paths) == 1
