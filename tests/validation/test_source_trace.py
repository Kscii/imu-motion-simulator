import io
import json
import tarfile

import numpy as np
import pytest

from imu_motion_simulator.validation import source_trace
from imu_motion_simulator.validation.source_trace import _step_degrees


def test_source_step_angle_is_quaternion_sign_invariant():
    trajectory = np.asarray([
        [[0., 0., 0., 1.]],
        [[0., 0., 0., -1.]],
        [[0., 0., 1., 0.]],
    ])
    assert _step_degrees(trajectory)[:, 0] == pytest.approx([0., 180.])


def test_shared_source_member_keeps_each_selection(tmp_path, monkeypatch):
    production = tmp_path / 'production'
    plan = production / 'native/sources/ACCAD/plan.json'
    plan.parent.mkdir(parents=True)
    plan.write_text(json.dumps({'inputs': {'amass_archive': 'archive.tar.bz2'}}))
    library = tmp_path / 'library'
    library.mkdir()
    with tarfile.open(library / 'archive.tar.bz2', 'w:bz2') as archive:
        payload = b'fixture'
        member = tarfile.TarInfo('shared.npz')
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    corpus = tmp_path / 'corpus.json'
    corpus.write_text(json.dumps({
        'candidates': [{'candidate_id': ident, 'source_dataset': 'ACCAD',
                        'source_member': 'shared.npz'} for ident in ('a', 'b')],
        'objects': []}))
    monkeypatch.setattr(source_trace, '_trace',
                        lambda candidate, objects, raw: {
                            'candidate_id': candidate['candidate_id'],
                            'raw': raw.decode()})
    report = source_trace.trace_sources(
        corpus, production, library, tmp_path / 'trace.json', ['a', 'b'])
    assert [row['candidate_id'] for row in report['results']] == ['a', 'b']
    assert [row['raw'] for row in report['results']] == ['fixture', 'fixture']
