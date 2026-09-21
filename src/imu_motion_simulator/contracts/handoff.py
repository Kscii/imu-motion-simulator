"""Strict candidate-review and immutable HDF5 3.3 snapshot handoff."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .common import ContractError, require, sha256_file, sha_string
from .delivery import validate_delivery, write_delivery
from .fixtures import fixture_inputs
from ..pipeline.machine_review import validate_candidate_corpus


MAX_SHARD_BYTES = 4 * 1024 ** 3
REVISION_FIELDS = {
    'schema', 'candidate_id', 'revision', 'decision', 'reviewer',
    'reason', 'labels', 'previous_revision_sha256', 'created_at_utc'}


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_review_revision(value, *, previous=None):
    require(isinstance(value, dict) and set(value) == REVISION_FIELDS,
            'review revision fields')
    require(value['schema'] == 'imu_motion_simulator.human_review_revision.v1',
            'review revision schema')
    require(isinstance(value['candidate_id'], str) and value['candidate_id'],
            'review candidate identity')
    require(type(value['revision']) is int and value['revision'] >= 1,
            'review revision number')
    require(value['decision'] in {'pass', 'reject'}, 'review decision')
    require(isinstance(value['reviewer'], str) and value['reviewer'],
            'reviewer identity')
    require(value['reason'] is None or isinstance(value['reason'], str),
            'review reason')
    require(isinstance(value['labels'], list), 'review labels')
    if value['decision'] == 'pass':
        require(bool(value['labels']), 'passing review requires labels')
    else:
        require(bool(value['reason']), 'rejected review requires a reason')
    if previous is None:
        require(value['revision'] == 1
                and value['previous_revision_sha256'] is None,
                'first review revision chain')
    else:
        require(value['candidate_id'] == previous['candidate_id']
                and value['revision'] == previous['revision'] + 1
                and sha_string(value['previous_revision_sha256']),
                'review revision chain')
    require(isinstance(value['created_at_utc'], str)
            and value['created_at_utc'], 'review timestamp')
    return {'valid': True, 'candidate_id': value['candidate_id'],
            'revision': value['revision'], 'decision': value['decision']}


def load_review_history(directory):
    directory = Path(directory)
    paths = sorted(directory.glob('review-r*.json'),
                   key=lambda path: int(path.stem.split('r')[-1]))
    require(bool(paths), 'review history is empty')
    previous = None
    for path in paths:
        value = json.loads(path.read_text())
        validate_review_revision(value, previous=previous)
        if previous is not None:
            require(value['previous_revision_sha256'] == sha256_file(paths[
                value['revision'] - 2]), 'review previous hash')
        previous = value
    return previous, paths


def validate_snapshot_manifest(value, *, root=None, full=True):
    fields = {'schema', 'snapshot_id', 'created_at_utc', 'prefix',
              'hdf5_version', 'candidate_manifest', 'review_revisions',
              'included_candidates', 'excluded_candidates', 'shards',
              'media_policy'}
    require(isinstance(value, dict) and set(value) == fields,
            'snapshot manifest fields')
    require(value['schema'] == 'imu_motion_simulator.synthetic_snapshot.v1',
            'snapshot schema')
    require(isinstance(value['snapshot_id'], str) and value['snapshot_id'],
            'snapshot identity')
    require(value['prefix'] == 'synthetic-motion/v1'
            and value['hdf5_version'] == '3.3.0', 'snapshot contract')
    require(value['media_policy'] == {
        'mp4': 'excluded', 'replay': 'required',
        'shared_assets': 'deduplicated-within-shard'},
        'snapshot media policy')
    included = value['included_candidates']
    excluded = value['excluded_candidates']
    require(isinstance(included, list) and len(included) == len(set(included)),
            'snapshot included candidates')
    require(isinstance(excluded, list)
            and all(set(item) == {'candidate_id', 'reason'}
                    for item in excluded), 'snapshot exclusions')
    require(not set(included) & {item['candidate_id'] for item in excluded},
            'snapshot inclusion conflict')
    require(isinstance(value['shards'], list) and value['shards'],
            'snapshot shards')
    root = None if root is None else Path(root).resolve()
    for shard in value['shards']:
        require(set(shard) == {
            'path', 'sha256', 'byte_length', 'object_key', 'sequences'},
            'snapshot shard fields')
        require(sha_string(shard['sha256'])
                and type(shard['byte_length']) is int
                and 0 < shard['byte_length'] <= MAX_SHARD_BYTES,
                'snapshot shard identity')
        require(shard['object_key'].startswith(
            f"synthetic-motion/v1/snapshots/{value['snapshot_id']}/shards/"),
            'snapshot shard key')
        if root is not None and full:
            path = (root / shard['path']).resolve()
            require(path.is_relative_to(root) and path.is_file(),
                    'snapshot shard missing')
            require(path.stat().st_size == shard['byte_length']
                    and sha256_file(path) == shard['sha256'],
                    'snapshot shard changed')
            delivery = validate_delivery(path)
            require(delivery['version'] == '3.3.0'
                    and delivery['capabilities']['replay'] is True
                    and delivery['capabilities']['video'] is False,
                    'snapshot shard must be replay-only HDF5 3.3')
    return {'valid': True, 'snapshot_id': value['snapshot_id'],
            'candidates': len(included), 'shards': len(value['shards'])}


def build_snapshot(candidate_manifest, reviews, shards, output, *, snapshot_id):
    candidate_manifest = Path(candidate_manifest).resolve()
    reviews = Path(reviews).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    working = output.with_name(output.name + '.partial')
    if working.exists():
        raise FileExistsError(working)
    corpus = json.loads(candidate_manifest.read_text())
    validate_candidate_corpus(corpus)
    candidate_ids = {item['candidate_id'] for item in corpus['candidates']}
    decisions, revision_rows = {}, []
    for candidate_id in sorted(candidate_ids):
        directory = reviews / candidate_id
        if not directory.is_dir():
            continue
        latest, paths = load_review_history(directory)
        decisions[candidate_id] = latest
        revision_rows.append({
            'candidate_id': candidate_id, 'revision': latest['revision'],
            'decision': latest['decision'],
            'sha256': sha256_file(paths[-1]),
            'path': str(paths[-1].relative_to(reviews))})
    included = sorted(candidate_id for candidate_id, value in decisions.items()
                      if value['decision'] == 'pass')
    excluded = []
    for candidate_id in sorted(candidate_ids):
        decision = decisions.get(candidate_id)
        if decision is None:
            reason = 'unreviewed'
        elif decision['decision'] == 'reject':
            reason = 'human-reject'
        else:
            continue
        excluded.append({'candidate_id': candidate_id, 'reason': reason})
    working.mkdir(parents=True)
    shard_rows = []
    names = set()
    for source in sorted(map(Path, shards), key=lambda path: path.name):
        report = validate_delivery(source)
        require(report['version'] == '3.3.0'
                and report['capabilities']['replay'] is True
                and report['capabilities']['video'] is False,
                'snapshot input shard contract')
        require(source.stat().st_size <= MAX_SHARD_BYTES,
                'snapshot shard exceeds 4 GiB')
        require(source.name not in names, 'duplicate snapshot shard name')
        names.add(source.name)
        target = working / source.name
        shutil.copyfile(source, target)
        shard_rows.append({
            'path': target.name, 'sha256': sha256_file(target),
            'byte_length': target.stat().st_size,
            'object_key': (
                f'synthetic-motion/v1/snapshots/{snapshot_id}/shards/'
                + target.name),
            'sequences': report['sequences']})
    manifest = {
        'schema': 'imu_motion_simulator.synthetic_snapshot.v1',
        'snapshot_id': snapshot_id,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'prefix': 'synthetic-motion/v1', 'hdf5_version': '3.3.0',
        'candidate_manifest': {
            'path': str(candidate_manifest),
            'sha256': sha256_file(candidate_manifest)},
        'review_revisions': revision_rows,
        'included_candidates': included, 'excluded_candidates': excluded,
        'shards': shard_rows,
        'media_policy': {'mp4': 'excluded', 'replay': 'required',
                         'shared_assets': 'deduplicated-within-shard'}}
    validate_snapshot_manifest(manifest, root=working)
    _atomic_json(working / 'snapshot.json', manifest)
    working.rename(output)
    return manifest


def build_handoff_fixture(output):
    """Create a local-only, runnable two-candidate handoff example."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    objects = output / 'objects'; objects.mkdir()
    descriptors = []
    role_hashes = {}
    for role, content in (
            ('motion', b'artificial canonical motion fixture\n'),
            ('sensors', b'artificial virtual IMU fixture\n'),
            ('selection', b'{"fixture":true}\n')):
        path = objects / role
        path.write_bytes(content)
        digest = sha256_file(path); role_hashes[role] = digest
        suffix = 'selection.json' if role == 'selection' else role + '.h5'
        descriptors.append({
            'role': role, 'sha256': digest,
            'byte_length': path.stat().st_size,
            'object_key': f'synthetic-motion/v1/objects/{digest}/{suffix}',
            'local_path': str(path)})
    candidates = []
    for candidate_id, action in (('fixture-pass', 'walk'),
                                 ('fixture-reject', 'jump')):
        candidates.append({
            'candidate_id': candidate_id, 'source_dataset': 'fixture',
            'source_member': 'fixture/' + candidate_id + '.npz',
            'label_candidates': [{'code': action, 'name': action.title(),
                                  'is_fall': False, 'origin': 'fixture'}],
            'warning_flags': [], 'objects': dict(role_hashes)})
    corpus = {
        'schema': 'imu_motion_simulator.candidate_corpus.v1',
        'corpus_id': 'handoff-fixture-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'prefix': 'synthetic-motion/v1',
        'policy': {'policy_id': 'fixture', 'sha256': '0' * 64,
                   'path': 'fixture'},
        'inputs': {'fixture': True},
        'statistics': {'total_clips': 2,
                       'machine_passed_before_quarantine': 2,
                       'published_candidates': 2, 'candidate_rate': 1.,
                       'publishable': True,
                       'minimum_corpus_pass_rate': .8},
        'objects': descriptors, 'candidates': candidates,
        'quarantined_sources': [], 'excluded': []}
    validate_candidate_corpus(corpus)
    candidate_path = output / 'candidate-corpus.json'
    _atomic_json(candidate_path, corpus)
    reviews = output / 'reviews'
    for candidate_id, decision in (('fixture-pass', 'pass'),
                                   ('fixture-reject', 'reject')):
        directory = reviews / candidate_id; directory.mkdir(parents=True)
        value = {
            'schema': 'imu_motion_simulator.human_review_revision.v1',
            'candidate_id': candidate_id, 'revision': 1,
            'decision': decision, 'reviewer': 'fixture-reviewer',
            'reason': None if decision == 'pass' else 'visible mismatch',
            'labels': ([{'code': 'walk', 'name': 'Walk',
                         'is_fall': False}] if decision == 'pass' else []),
            'previous_revision_sha256': None,
            'created_at_utc': datetime.now(timezone.utc).isoformat()}
        validate_review_revision(value)
        _atomic_json(directory / 'review-r1.json', value)
    seed = output / '.fixture-seed'; seed.write_bytes(b'fixture')
    core, labels, _, replay, assets, blobs = fixture_inputs(seed)
    shard = output / '.fixture-shard.h5'
    write_delivery(shard, **core, labels=labels, replay=replay,
                   assets=assets, blobs=blobs)
    seed.unlink()
    snapshot = build_snapshot(
        candidate_path, reviews, [shard], output / 'snapshot',
        snapshot_id='handoff-fixture-snapshot-v1')
    shard.unlink()
    return {'valid': True, 'candidates': 2,
            'included': len(snapshot['included_candidates']),
            'excluded': len(snapshot['excluded_candidates']),
            'output': str(output)}
