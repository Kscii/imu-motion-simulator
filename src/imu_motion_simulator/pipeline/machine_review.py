"""Deterministic machine gate and compact candidate-corpus manifests."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath

from ..contracts.common import ContractError, require, sha256_file, sha_string
from ..sensors.convergence import upgrade_legacy_convergence
from .plan import load_plan


POLICY_FIELDS = {
    'schema', 'policy_id', 'minimum_corpus_pass_rate',
    'minimum_source_pass_rate', 'required_checks', 'convergence'}
CONVERGENCE_FIELDS = {
    'require_boundary_guard_complete', 'acceleration_rms_max_m_s2',
    'angular_velocity_rms_max_rad_s'}
KNOWN_CHECKS = {
    'structure', 'trajectory', 'sensor', 'work_grid_convergence'}
ARTIFACT_ROLES = ('motion', 'sensors', 'selection')


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_machine_policy(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    require(set(value) in (POLICY_FIELDS, POLICY_FIELDS | {'paused_sources'}),
            'machine policy fields')
    require(value['schema'] == 'imu_motion_simulator.machine_review_policy.v1',
            'machine policy schema')
    require(isinstance(value['policy_id'], str) and value['policy_id'],
            'machine policy identity')
    for field in ('minimum_corpus_pass_rate', 'minimum_source_pass_rate'):
        require(type(value[field]) in (int, float)
                and 0 < float(value[field]) <= 1, field)
    paused = value.setdefault('paused_sources', [])
    require(isinstance(paused, list) and len(paused) == len(set(paused))
            and all(isinstance(source, str) and source for source in paused),
            'paused sources')
    checks = value['required_checks']
    require(isinstance(checks, list) and checks
            and len(checks) == len(set(checks))
            and set(checks) <= KNOWN_CHECKS,
            'machine policy checks')
    convergence = value['convergence']
    require(set(convergence) == CONVERGENCE_FIELDS,
            'machine convergence fields')
    require(type(convergence['require_boundary_guard_complete']) is bool,
            'machine boundary guard policy')
    for field in ('acceleration_rms_max_m_s2',
                  'angular_velocity_rms_max_rad_s'):
        require(type(convergence[field]) in (int, float)
                and float(convergence[field]) >= 0, field)
    return value


def evaluate_machine_qa(qa, policy, *, lineage_reasons=()):
    """Apply only hard structural/numerical gates; warnings remain advisory."""
    reasons = list(lineage_reasons)
    if not isinstance(qa, dict):
        reasons.append('qa-missing')
        return {'passed': False, 'reasons': sorted(set(reasons)),
                'warnings': []}
    checks = qa.get('checks')
    if not isinstance(checks, dict):
        reasons.append('qa-checks-missing')
        checks = {}
    for name in policy['required_checks']:
        check = checks.get(name)
        if not isinstance(check, dict):
            reasons.append('hard-check-missing:' + name)
        elif name != 'work_grid_convergence' and check.get('pass') is not True:
            reasons.append('hard-check-failed:' + name)
    convergence = checks.get('work_grid_convergence', {})
    limits = policy['convergence']
    if limits['require_boundary_guard_complete'] \
            and convergence.get('boundary_guard_complete') is not True:
        reasons.append('boundary-guard-incomplete')
    values = (
        ('acceleration_rms_m_s2', 'acceleration_rms_max_m_s2'),
        ('angular_velocity_rms_rad_s',
         'angular_velocity_rms_max_rad_s'))
    for actual, limit in values:
        value = convergence.get(actual)
        if type(value) not in (int, float) or value > limits[limit]:
            reasons.append('convergence-limit:' + actual)
    warnings = qa.get('warnings', [])
    if not isinstance(warnings, list):
        reasons.append('qa-warnings-invalid')
        warnings = []
    return {'passed': not reasons, 'reasons': sorted(set(reasons)),
            'warnings': list(warnings)}


def _report_path(batch, record, source):
    status = record.get('status')
    if status == 'complete':
        return batch / 'sources' / source / 'run/study-report.json', \
            batch / 'sources' / source / 'run'
    if status == 'reused-existing-study':
        study = Path(record['evidence_path']).resolve()
        report = study / 'run/study-report.json'
        if sha256_file(report) != record.get('evidence_report_sha256'):
            raise ContractError('reused report changed: ' + source)
        return report, study / 'run'
    return None, None


def _object_descriptor(path, role, *, digest=None):
    path = Path(path).resolve()
    digest = sha256_file(path) if digest is None else digest
    suffix = {'motion': 'motion.h5', 'sensors': 'sensors.h5',
              'selection': 'selection.json'}[role]
    return {
        'role': role, 'sha256': digest, 'byte_length': path.stat().st_size,
        'object_key': f'synthetic-motion/v1/objects/{digest}/{suffix}',
        'local_path': str(path)}


def _record_lineage(run, record):
    reasons, objects = [], {}
    for role in ARTIFACT_ROLES:
        relative = record.get(role)
        expected = record.get(role + '_sha256')
        if not isinstance(relative, str) or not isinstance(expected, str):
            reasons.append('lineage-field-missing:' + role)
            continue
        logical = PurePosixPath(relative)
        if logical.is_absolute() or '..' in logical.parts:
            reasons.append('lineage-path-invalid:' + role)
            continue
        path = (run / relative).resolve()
        if not path.is_relative_to(run.resolve()) or not path.is_file():
            reasons.append('payload-missing:' + role)
            continue
        actual = sha256_file(path)
        if actual != expected:
            reasons.append('payload-hash-mismatch:' + role)
            continue
        objects[role] = _object_descriptor(path, role, digest=actual)
    return reasons, objects


def _qa_with_record_convergence(record):
    """Use the canonical clip record to upgrade pre-boundary-flag QA reports."""
    qa = record.get('automatic_qa')
    if not isinstance(qa, dict):
        return qa
    convergence = upgrade_legacy_convergence(record.get('convergence'))
    if convergence is None:
        return qa
    checks = qa.get('checks')
    if not isinstance(checks, dict):
        return qa
    return {**qa, 'checks': {**checks, 'work_grid_convergence': {
        **convergence,
        'pass': checks.get('work_grid_convergence', {}).get('pass', False)}}}


def validate_candidate_corpus(value):
    fields = {'schema', 'corpus_id', 'created_at_utc', 'prefix',
              'policy', 'inputs', 'statistics', 'objects', 'candidates',
              'quarantined_sources', 'excluded'}
    require(isinstance(value, dict) and set(value) == fields,
            'candidate corpus fields')
    require(value['schema'] == 'imu_motion_simulator.candidate_corpus.v1',
            'candidate corpus schema')
    require(isinstance(value['corpus_id'], str) and value['corpus_id'],
            'candidate corpus identity')
    require(value['prefix'] == 'synthetic-motion/v1',
            'candidate corpus prefix')
    require(isinstance(value['policy'], dict)
            and set(value['policy']) == {'path', 'sha256', 'policy_id'}
            and sha_string(value['policy']['sha256'])
            and isinstance(value['policy']['policy_id'], str)
            and value['policy']['policy_id'], 'candidate corpus policy')
    statistic_fields = {
        'total_clips', 'machine_passed_before_quarantine',
        'published_candidates', 'candidate_rate', 'publishable',
        'minimum_corpus_pass_rate'}
    stats = value['statistics']
    require(isinstance(stats, dict) and set(stats) == statistic_fields,
            'candidate statistics fields')
    for field in ('total_clips', 'machine_passed_before_quarantine',
                  'published_candidates'):
        require(type(stats[field]) is int and stats[field] >= 0,
                'candidate statistic: ' + field)
    require(stats['total_clips'] > 0, 'candidate corpus is empty')
    require(type(stats['candidate_rate']) in (int, float)
            and 0 <= stats['candidate_rate'] <= 1
            and type(stats['minimum_corpus_pass_rate']) in (int, float)
            and 0 < stats['minimum_corpus_pass_rate'] <= 1
            and stats['publishable'] is True,
            'candidate publication statistics')
    object_ids = set(); object_roles = {}
    for item in value['objects']:
        require(set(item) == {
            'role', 'sha256', 'byte_length', 'object_key', 'local_path'},
                'candidate object fields')
        require(item['role'] in ARTIFACT_ROLES and sha_string(item['sha256'])
                and type(item['byte_length']) is int
                and item['byte_length'] >= 0,
                'candidate object identity')
        require(item['object_key'].startswith(
            'synthetic-motion/v1/objects/' + item['sha256'] + '/'),
            'candidate object key')
        require(isinstance(item['local_path'], str) and item['local_path'],
                'candidate local object path')
        require(item['sha256'] not in object_ids, 'duplicate candidate object')
        object_ids.add(item['sha256'])
        object_roles[item['sha256']] = item['role']
    candidate_ids = set()
    for item in value['candidates']:
        require(set(item) == {
            'candidate_id', 'source_dataset', 'source_member',
            'label_candidates', 'warning_flags', 'objects'},
                'candidate fields')
        require(item['candidate_id'] not in candidate_ids,
                'duplicate candidate')
        require(set(item['objects']) == set(ARTIFACT_ROLES)
                and all(digest in object_ids
                        for digest in item['objects'].values()),
                'candidate object references')
        require(all(object_roles[digest] == role
                    for role, digest in item['objects'].items()),
                'candidate object roles')
        require(isinstance(item['source_dataset'], str)
                and item['source_dataset']
                and isinstance(item['source_member'], str)
                and item['source_member']
                and isinstance(item['label_candidates'], list)
                and isinstance(item['warning_flags'], list)
                and all(isinstance(flag, str)
                        for flag in item['warning_flags']),
                'candidate metadata')
        candidate_ids.add(item['candidate_id'])
    excluded_ids = set()
    for item in value['excluded']:
        require(set(item) == {
            'candidate_id', 'source_dataset', 'source_member', 'reasons'}
            and isinstance(item['candidate_id'], str)
            and item['candidate_id']
            and isinstance(item['reasons'], list) and item['reasons']
            and all(isinstance(reason, str) for reason in item['reasons']),
            'excluded candidate')
        require(item['candidate_id'] not in candidate_ids
                and item['candidate_id'] not in excluded_ids,
                'excluded candidate identity')
        excluded_ids.add(item['candidate_id'])
    require(isinstance(value['quarantined_sources'], list)
            and len(value['quarantined_sources'])
            == len(set(value['quarantined_sources']))
            and all(isinstance(source, str) and source
                    for source in value['quarantined_sources']),
            'quarantined sources')
    require(stats['published_candidates'] == len(value['candidates']),
            'candidate count')
    require(stats['total_clips'] == len(candidate_ids) + len(excluded_ids)
            and stats['machine_passed_before_quarantine']
            >= stats['published_candidates']
            and abs(stats['candidate_rate']
                    - len(candidate_ids) / stats['total_clips']) < 1e-12,
            'candidate statistics')
    require(set(object_ids) == {
        digest for item in value['candidates']
        for digest in item['objects'].values()},
        'unreferenced candidate object')
    return {'valid': True, 'candidates': len(candidate_ids),
            'objects': len(object_ids)}


def build_machine_corpus(catalog_directory, batch_directory, output, policy_path):
    """Audit an existing batch without rebuilding motion or sensor payloads."""
    catalog_directory = Path(catalog_directory).resolve()
    batch = Path(batch_directory).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    catalog_path = catalog_directory / 'catalog.json'
    state_path = batch / 'batch-state.json'
    catalog = json.loads(catalog_path.read_text())
    state = json.loads(state_path.read_text())
    if catalog.get('schema') not in {
            'imu_motion_simulator.amass_catalog.v1',
            'imu_motion_simulator.amass_catalog.v2'} \
            or state.get('schema') != 'imu_motion_simulator.amass_batch_state.v1':
        raise ContractError('unsupported catalog or batch')
    if state['config']['catalog_sha256'] != sha256_file(catalog_path):
        raise ContractError('batch does not match catalog')
    policy = load_machine_policy(policy_path)
    ready = {item['source_dataset']: item for item in catalog['sources']
             if item['status'] == 'plan-ready'}
    source_rows, clip_rows, object_map = [], [], {}
    total = 0
    for source, item in sorted(ready.items()):
        plan_path = catalog_directory / item['plan']
        if sha256_file(plan_path) != item['plan_sha256']:
            raise ContractError('catalog plan changed: ' + source)
        plan = load_plan(plan_path)
        total += len(plan['clips'])
        state_record = state['sources'].get(source, {})
        report_path, run = _report_path(batch, state_record, source)
        report_clips = {}
        source_error = None
        if report_path is None or not report_path.is_file():
            source_error = 'source-report-missing'
        else:
            report = json.loads(report_path.read_text())
            report_clips = report.get('clips', {})
            if report.get('clips_planned') != len(plan['clips']) \
                    or len(report_clips) != len(plan['clips']):
                source_error = 'source-report-count-mismatch'
        passed = 0
        for clip in plan['clips']:
            record = report_clips.get(clip['id'])
            lineage_reasons = []
            objects = {}
            if source_error:
                lineage_reasons.append(source_error)
            elif not isinstance(record, dict):
                lineage_reasons.append('clip-report-missing')
            else:
                lineage_reasons, objects = _record_lineage(run, record)
            result = evaluate_machine_qa(
                None if not isinstance(record, dict)
                else _qa_with_record_convergence(record), policy,
                lineage_reasons=lineage_reasons)
            if result['passed']:
                passed += 1
                for descriptor in objects.values():
                    existing = object_map.get(descriptor['sha256'])
                    if existing is not None and existing != descriptor:
                        raise ContractError('object digest collision')
                    object_map[descriptor['sha256']] = descriptor
            clip_rows.append({
                'candidate_id': clip['id'], 'source_dataset': source,
                'source_member': clip['source_member'],
                'label_candidates': clip['label_candidates'],
                'warning_flags': result['warnings'],
                'machine_passed': result['passed'],
                'failure_reasons': result['reasons'],
                'objects': objects})
        rate = passed / len(plan['clips'])
        source_rows.append({
            'source_dataset': source, 'clips': len(plan['clips']),
            'passed': passed, 'failed': len(plan['clips']) - passed,
            'pass_rate': rate,
            'quarantined': source in policy['paused_sources']})
    quarantined = {row['source_dataset'] for row in source_rows
                   if row['quarantined']}
    candidates = []
    failure_counts = Counter()
    excluded = []
    for row in clip_rows:
        if not row['machine_passed'] or row['source_dataset'] in quarantined:
            reasons = list(row['failure_reasons'])
            if row['source_dataset'] in quarantined:
                reasons.append('source-quarantined')
            reasons = sorted(set(reasons))
            failure_counts.update(reasons)
            excluded.append({
                'candidate_id': row['candidate_id'],
                'source_dataset': row['source_dataset'],
                'source_member': row['source_member'], 'reasons': reasons})
            continue
        candidates.append({key: row[key] for key in (
            'candidate_id', 'source_dataset', 'source_member',
            'label_candidates', 'warning_flags')} | {
                'objects': {role: row['objects'][role]['sha256']
                            for role in ARTIFACT_ROLES}})
    candidate_rate = len(candidates) / total if total else 0.
    # A low pass rate is an alert for investigation, never a corpus-wide veto.
    publishable = True
    referenced_objects = {
        digest for candidate in candidates
        for digest in candidate['objects'].values()}
    audit = {
        'schema': 'imu_motion_simulator.machine_audit.v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'policy': {'path': str(Path(policy_path).resolve()),
                   'sha256': sha256_file(policy_path),
                   'policy_id': policy['policy_id']},
        'inputs': {'catalog': str(catalog_path),
                   'catalog_sha256': sha256_file(catalog_path),
                   'batch': str(batch), 'batch_state_sha256': sha256_file(state_path)},
        'statistics': {
            'total_clips': total, 'machine_passed_before_quarantine': sum(
                row['passed'] for row in source_rows),
            'published_candidates': len(candidates),
            'candidate_rate': candidate_rate, 'publishable': publishable,
            'minimum_corpus_pass_rate': policy['minimum_corpus_pass_rate']},
        'failure_reasons': dict(sorted(failure_counts.items())),
        'sources': source_rows,
        'quarantined_sources': sorted(quarantined)}
    output.mkdir(parents=True)
    _atomic_json(output / 'machine-audit.json', audit)
    corpus = {
        'schema': 'imu_motion_simulator.candidate_corpus.v1',
        'corpus_id': output.name,
        'created_at_utc': audit['created_at_utc'],
        'prefix': 'synthetic-motion/v1',
        'policy': audit['policy'], 'inputs': audit['inputs'],
        'statistics': audit['statistics'],
        'objects': sorted((object_map[digest] for digest in referenced_objects),
                          key=lambda item: item['sha256']),
        'candidates': candidates,
        'quarantined_sources': sorted(quarantined), 'excluded': excluded}
    validate_candidate_corpus(corpus)
    _atomic_json(output / 'candidate-corpus.json', corpus)
    return audit


def merge_candidate_corpora(corpora, output):
    """Combine independently audited native/adapter corpora without payload copies."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    paths, values = [], []
    for source in map(Path, corpora):
        path = source / 'candidate-corpus.json' if source.is_dir() else source
        path = path.resolve(); value = json.loads(path.read_text())
        validate_candidate_corpus(value)
        paths.append(path); values.append(value)
    require(bool(values), 'candidate corpora are required')
    policy_hash = values[0]['policy']['sha256']
    require(all(value['policy']['sha256'] == policy_hash for value in values),
            'candidate corpus policies differ')
    objects, candidates, excluded = {}, {}, {}
    for value in values:
        for item in value['objects']:
            previous = objects.get(item['sha256'])
            if previous is not None:
                comparable = {'role', 'sha256', 'byte_length', 'object_key'}
                require(all(previous[key] == item[key] for key in comparable),
                        'candidate object identity conflict')
            else:
                objects[item['sha256']] = item
        for item in value['candidates']:
            require(item['candidate_id'] not in candidates
                    and item['candidate_id'] not in excluded,
                    'candidate identity conflict')
            candidates[item['candidate_id']] = item
        for item in value['excluded']:
            require(item['candidate_id'] not in excluded
                    and item['candidate_id'] not in candidates,
                    'excluded candidate identity conflict')
            excluded[item['candidate_id']] = item
    total = sum(value['statistics']['total_clips'] for value in values)
    published = len(candidates)
    minimum = values[0]['statistics']['minimum_corpus_pass_rate']
    corpus = {
        'schema': 'imu_motion_simulator.candidate_corpus.v1',
        'corpus_id': output.name,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'prefix': 'synthetic-motion/v1', 'policy': values[0]['policy'],
        'inputs': {'component_corpora': [{
            'path': str(path), 'sha256': sha256_file(path)}
            for path in paths]},
        'statistics': {
            'total_clips': total,
            'machine_passed_before_quarantine': sum(
                value['statistics']['machine_passed_before_quarantine']
                for value in values),
            'published_candidates': published,
            'candidate_rate': published / total,
            'publishable': True,
            'minimum_corpus_pass_rate': minimum},
        'objects': sorted(objects.values(), key=lambda item: item['sha256']),
        'candidates': [candidates[key] for key in sorted(candidates)],
        'quarantined_sources': sorted({
            source for value in values
            for source in value['quarantined_sources']}),
        'excluded': [excluded[key] for key in sorted(excluded)]}
    validate_candidate_corpus(corpus)
    output.mkdir(parents=True)
    _atomic_json(output / 'candidate-corpus.json', corpus)
    return {'valid': True, 'output': str(output),
            'components': len(values), 'candidates': published,
            'total_clips': total,
            'candidate_rate': corpus['statistics']['candidate_rate']}
