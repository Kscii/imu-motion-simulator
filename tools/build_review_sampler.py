"""Build a small, resumable, unreviewed cross-source Three.js sample set."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

from imu_motion_simulator.contracts.common import sha256_file
from imu_motion_simulator.contracts.internal import read_internal
from imu_motion_simulator.motion.kinematics import load_model_member
from imu_motion_simulator.review.bundle import (build_bundle,
                                                review_recipe_sha256,
                                                validate_bundle)


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def recording_codes(candidate):
    return {item.get('code') for item in candidate['label_candidates']
            if item.get('kind') != 'temporal-candidate' and item.get('code')}


def choose_samples(candidates, qa_by_source, *, max_duration_s=30.):
    """Choose at most one representative of each available category per source."""
    grouped = defaultdict(list)
    for candidate in candidates:
        source = candidate['source_dataset']
        qa = qa_by_source[source].get(candidate['candidate_id'])
        if qa is None or not qa['passed']:
            raise ValueError('Candidate lacks passing QA: ' + candidate['candidate_id'])
        duration = (qa['checks']['structure']['sensor_samples'] - 1) / 25.
        grouped[source].append((candidate, qa, duration))
    selected = []
    for source, rows in sorted(grouped.items()):
        within_cap = [row for row in rows if row[2] <= max_duration_s]
        rows = within_cap or [min(rows, key=lambda row: (row[2],
                                                        row[0]['candidate_id']))]
        used = set()
        groups = {
            'ordinary': [row for row in rows if not row[1]['warnings']
                         and len(recording_codes(row[0])) == 1],
            'warning': [row for row in rows if row[1]['warnings']],
            'label-ambiguous': [row for row in rows
                                if len(recording_codes(row[0])) != 1],
            'high-risk': list(rows),
        }
        for category in ('ordinary', 'warning', 'label-ambiguous', 'high-risk'):
            eligible = [row for row in groups[category]
                        if row[0]['candidate_id'] not in used]
            if not eligible:
                continue
            if category == 'high-risk':
                eligible.sort(key=lambda row: (-row[1]['risk_score'],
                                               row[2], row[0]['candidate_id']))
            else:
                eligible.sort(key=lambda row: (abs(row[2] - 5.),
                                               -row[1]['risk_score'],
                                               row[0]['candidate_id']))
            candidate, qa, duration = eligible[0]
            used.add(candidate['candidate_id'])
            selected.append({'source': source, 'category': category,
                             'candidate': candidate, 'qa': qa,
                             'duration_s': duration})
    return selected


HTML = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>跨来源动作审核样本</title>
<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}
input{font:inherit;width:100%;padding:.5rem;box-sizing:border-box}table{border-collapse:collapse;width:100%}
td,th{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}a{color:#075ca8}</style>
<h1>跨来源动作审核样本</h1><p>仅供观察；所有动作仍是 unreviewed。输入和产物哈希见 <a href="index.json">index.json</a>。</p>
<input id="search" placeholder="筛选来源、类别、动作 ID"><p id="count"></p><table><thead><tr>
<th>来源</th><th>类别</th><th>动作</th><th>时长</th><th>风险</th><th>warning</th></tr></thead><tbody id="rows"></tbody></table>
<script>const box=document.querySelector('#search'),rows=document.querySelector('#rows'),count=document.querySelector('#count');
fetch('index.json').then(r=>r.json()).then(data=>{const render=()=>{rows.replaceChildren();
const list=data.bundles.filter(x=>(x.source+' '+x.category+' '+x.candidate_id).toLowerCase().includes(box.value.toLowerCase()));
count.textContent=list.length+' / '+data.bundles.length+' 个审核包';for(const x of list){const tr=document.createElement('tr');
for(const v of [x.source,x.category,x.candidate_id,x.duration_s.toFixed(1)+' s',x.risk_score.toFixed(3),x.warnings.join(', ')]){
const td=document.createElement('td');if(v===x.candidate_id){const a=document.createElement('a');a.href=x.bundle+'/index.html';a.textContent=v;td.append(a)}else td.textContent=v;tr.append(td)}rows.append(tr)}};
box.addEventListener('input',render);render()});</script></html>'''


def build_sample_set(corpus_path, production, library, checkout, output):
    corpus_path, production, library, checkout, output = map(
        lambda path: Path(path).resolve(),
        (corpus_path, production, library, checkout, output))
    corpus = json.loads(corpus_path.read_text())
    if corpus['schema'] != 'imu_motion_simulator.candidate_corpus.v1':
        raise ValueError('Unsupported candidate corpus')
    source_groups = {candidate['source_dataset'] for candidate in corpus['candidates']}
    source_data = {}
    for source in source_groups:
        group = 'stageii' if source in {'GRAB', 'SOMA'} else 'native'
        root = production / group / 'sources' / source
        plan_path = root / 'plan.json'
        report_path = root / 'run/study-report.json'
        plan = json.loads(plan_path.read_text())
        report = json.loads(report_path.read_text())
        source_data[source] = {
            'plan': plan, 'plan_sha256': sha256_file(plan_path),
            'report_sha256': sha256_file(report_path),
            'qa': {key: value['automatic_qa']
                   for key, value in report['clips'].items()}}
    chosen = choose_samples(corpus['candidates'],
                            {key: value['qa'] for key, value in source_data.items()})
    if {row['source'] for row in chosen} != source_groups:
        raise ValueError('A source has no reviewable clip under the duration cap')
    objects = {item['sha256']: item for item in corpus['objects']}
    binding = {'schema': 'imu_motion_simulator.review_sample_binding.v1',
               'corpus_sha256': sha256_file(corpus_path),
               'sampler_sha256': sha256_file(__file__),
               'review_recipe_sha256': review_recipe_sha256(),
               'source_evidence': {source: {
                   'plan_sha256': data['plan_sha256'],
                   'report_sha256': data['report_sha256']}
                   for source, data in sorted(source_data.items())}}
    output.mkdir(parents=True, exist_ok=True)
    binding_path = output / 'binding.json'
    if binding_path.exists():
        if json.loads(binding_path.read_text()) != binding:
            raise ValueError('Existing review set has different inputs or code')
    else:
        _write_json(binding_path, binding)
    index_path = output / 'index.json'
    existing = {row['candidate_id']: row for row in
                json.loads(index_path.read_text()).get('bundles', [])} \
               if index_path.exists() else {}
    cache = {}
    for row in chosen:
        candidate = row['candidate']; source = row['source']
        clip_id = candidate['candidate_id']
        paths = {role: Path(objects[digest]['local_path'])
                 for role, digest in candidate['objects'].items()}
        for role, path in paths.items():
            if sha256_file(path) != candidate['objects'][role]:
                raise ValueError('Corpus object hash mismatch: ' + str(path))
        plan = source_data[source]['plan']
        model_archive = library / plan['inputs']['smplh_archive']
        layout_path = checkout / plan['sensor']['layout']
        report = source_data[source]['qa'][clip_id]
        convergence = report['checks']['work_grid_convergence']
        destination = output / 'bundles' / source / (clip_id + '.review')
        if not destination.exists():
            temporary = destination.with_name(destination.name + '.partial')
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.parent.mkdir(parents=True, exist_ok=True)
            _, metadata, _ = read_internal(paths['motion'], 'motion')
            gender = metadata['kind_metadata']['source_gender']
            if gender not in cache:
                cache[gender] = load_model_member(model_archive, gender)
            build_bundle(paths['motion'], paths['sensors'], model_archive,
                         layout_path, temporary, selection=paths['selection'],
                         convergence=convergence, model=cache[gender])
            temporary.replace(destination)
        validation = validate_bundle(destination)
        manifest_path = destination / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if (manifest['motion_sha256'] != candidate['objects']['motion']
                or manifest['sensor_sha256'] != candidate['objects']['sensors']
                or manifest['recipe_sha256'] != binding['review_recipe_sha256']
                or not manifest['qa']['passed']
                or validation['decision'] != 'unreviewed'):
            raise ValueError('Review bundle differs from machine candidate: ' + clip_id)
        existing[clip_id] = {
            'candidate_id': clip_id, 'source': source,
            'source_member': candidate['source_member'],
            'category': row['category'], 'duration_s': row['duration_s'],
            'risk_score': row['qa']['risk_score'],
            'warnings': row['qa']['warnings'],
            'label_codes': sorted(recording_codes(candidate)),
            'bundle': str(destination.relative_to(output)),
            'manifest_sha256': sha256_file(manifest_path),
            'input_objects': candidate['objects'], 'decision': 'unreviewed'}
        _write_json(index_path, {
            'schema': 'imu_motion_simulator.review_sample_index.v1',
            'binding_sha256': sha256_file(binding_path),
            'planned': len(chosen),
            'bundles': sorted(existing.values(), key=lambda item:
                              (item['source'], item['category'], item['candidate_id']))})
    (output / 'index.html').write_text(HTML)
    return {'sources': len(source_groups), 'planned': len(chosen),
            'built': len(existing), 'output': str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('production', type=Path)
    parser.add_argument('library', type=Path)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(build_sample_set(args.corpus, args.production,
                                      args.library, args.checkout, args.output),
                     indent=2))
