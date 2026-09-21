"""Audit acquired originals and write source HDF5; no SI calibration or simulation."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np

from ..contracts.common import relative_path, require, sha256_file
from ..contracts.internal import source_metadata, write_source
from .motion import parse

CATALOG_SCHEMA = 'imu_motion_simulator.source_catalog.v1'


def audit_zip(path):
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, 'ZIP CRC failure')
        files, paths = [], set()
        for item in archive.infolist():
            relative_path(item.filename.rstrip('/'))
            require(item.filename not in paths and (item.external_attr >> 16) & 0o170000 != 0o120000, 'duplicate/symlink ZIP member')
            paths.add(item.filename)
            if item.is_dir(): continue
            raw = archive.read(item)
            row = dict(path=item.filename, byte_length=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            if item.filename.lower().endswith('.bvh'):
                meta, _ = parse(raw)
                row['bvh'] = meta
            files.append(row)
        return dict(crc_passed=True, files=files, bvh_count=sum('bvh' in x for x in files),
                    expanded_bytes=sum(x['byte_length'] for x in files), execution_ready=False)


def audit_amass(path):
    rows, names, frames, durations = [], set(), 0, 0.0
    with tarfile.open(path, 'r|bz2') as archive:
        for item in archive:
            relative_path(item.name.rstrip('/'))
            require(not item.issym() and not item.islnk() and not item.isdev(), 'unsafe TAR member')
            require(item.name not in names, 'duplicate TAR member')
            names.add(item.name)
            if not item.isfile(): continue
            raw = archive.extractfile(item).read()
            require(len(raw) == item.size, 'truncated TAR member')
            row = dict(path=item.name, byte_length=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            if item.name.endswith('.npz'):
                with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                    require(set(data.files) == {'trans', 'gender', 'mocap_framerate', 'betas', 'dmpls', 'poses'}, 'unexpected AMASS fields')
                    arrays = {key: data[key] for key in data.files}
                    n = len(arrays['poses']); rate = float(arrays['mocap_framerate'])
                    require(n > 0 and np.isfinite(rate) and rate > 0, 'AMASS frames/rate')
                    for key, shape in [('poses', (n, 156)), ('trans', (n, 3)), ('dmpls', (n, 8)), ('betas', (16,))]:
                        require(arrays[key].shape == shape and np.isfinite(arrays[key]).all(), 'AMASS shape/nonfinite: ' + key)
                    require(arrays['gender'].shape == () and str(arrays['gender']) in ('male', 'female', 'neutral'), 'AMASS gender')
                    row.update(frames=n, declared_rate_hz=rate, gender=str(arrays['gender']),
                               arrays={k: dict(dtype=str(a.dtype), shape=list(a.shape)) for k, a in arrays.items()})
                    frames += n; durations += n / rate
            rows.append(row)
    return dict(files=rows, npz_count=sum('frames' in x for x in rows), frames=frames,
                duration_frame_count_over_rate_s=durations, numeric_finite=True, execution_ready=False,
                limitations=['Body model weights not downloaded; model identity, axes and conversion must be validated before motion export.'])


def audit_babel(path, amass):
    stats, annotations, matches = {}, [], []
    amass_paths = {x['path']: x for x in amass['files'] if 'frames' in x}
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, 'BABEL ZIP CRC')
        for name in archive.namelist():
            relative_path(name.rstrip('/'))
            if not name.startswith('babel_v1.0_release/') or not name.endswith('.json'): continue
            data = json.loads(archive.read(name))
            counts, fall_sids = Counter(), set()
            matched, missing, duration_differences = 0, [], []
            for sid, record in data.items():
                feature = record['feat_p']
                relative_path(feature)
                # BABEL prefixes the dataset root once; retain both original
                # and resolved path as evidence instead of silently renaming.
                if feature.startswith('ACCAD/ACCAD/'):
                    resolved = feature[len('ACCAD/'):]
                    if resolved in amass_paths:
                        matched += 1
                        motion = amass_paths[resolved]
                        delta = abs(float(record['dur']) - motion['frames'] / motion['declared_rate_hz'])
                        duration_differences.append(delta)
                        matches.append(dict(split=Path(name).stem, babel_sid=sid, original_feat_p=feature,
                                            archive_path=resolved, duration_difference_s=delta))
                    else: missing.append(feature)
                for ann in [record.get('seq_ann'), record.get('frame_ann')] + (record.get('seq_anns') or []) + (record.get('frame_anns') or []):
                    if not ann: continue
                    for label in ann['labels']:
                        categories = label.get('act_cat') or []
                        counts.update(categories)
                        if 'fall' in categories: fall_sids.add(sid)
                        if 'fall' in categories or 'fall' in (label.get('proc_label') or '').lower():
                            annotations.append(dict(split=Path(name).stem, babel_sid=sid, feat_p=feature,
                                                    raw_label=label.get('raw_label'), proc_label=label.get('proc_label'),
                                                    act_cat=label.get('act_cat'), start_t=label.get('start_t'), end_t=label.get('end_t')))
            stats[Path(name).stem] = dict(sequences=len(data), fall_category_label_rows=counts['fall'],
                                          distinct_fall_category_sequences=len(fall_sids), accad_matches=matched,
                                          accad_missing=missing, max_duration_difference_s=max(duration_differences, default=None))
    return dict(splits=stats, accad_matches=matches, fall_search_candidates=annotations, numeric_labels_not_fall_truth=True,
                limitations=['Overlapping activity labels are preserved; do not coerce into exclusive v3 annotations.',
                             'Repeated annotations across dense/extra splits are not independent motions.',
                             'Word fall may refer to an object or avoided fall; generated outcomes need separate review.'])


def load_catalog(catalog_path, library_root):
    catalog_path, library_root = Path(catalog_path), Path(library_root).resolve()
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    require(catalog.get('schema') == CATALOG_SCHEMA, 'unsupported source catalog')
    require(catalog.get('catalog_id') and isinstance(catalog.get('sources'), list), 'invalid source catalog')
    rows, ids, paths = [], set(), set()
    for source in catalog['sources']:
        required = {'id', 'dataset', 'kind', 'path', 'source_url', 'license_url', 'observed_scope'}
        require(required <= set(source), 'incomplete source catalog entry')
        require(source['kind'] in {'amass', 'babel', 'zip'}, 'unsupported source kind')
        logical = relative_path(source['path'])
        require(source['id'] not in ids and str(logical) not in paths, 'duplicate source id or path')
        path = library_root.joinpath(*logical.parts)
        require(path.is_file(), f'missing library source: {logical}')
        ids.add(source['id']); paths.add(str(logical))
        rows.append({**source, 'resolved_path': path})
    require(any(row['kind'] == 'amass' for row in rows), 'catalog requires an AMASS source')
    for row in rows:
        if row['kind'] == 'babel':
            require(row.get('amass_source_id') in ids, 'BABEL source requires a catalogued AMASS source')
    return catalog['catalog_id'], rows


def prepare(output, library_root, catalog_path):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    catalog_id, definitions = load_catalog(catalog_path, library_root)
    producer = dict(name='imu-sim-library-sources', version='1.0.0', code_sha256=sha256_file(__file__))
    sources, audits = [], {}
    for row in definitions:
        if row['kind'] == 'amass':
            audits[row['id']] = audit_amass(row['resolved_path'])
        elif row['kind'] == 'zip':
            audits[row['id']] = audit_zip(row['resolved_path'])
    for row in definitions:
        if row['kind'] == 'babel':
            audits[row['id']] = audit_babel(row['resolved_path'], audits[row['amass_source_id']])
    for row in definitions:
        name, dataset, path, audit = row['id'], row['dataset'], row['resolved_path'], audits[row['id']]
        sha = sha256_file(path)
        provenance = dict(source_refs=[{'dataset_id': dataset, 'sha256': sha}], group_keys=[{'dataset_id': dataset}],
                          evidence=[row['source_url'], row['license_url']],
                          limitations=['Source audit only; no SI motion or accepted synthetic data.'])
        metadata = source_metadata(producer=producer, provenance=provenance, kind_metadata=dict(
            source_id=name, dataset_id=dataset, files=[dict(logical_path=path.name, sha256=sha, byte_length=path.stat().st_size,
                media_type='application/x-bzip2' if path.name.endswith('.bz2') else 'application/zip',
                source_uri=row['source_url'])],
            license=dict(source_url=row['license_url'], observed_scope=row['observed_scope'], public_delivery_approved=False),
            access_scope='registered-user-internal-research', audit=audit,
            missing_information=['Calibrated source-to-canonical motion and model-specific decoding acceptance are not complete.']))
        h5 = output / (name + '.source.h5')
        result = write_source(h5, metadata)
        (output / (name + '.audit.json')).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
        sources.append(dict(source_id=name, original_path=str(path), original_sha256=sha, original_bytes=path.stat().st_size,
                            artifact_path=h5.name, artifact_sha256=sha256_file(h5), **result))
    amass = next(audits[row['id']] for row in definitions if row['kind'] == 'amass')
    babel = next((audits[row['id']] for row in definitions if row['kind'] == 'babel'), None)
    report = dict(catalog_id=catalog_id, sources=sources, source_count=len(sources),
                  amass_npz_count=amass['npz_count'], amass_frames=amass['frames'],
                  babel_splits=None if babel is None else babel['splits'], execution_ready=False)
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--library-root', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.library_root, args.catalog), ensure_ascii=False, indent=2))
