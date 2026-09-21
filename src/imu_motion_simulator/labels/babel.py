"""Read BABEL 1.0 labels as non-authoritative review candidates."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile


def _annotations(value, name):
    singular = value.get(name)
    if singular is not None:
        yield singular
    yield from (value.get(name + 's') or [])


def load_index(archive):
    """Load all primary and extra annotations once, grouped by AMASS member.

    BABEL intentionally repeats a sequence in ``train``/``extra_train`` (and
    equivalently for validation) when extra annotators are available.  Those
    files are annotation sets, not conflicting dataset splits.
    """
    archive = Path(archive); grouped = {}
    with zipfile.ZipFile(archive) as handle:
        names = sorted(name for name in handle.namelist()
                       if name.endswith('.json') and '/._' not in name)
        for split_path in names:
            split = Path(split_path).stem
            values = json.loads(handle.read(split_path))
            for value in values.values():
                source_member = value.get('feat_p')
                if not source_member:
                    continue
                key = (source_member, str(value['babel_sid']))
                result = grouped.setdefault(key, {
                    'babel_sid': value['babel_sid'], 'splits': [],
                    'source_member': source_member,
                    'duration_s': float(value['dur']), 'labels': []})
                if abs(result['duration_s'] - float(value['dur'])) > 1e-6:
                    raise ValueError('BABEL duplicate duration mismatch')
                result['splits'].append(split)
                for field, kind in (('seq_ann', 'recording-candidate'),
                                    ('frame_ann', 'temporal-candidate')):
                    for annotation_set, annotation in enumerate(
                            _annotations(value, field)):
                        for item in annotation.get('labels', []):
                            label = {
                                'kind': kind,
                                'code': item.get('proc_label') or item['raw_label'],
                                'raw_label': item['raw_label'],
                                'categories': item.get('act_cat', []),
                                'start_time_s': (None if field == 'seq_ann'
                                                 else float(item['start_t'])),
                                'stop_time_s': (None if field == 'seq_ann'
                                                else float(item['end_t'])),
                                'origin': 'babel-1.0', 'release_set': split,
                                'annotation_set': annotation_set}
                            result['labels'].append(label)
    index = {}
    for (member, _), value in grouped.items():
        if member in index:
            raise ValueError('BABEL member has multiple sequence identities')
        value['splits'] = sorted(set(value['splits']))
        index[member] = value
    return index


def candidate_from_index(index, source_member, *, source_dataset=None):
    """Match exact paths and explicitly bounded dataset-root aliases."""
    parts = Path(source_member).parts
    aliases = [source_member]
    if source_dataset:
        aliases.append(str(Path(source_dataset) / source_member))
        if parts and parts[0] == source_dataset and len(parts) > 1:
            aliases.append(str(Path(*parts[1:])))
    if len(parts) > 1:
        aliases.append(str(Path(parts[0]) / source_member))
        if parts[0] == parts[1]:
            aliases.append(str(Path(*parts[1:])))
    matches = [index[name] for name in dict.fromkeys(aliases) if name in index]
    identities = {str(value['babel_sid']) for value in matches}
    if len(identities) > 1:
        raise ValueError('BABEL aliases resolve to multiple sequence identities')
    return None if not matches else matches[0]


def candidates_for_member(archive, source_member, *, source_dataset=None):
    return candidate_from_index(
        load_index(archive), source_member, source_dataset=source_dataset)
