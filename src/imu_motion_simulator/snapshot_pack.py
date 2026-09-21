"""Pack validated replay-only clip deliveries into bounded HDF5 3.3 shards."""

from __future__ import annotations

import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

import h5py
import numpy as np

from .contracts.common import LIBVER, get_json, json_dump, put_json, sha256_file, text
from .contracts.core import COLUMNS, UNITS, logical_content_sha256
from .contracts.delivery import REPLAY, table, validate_delivery, write_delivery


class StreamingShard:
    """Append validated clips, embedding each model blob once per final shard.

    The temporary file is never advertised as a delivery. A crash drops only
    the unfinished shard; checkpointed final shards remain immutable.
    """

    def __init__(self, target: Path, dataset_id: str):
        self.target = Path(target)
        self.dataset_id = dataset_id
        self.temporary = self.target.with_name(
            '.' + self.target.name + '.' + uuid4().hex + '.building')
        self.handle = h5py.File(self.temporary, 'w', libver=LIBVER)
        self.samples = self.handle.create_dataset('samples', shape=(0, 6),
                                                  maxshape=(None, 6),
                                                  chunks=(4096, 6), dtype='<f4')
        self.samples.attrs.update(columns=json_dump(COLUMNS), units=json_dump(UNITS))
        self.handle.create_group('assets/blobs')
        self.handle.create_group('replay/records')
        self.sequences, self.annotations, self.versions = [], [], []
        self.catalog_rows, self.replay_rows = [], []
        self.catalog_seen, self.assets_by_id = {}, {}
        self.provenance_sources, self.input_files, self.limitations = [], [], set()
        self.sequence_count = self.clip_count = 0
        self.sequence_dtype = self.annotation_dtype = None
        self.catalog_dtype = self.version_dtype = None

    def append(self, path: Path, *, frozen_entry: dict | None = None) -> None:
        path = Path(path)
        with h5py.File(path, 'r') as source:
            if text(source.attrs['dataset_id']) != self.dataset_id \
                    or 'media' in source or 'replay' not in source \
                    or 'assets' not in source:
                raise ValueError('Only matching replay-only clips may enter a snapshot shard')
            if self.sequence_dtype is None:
                self.sequence_dtype = source['sequences'].dtype
                self.annotation_dtype = source['annotations'].dtype
                self.catalog_dtype = source['labels/catalog'].dtype
                self.version_dtype = source['labels/sequence_versions'].dtype
            offset = len(self.samples)
            count = len(source['samples'])
            self.samples.resize((offset + count, 6))
            for start in range(0, count, 65536):
                self.samples[offset + start:offset + min(count, start + 65536)] = (
                    source['samples'][start:min(count, start + 65536)])
            sequence_rows = source['sequences'][:].copy()
            sequence_rows['sample_start'] += offset
            sequence_rows['sample_stop'] += offset
            self.sequences.append(sequence_rows)
            annotation_rows = source['annotations'][:].copy()
            annotation_rows['sequence_index'] += self.sequence_count
            self.annotations.append(annotation_rows)
            version_rows = source['labels/sequence_versions'][:].copy()
            version_rows['sequence_index'] += self.sequence_count
            self.versions.append(version_rows)
            for row in source['labels/catalog'][:]:
                identity = tuple(text(row[key]) for key in
                                 ('taxonomy_id', 'taxonomy_version', 'code'))
                value = (text(row['name']), bool(row['is_fall']), bool(row['active']))
                if identity in self.catalog_seen:
                    if self.catalog_seen[identity] != value:
                        raise ValueError('Conflicting activity concept in packed clips')
                else:
                    self.catalog_seen[identity] = value
                    self.catalog_rows.append(row)
            for asset in get_json(source, 'assets/index'):
                prior = self.assets_by_id.get(asset['asset_id'])
                if prior is not None and prior != asset:
                    raise ValueError('Conflicting model asset in packed clips')
                self.assets_by_id[asset['asset_id']] = asset
                for item in asset['files']:
                    sha = item['sha256']
                    if sha not in self.handle['assets/blobs']:
                        source.copy(source['assets/blobs/' + sha],
                                    self.handle['assets/blobs'], name=sha)
            self.clip_count += 1
            for row in source['replay/index'][:]:
                record_id = text(row['record_id'])
                packed_id = f'clip{self.clip_count:04d}-{record_id}'
                self.replay_rows.append((int(row['sequence_index']) + self.sequence_count,
                                         packed_id, int(row['sample_zero_replay_time_ns'])))
                source.copy(source['replay/records/' + record_id],
                            self.handle['replay/records'], name=packed_id)
            provenance = get_json(source, 'provenance/metadata')
            for row in provenance['sequence_sources']:
                self.provenance_sources.append({**row,
                    'sequence_index': row['sequence_index'] + self.sequence_count})
            self.input_files.append({
                'role': 'frozen-snapshot-entry',
                'candidate_id': frozen_entry['candidate_id'],
                'version_id': frozen_entry['version_id'],
                'commit_sha256': frozen_entry['commit_sha256'],
                'review_sha256': frozen_entry['review_sha256'],
                'label_sha256': frozen_entry.get('label_sha256'),
            } if frozen_entry is not None else {
                'role': 'validated-clip-delivery', 'sha256': sha256_file(path)})
            self.limitations.update(provenance['limitations'])
            self.sequence_count += len(sequence_rows)

    def finish(self, *, max_shard_bytes: int) -> Path:
        if not self.clip_count:
            raise ValueError('Cannot finish an empty snapshot shard')
        handle = self.handle
        sequences = np.concatenate(self.sequences)
        annotations = np.concatenate(self.annotations)
        handle.create_dataset('sequences', data=sequences, dtype=self.sequence_dtype)
        handle.create_dataset('annotations', data=annotations, dtype=self.annotation_dtype)
        handle.create_dataset('labels/catalog', data=np.asarray(
            self.catalog_rows, dtype=self.catalog_dtype), dtype=self.catalog_dtype)
        handle.create_dataset('labels/sequence_versions',
                              data=np.concatenate(self.versions), dtype=self.version_dtype)
        handle.create_dataset('replay/index', data=table(self.replay_rows, REPLAY))
        put_json(handle, 'assets/index', list(self.assets_by_id.values()))
        put_json(handle, 'provenance/metadata', {
            'producer': {'name': 'imu_motion_simulator.snapshot_pack', 'version': '2.0.0'},
            'input_files': self.input_files,
            'sequence_sources': self.provenance_sources,
            'limitations': sorted(self.limitations),
        })
        handle.attrs.update(imu_schema_version='3.3.0', artifact_profile='imu_dataset',
                            artifact_id=str(uuid5(NAMESPACE_URL,
                                                  self.dataset_id + '/' + self.target.name)),
                            dataset_id=self.dataset_id, sampling_rate_hz=np.float64(25),
                            axis_frame='sensor_local', hdf5_compatibility='1.14',
                            evaluation_role='training_only', feature_columns=json_dump(COLUMNS),
                            sequence_count=np.int64(len(sequences)),
                            sample_count=np.int64(len(self.samples)),
                            annotation_count=np.int64(len(annotations)),
                            logical_content_sha256=logical_content_sha256(
                                self.samples, sequences, annotations,
                                dataset_id=self.dataset_id))
        handle.flush()
        handle.close()
        validate_delivery(self.temporary)
        if self.temporary.stat().st_size > max_shard_bytes:
            raise ValueError('Packed snapshot shard exceeds the size limit')
        if self.target.exists():
            validate_delivery(self.target)
            with h5py.File(self.target) as prior:
                old = get_json(prior, 'provenance/metadata')
                if text(prior.attrs['dataset_id']) != self.dataset_id \
                        or old['input_files'] != self.input_files:
                    raise ValueError('Existing snapshot shard differs from rebuilt input')
            self.temporary.unlink()
        else:
            os.link(self.temporary, self.target)
            self.temporary.unlink()
        return self.target

    def abort(self) -> None:
        if self.handle.id.valid:
            self.handle.close()
        self.temporary.unlink(missing_ok=True)


def _estimate(path: Path) -> tuple[int, dict[str, int]]:
    with h5py.File(path, 'r') as handle:
        blobs = {sha: len(ds) for sha, ds in handle['assets/blobs'].items()}
    return max(0, path.stat().st_size - sum(blobs.values())), blobs


def _pack_group(paths: list[Path], target: Path, dataset_id: str) -> None:
    if target.exists():
        validate_delivery(target)
        with h5py.File(target, 'r') as handle:
            if text(handle.attrs['dataset_id']) != dataset_id:
                raise ValueError('Existing snapshot shard belongs to another request')
            if len(paths) > 1:
                provenance = get_json(handle, 'provenance/metadata')
                expected = [{'role': 'validated-clip-delivery',
                             'sha256': sha256_file(path)} for path in paths]
                if provenance['input_files'] != expected:
                    raise ValueError(
                        'Existing snapshot shard has different source clips')
            elif sha256_file(target) != sha256_file(paths[0]):
                raise ValueError('Existing snapshot shard differs from its source clip')
        return
    if len(paths) == 1:
        temporary = target.with_name('.' + target.name + '.' + uuid4().hex + '.partial')
        try:
            shutil.copyfile(paths[0], temporary)
            validate_delivery(temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        validate_delivery(target)
        return
    sample_temp = target.with_name('.' + target.name + '.' + uuid4().hex + '.samples')
    samples = None
    try:
        with ExitStack() as stack:
            handles = [stack.enter_context(h5py.File(path, 'r')) for path in paths]
            counts = [len(handle['samples']) for handle in handles]
            samples = np.memmap(sample_temp, dtype='<f4', mode='w+',
                                shape=(sum(counts), 6))
            sequences, annotations, versions = [], [], []
            catalog_rows, replay_rows = [], []
            catalog_seen, assets_by_id, blobs, records = {}, {}, {}, {}
            provenance_sources, input_files, limitations = [], [], []
            sample_offset = sequence_offset = 0
            for ordinal, (path, handle, count) in enumerate(zip(paths, handles, counts), 1):
                if text(handle.attrs['imu_schema_version']) != '3.3.0' \
                        or 'media' in handle \
                        or 'replay' not in handle or 'assets' not in handle:
                    raise ValueError('Only replay-only synthetic HDF5 3.3 clips can be packed')
                for start in range(0, count, 65536):
                    stop = min(count, start + 65536)
                    samples[sample_offset + start:sample_offset + stop] = (
                        handle['samples'][start:stop])
                sequence_rows = handle['sequences'][:].copy()
                sequence_rows['sample_start'] += sample_offset
                sequence_rows['sample_stop'] += sample_offset
                sequences.append(sequence_rows)
                annotation_rows = handle['annotations'][:].copy()
                annotation_rows['sequence_index'] += sequence_offset
                annotations.append(annotation_rows)
                version_rows = handle['labels/sequence_versions'][:].copy()
                version_rows['sequence_index'] += sequence_offset
                versions.append(version_rows)
                for row in handle['labels/catalog'][:]:
                    identity = tuple(text(row[key]) for key in (
                        'taxonomy_id', 'taxonomy_version', 'code'))
                    value = (text(row['name']), bool(row['is_fall']), bool(row['active']))
                    if identity in catalog_seen:
                        if catalog_seen[identity] != value:
                            raise ValueError('Conflicting activity concept in packed clips')
                    else:
                        catalog_seen[identity] = value
                        catalog_rows.append(row)
                for asset in get_json(handle, 'assets/index'):
                    previous = assets_by_id.get(asset['asset_id'])
                    if previous is not None and previous != asset:
                        raise ValueError('Conflicting embedded model asset')
                    assets_by_id[asset['asset_id']] = asset
                    for item in asset['files']:
                        sha = item['sha256']
                        if sha not in blobs:
                            blobs[sha] = handle['assets/blobs/' + sha][:].tobytes()
                for row in handle['replay/index'][:]:
                    record_id = text(row['record_id'])
                    packed_id = f'clip{ordinal:04d}-{record_id}'
                    replay_rows.append((int(row['sequence_index']) + sequence_offset,
                                        packed_id, int(row['sample_zero_replay_time_ns'])))
                    record = handle['replay/records/' + record_id]
                    records[packed_id] = {
                        'metadata': get_json(record, 'metadata'),
                        'arrays': {name: record[name] for name in record
                                   if name != 'metadata'},
                    }
                provenance = get_json(handle, 'provenance/metadata')
                for source in provenance['sequence_sources']:
                    provenance_sources.append({**source,
                                               'sequence_index': source['sequence_index']
                                               + sequence_offset})
                input_files.append({'role': 'validated-clip-delivery',
                                    'sha256': sha256_file(path)})
                limitations.extend(provenance['limitations'])
                sample_offset += count
                sequence_offset += len(sequence_rows)
            samples.flush()
            first = handles[0]
            write_delivery(
                target, samples=samples,
                sequences=np.concatenate(sequences).astype(first['sequences'].dtype),
                annotations=np.concatenate(annotations).astype(
                    first['annotations'].dtype), dataset_id=dataset_id,
                labels={
                    'catalog': np.asarray(catalog_rows,
                                          dtype=first['labels/catalog'].dtype),
                    'sequence_versions': np.concatenate(versions).astype(
                        first['labels/sequence_versions'].dtype),
                },
                replay={'index': table(replay_rows, REPLAY), 'records': records},
                assets=list(assets_by_id.values()), blobs=blobs,
                provenance={
                    'producer': {'name': 'imu_motion_simulator.snapshot_pack',
                                 'version': '1.0.0'},
                    'input_files': input_files,
                    'sequence_sources': provenance_sources,
                    'limitations': sorted(set(limitations)),
                }, evaluation_role='training_only')
    finally:
        if samples is not None:
            del samples
        sample_temp.unlink(missing_ok=True)


def estimate_clip_delivery(path, *, dataset_id):
    """Validate a source clip and return its body and deduplicated asset sizes."""
    path = Path(path)
    validate_delivery(path)
    with h5py.File(path, 'r') as handle:
        if text(handle.attrs['imu_schema_version']) != '3.3.0' \
                or text(handle.attrs['dataset_id']) != dataset_id \
                or text(handle.attrs['evaluation_role']) != 'training_only' \
                or 'media' in handle or 'replay' not in handle \
                or 'assets' not in handle or 'provenance' not in handle \
                or any(source['source_kind'] != 'synthetic' for source in
                       get_json(handle, 'provenance/metadata')['sequence_sources']):
            raise ValueError('Only matching replay-only synthetic HDF5 3.3 clips can be packed')
    return _estimate(path)


def pack_clip_deliveries(paths, output, *, dataset_id, max_shard_bytes,
                         start_ordinal=1):
    """Return (packed shard path, source clip paths) in deterministic order."""
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError('No clip deliveries to pack')
    if max_shard_bytes <= 0:
        raise ValueError('Shard size limit must be positive')
    if start_ordinal <= 0:
        raise ValueError('Shard ordinal must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups, current, body_bytes, model_blobs = [], [], 0, {}
    target_bytes = int(max_shard_bytes * .85)
    for path in paths:
        body, blobs = estimate_clip_delivery(path, dataset_id=dataset_id)
        projected_blobs = {**model_blobs, **blobs}
        projected = body_bytes + body + sum(projected_blobs.values())
        if current and projected > target_bytes:
            groups.append(current)
            current, body_bytes, model_blobs = [], 0, {}
        current.append(path)
        body_bytes += body
        model_blobs.update(blobs)
    if current:
        groups.append(current)
    packed = []
    for ordinal, group in enumerate(groups, start_ordinal):
        target = output / f'shard-{ordinal:04d}.h5'
        _pack_group(group, target, dataset_id)
        if target.stat().st_size > max_shard_bytes:
            raise ValueError('Packed snapshot shard exceeds 4 GiB')
        packed.append((target, group))
    return packed
