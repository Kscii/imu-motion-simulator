"""Hash-addressed embedded assets shared by internal and delivery contracts."""
from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .common import (digest_dataset, get_json, json_load, put_json, relative_path,
                     require, sha_string)


def package_asset(directory, *, asset_id, role, revision, entrypoint, license, provenance):
    directory = Path(directory)
    files, blobs = [], {}
    for path in sorted(directory.rglob('*')):
        require(not path.is_symlink(), 'symlink in asset package')
        if not path.is_file():
            continue
        logical = path.relative_to(directory).as_posix()
        relative_path(logical)
        content = path.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        blobs[sha] = content
        files.append(dict(logical_path=logical, sha256=sha, byte_length=len(content),
                          media_type=mimetypes.guess_type(logical)[0] or 'application/octet-stream',
                          blob_path='/assets/blobs/' + sha, external_ref=None))
    return dict(asset_id=asset_id, role=role, revision=revision, media_type='application/json',
                entrypoint=entrypoint, files=files, dependencies=[], license=license,
                provenance=provenance), blobs


def write_assets(handle, assets, blobs):
    put_json(handle, 'assets/index', assets)
    group = handle.require_group('assets/blobs')
    for sha, data in sorted(blobs.items()):
        require(sha_string(sha) and hashlib.sha256(data).hexdigest() == sha, 'asset input hash')
        group.create_dataset(sha, data=np.frombuffer(data, dtype='u1'))


def validate_assets(handle):
    """This initial implementation supports embedded assets only, explicitly."""
    require(set(handle['assets']) == {'index', 'blobs'}, 'asset root objects')
    assets = get_json(handle, 'assets/index')
    require(isinstance(assets, list) and assets, 'empty asset index')
    registry, used = {}, set()
    required = {'asset_id', 'role', 'revision', 'media_type', 'entrypoint', 'files',
                'dependencies', 'license', 'provenance'}
    for asset in assets:
        require(isinstance(asset, dict) and set(asset) == required, 'asset descriptor fields')
        aid = asset['asset_id']
        require(isinstance(aid, str) and aid and aid not in registry, 'duplicate/empty asset identity')
        require(asset['revision'] and asset['revision'] != 'latest', 'asset revision')
        require(asset['role'] in ('model_package', 'smplh_model', 'visual', 'binding',
                                  'scene', 'rigid_object', 'dependency'), 'asset role')
        require(set(asset['license']) == {'id', 'source_url', 'attribution', 'distribution_scope', 'evidence_refs'}, 'asset license fields')
        require(asset['license']['distribution_scope'] and asset['license']['evidence_refs'], 'asset permission evidence missing')
        require(set(asset['provenance']) == {'source_ids', 'authoring_tool_versions', 'transformations', 'limitations'}, 'asset provenance fields')
        require(isinstance(asset['dependencies'], list) and len(set(asset['dependencies'])) == len(asset['dependencies']), 'asset dependencies')
        files = {}
        for item in asset['files']:
            require(set(item) == {'logical_path', 'sha256', 'byte_length', 'media_type', 'blob_path', 'external_ref'}, 'asset file fields')
            relative_path(item['logical_path'])
            require(item['logical_path'] not in files, 'duplicate asset path')
            require(item['external_ref'] is None, 'external assets not supported by this reader')
            sha = item['sha256']
            require(sha_string(sha) and item['blob_path'] == '/assets/blobs/' + sha, 'asset blob identity')
            require(item['blob_path'] in handle, 'missing asset blob')
            dataset = handle[item['blob_path']]
            require(isinstance(dataset, h5py.Dataset) and dataset.ndim == 1 and dataset.dtype == np.dtype('u1'), 'asset byte dtype')
            require(type(item['byte_length']) is int and len(dataset) == item['byte_length'], 'asset byte length')
            if sha not in used:
                require(digest_dataset(dataset) == sha, 'asset blob hash mismatch')
            used.add(sha)
            files[item['logical_path']] = item
        require(asset['entrypoint'] in files, 'asset entrypoint missing')
        registry[aid] = (asset, files)
    require(set(handle['assets/blobs']) == used, 'unreferenced asset blobs')
    active, done = set(), set()

    def visit(aid):
        require(aid in registry and aid not in active, 'missing/cyclic asset dependency')
        if aid in done:
            return
        active.add(aid)
        for dependency in registry[aid][0]['dependencies']:
            visit(dependency)
        active.remove(aid)
        done.add(aid)
    for aid in registry:
        visit(aid)
    # Semantic closure for the initial, frozen human package and JSON scenes.
    for aid, (asset, files) in registry.items():
        def read(path):
            require(path in files, 'missing logical asset file: ' + path)
            return handle[files[path]['blob_path']][:].tobytes()
        if asset['role'] == 'model_package':
            manifest = json_load(read(asset['entrypoint']).decode())
            require(manifest['format_version'] == '1.0.0', 'unsupported model format')
            require(manifest['model_revision'] == asset['revision'], 'model revision mismatch')
            for row in manifest['files']:
                require(row['path'] in files and files[row['path']]['sha256'] == row['sha256'], 'model manifest hash mismatch')
            urdf = ET.fromstring(read('model.urdf'))
            for mesh in urdf.findall('.//mesh'):
                name = mesh.attrib['filename']
                relative_path(name)
                require(name in files, 'URDF mesh dependency missing')
        elif asset['role'] == 'scene':
            scene = json_load(read(asset['entrypoint']).decode())
            require(scene['format_version'] == '1.0.0', 'unsupported scene format')
            for obj in scene['static_objects'] + scene['moving_objects']:
                require(obj['asset_id'] in asset['dependencies'], 'scene object dependency not declared')
    return registry
