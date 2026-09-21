"""Build small, explicitly artificial HDF5 conformance files, never training data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from .assets import package_asset
from .common import COORDINATES, json_dump, sha256_file, time_ns
from .core import ANNOTATIONS, SEQUENCES
from .delivery import CATALOG, REPLAY, VERSIONS, table, write_delivery

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / 'pyproject.toml').is_file())


def fixture_inputs(video_path):
    fixture = json.loads((ROOT / 'tests/fixtures/contracts/core.json').read_text())
    core = dict(samples=np.asarray(fixture['samples'], dtype='<f4'),
                sequences=table(fixture['sequences'], SEQUENCES),
                annotations=table(fixture['annotations'], ANNOTATIONS), dataset_id=fixture['dataset_id'])
    labels = dict(catalog=table(fixture['labels']['catalog'], CATALOG),
                  sequence_versions=table(fixture['labels']['sequence_versions'], VERSIONS))
    license = dict(id='MIT', source_url=None, attribution='imu-motion-simulator',
                   distribution_scope='redistributable-test-fixture',
                   evidence_refs=['repository-license'])
    provenance = dict(source_ids=['artificial-contract-fixture'],
                      authoring_tool_versions={}, transformations=[],
                      limitations=['Artificial fixture poses are not physical trajectories.'])
    model_root = ROOT / 'tests/fixtures/assets/artificial-model'
    model, blobs = package_asset(model_root, asset_id='artificial-model/r1',
                                 role='model_package', revision='r1', entrypoint='manifest.json',
                                 license=license, provenance=provenance)
    scene_json = dict(format_version='1.0.0', coordinates=COORDINATES, static_objects=[], moving_objects=[], cameras=[], lighting=[])
    content = json_dump(scene_json).encode()
    sha = hashlib.sha256(content).hexdigest()
    blobs[sha] = content
    notice = (model_root / 'LICENSE').read_bytes()
    notice_sha = hashlib.sha256(notice).hexdigest()
    blobs[notice_sha] = notice
    scene = dict(asset_id='empty-fixture-scene/r1', role='scene', revision='r1', media_type='application/json',
                 entrypoint='scene.json', files=[
                 dict(logical_path='scene.json', sha256=sha, byte_length=len(content),
                      media_type='application/json', blob_path='/assets/blobs/' + sha, external_ref=None),
                 dict(logical_path='LICENSE', sha256=notice_sha, byte_length=len(notice),
                      media_type='text/plain', blob_path='/assets/blobs/' + notice_sha, external_ref=None)],
                 dependencies=[], license=license, provenance=provenance)
    urdf = ET.parse(model_root / 'model.urdf').getroot()
    joints = [j.attrib['name'] for j in urdf.findall('joint') if j.attrib['type'] != 'fixed']
    roots = {link.attrib['name'] for link in urdf.findall('link')} - {j.find('child').attrib['link'] for j in urdf.findall('joint')}
    frames = 481
    q = np.zeros((frames, len(joints)), dtype='<f4')
    q[:, 0] = np.linspace(-0.1, 0.1, frames)
    position = np.zeros((frames, 3), dtype='<f4')
    position[:, 0] = np.linspace(0, 0.1, frames)
    rotation = np.zeros((frames, 4), dtype='<f4'); rotation[:, 0] = 1
    metadata = dict(model_asset_id=model['asset_id'], scene_asset_id=scene['asset_id'], visual_asset_id=None,
                    binding_asset_id=None, root_link=next(iter(roots)), joint_names=joints, joint_units=['rad'] * len(joints),
                    coordinates=COORDINATES, clock=dict(original_rate_hz=960.0, period_s=dict(numerator=1, denominator=960),
                    origin='artificial fixture time zero'), objects=[], source_episode_id='00000000-0000-4000-8000-000000000001')
    replay = dict(index=table([(0, 'fixture-motion', 0), (1, 'fixture-motion', 100_000_000)], REPLAY),
                  records={'fixture-motion': dict(metadata=metadata, arrays=dict(time_ns=time_ns(np.arange(frames), 1, 960),
                            root_position_m=position, root_quaternion_wxyz=rotation, joint_position=q))})
    media = [dict(sequence_index=i, bytes=video_path.read_bytes(), timing=np.array([[0, 0], [500_000_000, 500_000_000]], dtype='<i8'),
                  media_duration_ns=600_000_000, sample_zero_recording_time_ns=0, sample_zero_media_time_ns=0) for i in range(2)]
    return core, labels, media, replay, [model, scene], blobs


def create_video(path):
    subprocess.run(['ffmpeg', '-v', 'error', '-n', '-f', 'lavfi', '-i', 'color=c=blue:s=64x64:r=10:d=0.6',
                    '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], check=True)
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=codec_name,nb_frames',
                             '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def build(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    probe = create_video(directory / 'artificial-blue.mp4')
    core, labels, media, replay, assets, blobs = fixture_inputs(directory / 'artificial-blue.mp4')
    rows = []
    definitions = [('v32-training/' + core['dataset_id'] + '.h5', dict(version='3.2.0')),
                   ('v32-client.h5', dict(version='3.2.0', profile='client_delivery', labels=labels, media=media)),
                   ('v33_imu.h5', dict(labels=labels)), ('v33_video.h5', dict(labels=labels, media=media)),
                   ('v33_replay.h5', dict(labels=labels, replay=replay, assets=assets, blobs=blobs)),
                   ('v33_video_replay.h5', dict(labels=labels, media=media, replay=replay, assets=assets, blobs=blobs)),
                   ('v33_mixed.h5', dict(labels=labels, media=media[:1],
                      replay=dict(index=replay['index'][1:], records=replay['records']), assets=assets, blobs=blobs))]
    for name, options in definitions:
        path = directory / name
        report = write_delivery(path, **core, **options)
        rows.append(dict(path=name, byte_length=path.stat().st_size, sha256=sha256_file(path), **report))
    mixed = dict(core)
    mixed['sequences'] = core['sequences'].copy()
    mixed['sequences']['activity_code'][1] = 'mixed'
    path = directory / 'v33_mixed-aggregate.h5'
    report = write_delivery(path, **mixed, labels=labels)
    rows.append(dict(path=path.name, byte_length=path.stat().st_size, sha256=sha256_file(path), **report))
    result = dict(fixture_only=True, not_scientific_or_training_data=True, ffprobe=probe,
                  files=rows, asset_logical_files=sum(len(a['files']) for a in assets), unique_blobs=len(blobs))
    (directory / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    print(json.dumps(build(parser.parse_args().output), ensure_ascii=False, indent=2))
