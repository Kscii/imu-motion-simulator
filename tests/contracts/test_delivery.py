"""Conformance, independent compatibility and corruption rejection."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

from imu_motion_simulator.contracts.common import ContractError, get_json, json_dump, json_load, sha256_file
from imu_motion_simulator.contracts.core import logical_content_sha256
from imu_motion_simulator.contracts.delivery import (DeliveryReader, migrate_v32, validate_delivery, write_delivery)
from imu_motion_simulator.contracts.internal import (read_source, source_metadata, validate_source,
                                           write_source)

ROOT = Path(__file__).resolve().parents[2]
from imu_motion_simulator.contracts.fixtures import build, fixture_inputs


@pytest.fixture(scope='module')
def fixture_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp('contracts') / 'files'
    build(path)
    return path


def mutated(fixture_dir, tmp_path, name='v33_video_replay.h5'):
    path = tmp_path / name
    shutil.copyfile(fixture_dir / name, path)
    return path


def replace_json(handle, path, value):
    handle[path][()] = json_dump(value)


def test_all_real_files_and_core_equality(fixture_dir):
    manifest = json.loads((fixture_dir / 'manifest.json').read_text())
    expected = json.loads((ROOT / 'tests/fixtures/contracts/core.json').read_text())['expected_logical_content_sha256']
    with h5py.File(fixture_dir / manifest['files'][0]['path']) as base:
        for item in manifest['files']:
            path = fixture_dir / item['path']
            assert validate_delivery(path)['validation'] == 'full'
            assert sha256_file(path) == item['sha256']
            if 'aggregate' in item['path']:
                continue
            assert item['logical_content_sha256'] == expected
            with h5py.File(path) as handle:
                for name in ('samples', 'sequences', 'annotations'):
                    assert handle[name].dtype == base[name].dtype
                    assert np.array_equal(handle[name][:], base[name][:])
    assert manifest['unique_blobs'] < manifest['asset_logical_files']


def test_independent_collector_hash(fixture_dir):
    root = os.environ.get('IMU_COLLECTOR_REPO')
    if not root:
        pytest.skip('set IMU_COLLECTOR_REPO for the independent implementation check')
    path = Path(root) / 'src/imu_data_collector/logical_content.py'
    if not path.exists():
        pytest.skip('independent sibling implementation unavailable')
    spec = importlib.util.spec_from_file_location('independent_logical_content', path)
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    for file in fixture_dir.rglob('*.h5'):
        with h5py.File(file) as handle:
            actual = module.logical_content_sha256(handle['samples'][:], handle['sequences'][:], handle['annotations'][:],
                        dataset_id=handle.attrs['dataset_id'], sampling_rate_hz=25.0)
            assert actual == handle.attrs['logical_content_sha256']


def test_independent_benchmark_reader(fixture_dir):
    root = os.environ.get('IMU_BENCHMARK_REPO')
    if not root:
        pytest.skip('set IMU_BENCHMARK_REPO for the independent reader check')
    repo = Path(root)
    if not (repo / '.venv/bin/python').exists():
        pytest.skip('independent benchmark environment unavailable')
    program = '''import json,sys
from pathlib import Path
from imu_benchmark.dataset import validate_hdf5_file
p=Path(sys.argv[1])
a=validate_hdf5_file(p/'v32-training/ims-contract-fixture.h5')
b=validate_hdf5_file(p/'v32-client.h5',allowed_profiles=('client_delivery',))
try: validate_hdf5_file(p/'v33_imu.h5')
except (ValueError,RuntimeError): pass
else: raise AssertionError('old reader must reject 3.3')
print('v32 training/client accepted; v33 rejected by old reader')
'''
    result = subprocess.run([str(repo / '.venv/bin/python'), '-B', '-c', program, str(fixture_dir)], cwd=repo,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_range_reads_and_mapping(fixture_dir):
    with DeliveryReader(fixture_dir / 'v33_replay.h5', full=True) as reader:
        assert reader.samples(1, 1, 3).shape == (2, 6)
        result = reader.replay_window('fixture-motion', 101_000_000, 105_000_000)
        assert result['row_stop'] - result['row_start'] < 10
        assert result['arrays']['time_ns'][0] <= 101_000_000
        assert result['arrays']['time_ns'][-1] >= 105_000_000
        assert not result['out_of_range']
        assert reader.replay_window('fixture-motion', -1, 501_000_000)['out_of_range']
        with pytest.raises(ContractError):
            reader.samples(0, -1, 5)
    with DeliveryReader(fixture_dir / 'v33_imu.h5') as reader:
        assert not reader.description['capabilities']['replay']


@pytest.mark.parametrize('corruption', ['version', 'extra_root', 'core_field', 'nan', 'hash', 'taxonomy', 'missing_asset',
 'asset_hash', 'path_escape', 'asset_revision', 'asset_cycle', 'video_offset', 'video_hash', 'media_zero',
 'replay_coverage', 'replay_clock', 'quaternion', 'joint_order', 'external_link', 'empty_media'])
def test_corruption_rejected(fixture_dir, tmp_path, corruption):
    path = mutated(fixture_dir, tmp_path)
    with h5py.File(path, 'r+') as h:
        if corruption == 'version': h.attrs['imu_schema_version'] = '3.4.0'
        elif corruption == 'extra_root': h.create_group('unregistered')
        elif corruption == 'core_field':
            rows=h['sequences'][:];del h['sequences'];h.create_dataset('sequences',data=rows[['sample_start','sample_stop']])
        elif corruption == 'nan': h['samples'][0, 0] = np.nan
        elif corruption == 'hash': h['samples'][0, 0] = 42
        elif corruption == 'taxonomy':
            rows=h['labels/catalog'][:];rows['code'][0]='missing';h['labels/catalog'][:]=rows
        elif corruption == 'missing_asset': del h['assets/blobs'][next(iter(h['assets/blobs']))]
        elif corruption == 'asset_hash': h['assets/blobs'][next(iter(h['assets/blobs']))][0] ^= 1
        elif corruption in ('path_escape', 'asset_revision', 'asset_cycle'):
            assets=get_json(h,'assets/index')
            if corruption=='path_escape': assets[0]['files'][0]['logical_path']='../unsafe'
            elif corruption=='asset_revision': assets[0]['revision']='r999'
            else: assets[0]['dependencies']=[assets[0]['asset_id']]
            replace_json(h,'assets/index',assets)
        elif corruption in ('video_offset','video_hash','media_zero'):
            rows=h['media/index'][:]
            if corruption=='video_offset':rows['file_offset'][0]+=1
            elif corruption=='video_hash':rows['sha256'][0]='0'*64
            else:rows['sample_zero_media_time_ns'][0]=100
            h['media/index'][:]=rows
        elif corruption=='replay_coverage':
            rows=h['replay/index'][:];rows['sample_zero_replay_time_ns'][0]=-1;h['replay/index'][:]=rows
        elif corruption=='replay_clock': h['replay/records/fixture-motion/time_ns'][12]+=100
        elif corruption=='quaternion': h['replay/records/fixture-motion/root_quaternion_wxyz'][0]=[0,0,0,0]
        elif corruption=='joint_order':
            m=get_json(h,'replay/records/fixture-motion/metadata');m['joint_names'].reverse();replace_json(h,'replay/records/fixture-motion/metadata',m)
        elif corruption=='external_link': h['unsafe']=h5py.ExternalLink('/tmp/no-file.h5','/')
        elif corruption=='empty_media':
            dtype=h['media/index'].dtype;del h['media'];h.create_dataset('media/index',shape=(0,),dtype=dtype);h.create_group('media/videos');h.create_group('media/timing')
    with pytest.raises(ContractError): validate_delivery(path)


def test_core_reader_does_not_read_corrupt_attachment(fixture_dir,tmp_path):
    path=mutated(fixture_dir,tmp_path)
    with h5py.File(path,'r+') as h:h['media/videos/0'][20]^=1
    assert validate_delivery(path,full=False)['validation']=='core'
    with pytest.raises(ContractError):validate_delivery(path)


def test_immutable_and_failed_write(fixture_dir,tmp_path):
    core,labels,*_=fixture_inputs(fixture_dir/'artificial-blue.mp4')
    path=tmp_path/'new.h5'
    write_delivery(path,**core,labels=labels)
    before=sha256_file(path)
    with pytest.raises(ContractError):write_delivery(path,**core,labels=labels)
    assert sha256_file(path)==before
    core['samples']=core['samples'].astype('f8')
    with pytest.raises(ContractError):write_delivery(tmp_path/'bad.h5',**core,labels=labels)
    assert not (tmp_path/'bad.h5').exists()
    assert not list(tmp_path.glob('*.partial')) and not list(tmp_path.glob('.*.partial'))


@pytest.mark.parametrize('value',['{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}','{"x":1e999}','{bad'])
def test_json_rejects_ambiguity(value):
    with pytest.raises(ContractError):json_load(value)


def test_source_roundtrip_and_unimplemented_kind(tmp_path):
    meta=source_metadata(producer=dict(name='test',version='1',code_sha256='a'*64),
          kind_metadata=dict(source_id='example',dataset_id='example',files=[dict(logical_path='raw.bvh',sha256='b'*64,
           byte_length=1,media_type='text/plain',source_uri='https://example.org/raw.bvh')],
           license={'status':'fixture'},access_scope='test-only',audit={'numeric':False},missing_information=['units','axes']),
          provenance=dict(source_refs=[],group_keys=[],evidence=[],limitations=['Artificial metadata test']))
    path=tmp_path/'source.h5';write_source(path,meta)
    desc,actual=read_source(path)
    assert actual==meta and desc['execution_ready'] is False
    with h5py.File(path,'r+') as h:h.attrs['artifact_kind']='episode'
    with pytest.raises(ContractError,match='kind mismatch'):validate_source(path)


@pytest.mark.parametrize('kind',['training','client'])
def test_migration_preserves_core(fixture_dir,tmp_path,kind):
    core,labels,*_=fixture_inputs(fixture_dir/'artificial-blue.mp4')
    src=fixture_dir/('v32-training/ims-contract-fixture.h5' if kind=='training' else 'v32-client.h5')
    before=sha256_file(src)
    provenance=dict(producer='conformance-test',input_files=[{'sha256':before}],
        sequence_sources=[dict(sequence_index=i,source_kind='real',group_keys=[],refs={}) for i in range(2)],
        limitations=['Artificial contract fixture only; no real participant or simulated trajectory.'])
    dst=tmp_path/'renamed-migration.h5'
    migrate_v32(src,dst,labels=labels if kind=='training' else None,provenance=provenance)
    assert sha256_file(src)==before
    with h5py.File(src) as a,h5py.File(dst) as b:
        assert a.attrs['logical_content_sha256']==b.attrs['logical_content_sha256']
        for name in ('samples','sequences','annotations'):assert np.array_equal(a[name][:],b[name][:])
        if kind=='client':
            for i in range(2):assert np.array_equal(a[f'media/videos/{i}'][:],b[f'media/videos/{i}'][:])


def test_moving_object_and_sparse_attachment(fixture_dir,tmp_path):
    core,labels,media,replay,assets,blobs=fixture_inputs(fixture_dir/'artificial-blue.mp4')
    rigid=copy.deepcopy(assets[1]);rigid.update(asset_id='fixture-object/r1',role='rigid_object')
    assets.append(rigid)
    record=replay['records']['fixture-motion']
    record['metadata']['objects']=[dict(object_id='moving-cube',asset_id='fixture-object/r1')]
    poses=np.zeros((481,1,7),dtype='<f4');poses[:,:,3]=1;poses[:,0,1]=np.linspace(0,1,481)
    record['arrays']['object_pose_world']=poses
    path=tmp_path/'objects.h5'
    write_delivery(path,**core,labels=labels,media=media[:1],replay=replay,assets=assets,blobs=blobs)
    with DeliveryReader(path,full=True) as reader:
        window=reader.replay_window('fixture-motion',490_000_000,500_000_000)
        assert window['arrays']['object_pose_world'][-1,0,1]==1


def test_v32_mixed_activity_and_filename_dispatch(fixture_dir,tmp_path):
    core,labels,media,*_=fixture_inputs(fixture_dir/'artificial-blue.mp4')
    core['sequences']['activity_code'][1]='mixed'
    path=tmp_path/'legacy-client.h5'
    write_delivery(path,**core,version='3.2.0',profile='client_delivery',labels=labels,media=media)
    assert validate_delivery(path)['version']=='3.2.0'
    renamed=tmp_path/'renamed.h5'
    shutil.copyfile(fixture_dir/'v32-training/ims-contract-fixture.h5',renamed)
    with pytest.raises(ContractError,match='filename'):validate_delivery(renamed)


def test_hdf5_2_reader_reads_114_files(fixture_dir):
    runtime = os.environ.get('IMU_LEGACY_PYTHON')
    if not runtime:
        pytest.skip('set IMU_LEGACY_PYTHON for the legacy HDF5 runtime check')
    python=Path(runtime)
    if not python.exists():pytest.skip('independent runtime unavailable')
    result=subprocess.run([str(python),'-B','-c',
        'import h5py,sys; h=h5py.File(sys.argv[1]); assert h["samples"].shape==(10,6); print(h5py.version.hdf5_version)',
        str(fixture_dir/'v33_video_replay.h5')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
