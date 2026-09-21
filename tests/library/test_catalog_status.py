import hashlib
import json

import pytest

from imu_motion_simulator.contracts.common import ContractError
from imu_motion_simulator.library.sources import load_catalog
from imu_motion_simulator.library.status import inspect_library


def test_source_catalog_resolves_relative_library_paths(tmp_path):
    root = tmp_path / "library"
    source = root / "datasets/amass/ACCAD.tar.bz2"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"archive")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({
        "schema": "imu_motion_simulator.source_catalog.v1",
        "catalog_id": "test",
        "sources": [{
            "id": "amass",
            "dataset": "AMASS",
            "kind": "amass",
            "path": "datasets/amass/ACCAD.tar.bz2",
            "source_url": "https://example.org/source",
            "license_url": "https://example.org/license",
            "observed_scope": "test",
        }],
    }))

    catalog_id, rows = load_catalog(catalog, root)

    assert catalog_id == "test"
    assert rows[0]["resolved_path"] == source


def test_source_catalog_rejects_escaping_path(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({
        "schema": "imu_motion_simulator.source_catalog.v1",
        "catalog_id": "test",
        "sources": [{
            "id": "amass",
            "dataset": "AMASS",
            "kind": "amass",
            "path": "../archive.tar.bz2",
            "source_url": "https://example.org/source",
            "license_url": "https://example.org/license",
            "observed_scope": "test",
        }],
    }))

    with pytest.raises(ContractError, match="path escapes package"):
        load_catalog(catalog, tmp_path / "library")


def test_inventory_fast_and_full_checks_observed_receipt(tmp_path):
    root = tmp_path / "library"
    target = root / "datasets/example.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"verified archive")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    receipt = target.with_name(target.name + ".receipt.json")
    receipt.write_text(json.dumps({
        "observed_bytes": target.stat().st_size,
        "observed_sha256": digest,
    }))
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({
        "items": [{
            "id": "example",
            "path": "datasets/example.zip",
            "receipt": "datasets/example.zip.receipt.json",
            "bytes": target.stat().st_size,
            "sha256": digest,
        }],
    }))

    fast = inspect_library(root, inventory=inventory)
    full = inspect_library(root, inventory=inventory, verify_content=True)

    assert fast["counts"] == {"verified": 1}
    assert fast["entries"][0]["check"] == "receipt_and_size"
    assert full["counts"] == {"verified": 1}
    assert full["content_hashes_checked"] is True


def test_inventory_rejects_path_escape(tmp_path):
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"items": [{"id": "bad", "path": "../bad"}]}))

    with pytest.raises(ValueError, match="escapes root"):
        inspect_library(tmp_path / "library", inventory=inventory)


def test_inventory_checks_collection_receipt(tmp_path):
    root = tmp_path / "library"
    dataset = root / "datasets/collection"
    dataset.mkdir(parents=True)
    payload = dataset / "subject.csv"
    payload.write_bytes(b"measurements")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    receipt = dataset / "selected-acquisition.receipt.json"
    receipt.write_text(json.dumps({
        "total_bytes": payload.stat().st_size,
        "files": [{"path": payload.name, "bytes": payload.stat().st_size, "sha256": digest}],
    }))
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({
        "items": [{
            "id": "collection",
            "kind": "collection",
            "path": "datasets/collection",
            "receipt": "datasets/collection/selected-acquisition.receipt.json",
            "bytes": payload.stat().st_size,
        }],
    }))

    report = inspect_library(root, inventory=inventory, verify_content=True)

    assert report["counts"] == {"verified": 1}
    assert report["entries"][0]["files"] == 1
