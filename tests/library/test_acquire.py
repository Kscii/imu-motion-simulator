import hashlib
import json

from imu_motion_simulator.library.acquire import main


def test_incremental_manifest_preserves_existing_status(tmp_path):
    root = tmp_path / "library"
    logs = root / "logs"
    logs.mkdir(parents=True)
    previous = {
        "existing": {
            "id": "existing",
            "state": "blocked",
            "error": "belongs to another manifest",
        }
    }
    (logs / "acquisition-status.json").write_text(json.dumps(previous))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": []}))

    assert main(["--root", str(root), "--manifest", str(manifest)]) == 0
    assert json.loads((logs / "acquisition-status.json").read_text()) == previous


def test_successful_retry_clears_old_error(tmp_path):
    root = tmp_path / "library"
    logs = root / "logs"
    target = root / "dataset.bin"
    logs.mkdir(parents=True)
    target.write_bytes(b"verified")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_name("dataset.bin.receipt.json").write_text(
        json.dumps({"url": "https://example.org/dataset.bin", "sha256": digest})
    )
    (logs / "acquisition-status.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "id": "dataset",
                    "url": "https://old.example.org/dataset.bin",
                    "state": "blocked",
                    "error": "old failure",
                }
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "id": "dataset",
                        "path": "dataset.bin",
                        "url": "https://example.org/dataset.bin",
                        "license_url": "https://example.org/license"
                    }
                ]
            }
        )
    )

    assert main(["--root", str(root), "--manifest", str(manifest)]) == 0
    state = json.loads((logs / "acquisition-status.json").read_text())["dataset"]
    assert state["state"] == "verified"
    assert state["url"] == "https://example.org/dataset.bin"
    assert "error" not in state


def test_existing_final_must_match_current_manifest(tmp_path):
    root = tmp_path / "library"
    target = root / "dataset.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"previous release")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_name("dataset.bin.receipt.json").write_text(
        json.dumps({"url": "https://example.org/dataset.bin", "sha256": digest})
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "id": "dataset",
                        "path": "dataset.bin",
                        "url": "https://example.org/dataset.bin",
                        "license_url": "https://example.org/license",
                        "bytes": target.stat().st_size,
                        "sha256": hashlib.sha256(b"current release").hexdigest(),
                    }
                ]
            }
        )
    )

    assert main(["--root", str(root), "--manifest", str(manifest)]) == 1
    state = json.loads((root / "logs" / "acquisition-status.json").read_text())["dataset"]
    assert state["state"] == "blocked"
    assert "official sha256" in state["error"]


def test_complete_partial_is_checked_without_network(tmp_path, monkeypatch):
    root = tmp_path / "library"
    partial = root / "dataset.bin.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"complete partial")
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "id": "dataset",
                        "path": "dataset.bin",
                        "url": "https://example.org/dataset.bin",
                        "bytes": partial.stat().st_size,
                        "sha256": digest,
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        "imu_motion_simulator.library.acquire.subprocess.Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network called")),
    )

    assert main(["--root", str(root), "--manifest", str(manifest)]) == 0
    assert (root / "dataset.bin").read_bytes() == b"complete partial"
