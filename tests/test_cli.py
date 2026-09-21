import json
from pathlib import Path

import pytest

from imu_motion_simulator.cli import main
from imu_motion_simulator.docs_check import DOC_FILES, check


def test_document_policy_matches_active_tree():
    report = check(Path(__file__).resolve().parents[1])
    assert report["passed"], report["errors"]
    assert report["documents"] == len(DOC_FILES) + 3


@pytest.mark.parametrize("group", ["human", "simulation", "learning"])
def test_archived_physics_groups_are_not_public_cli_commands(group):
    with pytest.raises(SystemExit):
        main([group])


def test_pipeline_finalize_exit_requires_mechanical_and_automatic_qa(monkeypatch, capsys):
    import imu_motion_simulator.pipeline.run

    monkeypatch.setattr(imu_motion_simulator.pipeline.run, 'run_pipeline',
                        lambda *args, **kwargs: {'stage': 'review',
                                                'automatic_qa_passed': False})
    assert main(['pipeline', 'run', 'plan.json', '--library-root', 'library',
                 '--workspace', 'workspace', '--output', 'output']) == 2
    assert json.loads(capsys.readouterr().out)['automatic_qa_passed'] is False


def test_export_core_watch_stops_on_failed_audit_and_runs_only_after_completion(
        tmp_path, monkeypatch):
    from argparse import Namespace
    from types import SimpleNamespace
    import imu_motion_simulator.cli as cli

    report = tmp_path / 'audit.json'
    args = Namespace(audit_report=report, chain_unit='prod-chain.service',
                     poll_interval_s=5, config=[])
    calls = []
    monkeypatch.setattr(cli, 'production_export_core',
                        lambda value: calls.append(value) or 0)
    report.write_text(json.dumps({'phase': 'failed', 'error': 'cloud mismatch'}))
    assert cli.production_export_core_watch(args) == 3
    assert calls == []
    report.write_text(json.dumps({'phase': 'complete'}))
    assert cli.production_export_core_watch(args) == 0
    assert calls == [args]
    report.write_text(json.dumps({'phase': 'waiting-native'}))
    monkeypatch.setattr(cli.subprocess, 'run',
                        lambda *items, **options: SimpleNamespace(stdout='inactive\n'))
    assert cli.production_export_core_watch(args) == 3
    assert calls == [args]
