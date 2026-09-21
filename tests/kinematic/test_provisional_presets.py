"""The local presets increase coverage without inventing non-fall truth."""
from __future__ import annotations

import json
from pathlib import Path

from imu_motion_simulator.local_presets import compile_rules, validate_presets
from imu_motion_simulator.provisional import validate_rules, weak_label

PRESETS = (Path(__file__).resolve().parents[2]
           / "configs/production/local-activity-presets-v1.json")


def _commit(source: str, recording: list[str] | None = None,
            temporal: list[str] | None = None, code: str = "walk") -> dict:
    labels = []
    if recording is not None:
        labels.append({"origin": "babel-1.0", "kind": "recording-candidate",
                       "code": code, "categories": recording})
    if temporal is not None:
        labels.append({"origin": "babel-1.0", "kind": "temporal-candidate",
                       "code": "temporal", "categories": temporal})
    return {"source_dataset": source, "label_candidates": labels}


def test_local_preset_exact_combinations_and_temporal_contains():
    presets = validate_presets(json.loads(PRESETS.read_text()))
    walking_turn = _commit("KIT", ["turn", "walk"],
                           ["stand", "transition", "turn", "walk"])
    jog_run = _commit("ACCAD", ["jog", "run"], code="jog")
    stageii = {"source_dataset": "GRAB", "label_candidates": [
        {"origin": "stageii-source-member", "kind": "recording-candidate",
         "code": "pass", "categories": []}]}
    danger = _commit("KIT", ["walk"], code="stumble while walking")
    no_evidence = _commit("KIT")
    filename = {**_commit("BMLrub", code="none"),
                "source_member": "BMLrub/0006_normal_walk2_poses.npz"}
    unsafe_filename = {**_commit("KIT"),
                       "source_member": "KIT/200/push_recovery_walk02_poses.npz"}
    rules, preview = compile_rules(
        [walking_turn, jog_run, stageii, danger, no_evidence,
         filename, unsafe_filename], presets)
    validate_rules(rules)
    assert rules["revision"] == 2
    assert preview["weak_count"] == 4
    assert preview["unresolved_count"] == 3
    assert weak_label(walking_turn, rules)[1] == "contains_walking"
    assert weak_label(jog_run, rules)[1] == "running"
    assert weak_label(stageii, rules)[1] == "passing_object"
    assert weak_label(danger, rules) == ("unresolved", "", "")
    assert weak_label(no_evidence, rules) == ("unresolved", "", "")
    assert weak_label(filename, rules)[1] == "walking"
    assert weak_label(unsafe_filename, rules) == ("unresolved", "", "")


def test_local_preset_cannot_assign_fall_or_adl_by_default():
    presets = json.loads(PRESETS.read_text())
    presets["stageii_codes"]["GRAB"]["random"] = "nonfall"
    try:
        validate_presets(presets)
    except ValueError as error:
        assert "unsafe" in str(error).lower() or "invalid" in str(error).lower()
    else:
        raise AssertionError("Automatic nonfall truth must be rejected")
