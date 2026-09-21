"""Compile a hand-maintained local vocabulary into exact, frozen weak rules.

This is an offline preview over immutable published-commit outbox copies. It
does not change the platform's current rules or any production job. The frozen
rules can be applied to a separately generated local core H5 after formal
audit; publishing that H5 is a separate operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from imu_motion_simulator.provisional import default_rules, validate_rules, weak_label
from imu_motion_simulator.publication import _json_bytes

SCHEMA = "imu_motion_simulator.local_activity_presets.v1"
CODE = re.compile(r"^[a-z][a-z0-9_]*$")
RISK = ("fall", "stumble", "trip", "collapse")


def _values(values: object) -> set[str]:
    if not isinstance(values, list) or not values or any(
            not isinstance(item, str) or not item or item != item.strip().lower()
            or "|" in item or ";temporal:" in item
            for item in values):
        raise ValueError("Preset values must be nonempty lowercase strings")
    result = set(values)
    if len(result) != len(values):
        raise ValueError("Duplicate preset values")
    return result


def validate_presets(value: dict) -> dict:
    if (not isinstance(value, dict)
            or set(value) != {"schema", "revision", "babel_families", "babel_codes",
                              "stageii_codes", "member_tokens"}
            or value["schema"] != SCHEMA or type(value["revision"]) is not int
            or value["revision"] < 1 or not isinstance(value["babel_families"], list)
            or not isinstance(value["babel_codes"], dict)
            or not isinstance(value["stageii_codes"], dict)
            or not isinstance(value["member_tokens"], dict)):
        raise ValueError("Invalid local activity preset table")
    aliases, codes = set(), set()
    for family in value["babel_families"]:
        if not isinstance(family, dict) or set(family) != {"code", "aliases", "modifiers"}:
            raise ValueError("Invalid BABEL family")
        code = family["code"]
        if not isinstance(code, str) or not CODE.fullmatch(code) or code in codes \
                or any(risk in code for risk in RISK):
            raise ValueError("Duplicate or unsafe BABEL family code")
        own = _values(family["aliases"])
        modifiers = _values(family["modifiers"]) if family["modifiers"] else set()
        if own & (aliases | modifiers):
            raise ValueError("BABEL family alias overlaps another alias or own modifier")
        aliases.update(own)
        codes.add(code)
    for source, target in value["babel_codes"].items():
        if (not isinstance(source, str) or source != source.strip().lower()
                or not source or not isinstance(target, str) or not CODE.fullmatch(target)
                or any(risk in source or risk in target for risk in RISK)):
            raise ValueError("Invalid BABEL code alias")
    for dataset, mapping in value["stageii_codes"].items():
        if not dataset or not isinstance(mapping, dict):
            raise ValueError("Invalid Stage-II dataset mapping")
        for source, target in mapping.items():
            if (not isinstance(source, str) or source != source.strip().lower()
                    or not source or not isinstance(target, str) or not CODE.fullmatch(target)
                    or any(risk in source or risk in target for risk in RISK)):
                raise ValueError("Invalid Stage-II code mapping")
    token_targets = {}
    for target, tokens in value["member_tokens"].items():
        if not CODE.fullmatch(target) or any(risk in target for risk in RISK):
            raise ValueError("Invalid filename weak target")
        for token in _values(tokens):
            if not re.fullmatch(r"[a-z]+", token) or token in token_targets \
                    or any(risk in token for risk in RISK):
                raise ValueError("Duplicate or unsafe filename token")
            token_targets[token] = target
    return value


def _source_rows(paths: list[Path]) -> list[dict]:
    rows, identities = [], set()
    for root in paths:
        for path in root.glob("*.json"):
            if path.name.endswith(".feed.json"):
                continue
            commit = json.loads(path.read_text(encoding="utf-8"))
            if commit.get("schema") != "imu_motion_simulator.candidate_commit.v1":
                continue
            identity = (commit["candidate_id"], commit["version_id"])
            if identity in identities:
                raise ValueError(f"Duplicate candidate version: {identity}")
            identities.add(identity)
            rows.append(commit)
    return sorted(rows, key=lambda item: (item["candidate_id"], item["version_id"]))


def _has_risk(labels: list[dict]) -> bool:
    return any(risk in str(value).lower()
               for item in labels
               for value in (item.get("code", ""), *(item.get("categories") or []))
               for risk in RISK)


def _candidate_rule(commit: dict, presets: dict) -> tuple[str, str, str | None] | None:
    labels = commit.get("label_candidates") or []
    if _has_risk(labels):
        return None
    rec = [row for row in labels if row.get("kind") == "recording-candidate"
           and row.get("code")]
    temporal = [row for row in labels if row.get("kind") == "temporal-candidate"
                and row.get("code")]
    source = commit["source_dataset"]
    if source in presets["stageii_codes"]:
        own = [row for row in rec if row.get("origin") == "stageii-source-member"]
        if len(own) == 1:
            code = str(own[0]["code"]).lower()
            target = presets["stageii_codes"][source].get(code)
            if target:
                return "stageii-source-member", code, source
        return None
    own = [row for row in rec if row.get("origin") == "babel-1.0"]
    categories = {str(category).lower() for row in own
                  for category in row.get("categories") or []}
    temporal_values = {str(category).lower() for row in temporal
                       if row.get("origin") == "babel-1.0"
                       for category in row.get("categories") or []}
    if not categories:
        codes = {str(row["code"]).lower() for row in own}
        if len(codes) == 1 and not temporal_values:
            code = next(iter(codes))
            if code in presets["babel_codes"]:
                return "babel-1.0", "code:" + code, None
        if codes <= {"none"}:
            stem = Path(str(commit.get("source_member", ""))).stem.lower().removesuffix(
                "_poses")
            if any(risk in stem for risk in (*RISK, "recovery", "push", "accident")):
                return None
            tokens = set(re.findall(r"[a-z]+", stem))
            hits = [(token, target) for target, words in presets["member_tokens"].items()
                    for token in words if token in tokens]
            if len({target for _, target in hits}) == 1 and hits:
                return "source-member-token", min(token for token, _ in hits), source
        return None
    matched = [family for family in presets["babel_families"]
               if categories & set(family["aliases"])
               and categories <= set(family["aliases"]) | set(family["modifiers"])]
    if len(matched) != 1:
        return None
    family = matched[0]
    encoded = "catset:" + "|".join(sorted(categories))
    if not temporal_values or temporal_values == categories:
        return "babel-1.0", encoded, None
    # A recording-level label does not claim that the entire clip has one
    # activity. Explicit temporal evidence permits only a `contains_*` code.
    if temporal_values & set(family["aliases"]):
        return ("babel-1.0", encoded + ";temporal:" + "|".join(sorted(temporal_values)),
                None)
    return None


def compile_rules(commits: list[dict], presets: dict) -> tuple[dict, dict]:
    validate_presets(presets)
    base = default_rules()
    base["revision"] = 2
    generated: dict[tuple[str, str, str | None], dict] = {}
    families = presets["babel_families"]
    for commit in commits:
        if weak_label(commit, base)[0] == "weak":
            continue
        identity = _candidate_rule(commit, presets)
        if identity is None:
            continue
        origin, source_value, source_dataset = identity
        if origin == "stageii-source-member":
            target = presets["stageii_codes"][source_dataset][source_value]
        elif origin == "source-member-token":
            target = next(target for target, tokens in presets["member_tokens"].items()
                          if source_value in tokens)
        elif source_value.startswith("code:"):
            target = presets["babel_codes"][source_value[5:]]
        else:
            categories = set(source_value.removeprefix("catset:").split(";temporal:")[0].split("|"))
            family = next(family for family in families
                          if categories & set(family["aliases"])
                          and categories <= set(family["aliases"]) | set(family["modifiers"]))
            target = family["code"]
            if ";temporal:" in source_value:
                target = "contains_" + target
        key = (origin, source_value, source_dataset)
        if key in generated and generated[key]["target_code"] != target:
            raise ValueError(f"Conflicting local preset for {key}")
        digest = hashlib.sha256(_json_bytes(key)).hexdigest()[:12]
        generated[key] = {"rule_id": "local-preset-" + digest,
                          "origin": origin, "source_value": source_value,
                          "source_dataset": source_dataset,
                          "target_code": target, "is_fall": False}
    rules = {**base, "rules": base["rules"] + sorted(
        generated.values(), key=lambda row: row["rule_id"])}
    validate_rules(rules)
    decisions = [(commit["source_dataset"], *weak_label(commit, rules))
                 for commit in commits]
    counts = Counter(state for _, state, _, _ in decisions)
    by_source = Counter(commit["source_dataset"] for commit in commits)
    source_states = Counter((source, state) for source, state, _, _ in decisions)
    targets = Counter(code for _, state, code, _ in decisions if state == "weak")
    report = {"schema": "imu_motion_simulator.local_preset_preview.v1",
              "candidate_count": len(commits), "weak_count": counts["weak"],
              "unresolved_count": counts["unresolved"],
              "base_weak_count": sum(weak_label(commit, default_rules())[0] == "weak"
                                     for commit in commits),
              "compiled_rule_count": len(rules["rules"]),
              "source_counts": dict(sorted(by_source.items())),
              "weak_by_source": {source: source_states[source, "weak"]
                                 for source in sorted(by_source)},
              "unresolved_by_source": {source: source_states[source, "unresolved"]
                                       for source in sorted(by_source)},
              "weak_by_code": dict(sorted(targets.items())),
              "presets_sha256": hashlib.sha256(_json_bytes(presets)).hexdigest(),
              "rules_sha256": hashlib.sha256(_json_bytes(rules)).hexdigest()}
    return rules, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", type=Path, required=True)
    parser.add_argument("--outbox", type=Path, action="append", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    presets = json.loads(args.presets.read_text(encoding="utf-8"))
    commits = _source_rows(args.outbox)
    if len(commits) != args.expected_count:
        raise ValueError(f"Outbox commits {len(commits)} != expected {args.expected_count}")
    rules, report = compile_rules(commits, presets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(rules, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != data:
        raise FileExistsError("Existing frozen rule output differs")
    args.output.write_text(data, encoding="utf-8")
    preview = args.output.with_suffix(".preview.json")
    preview.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
