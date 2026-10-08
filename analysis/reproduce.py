"""Verify retained data and recompute the paper's world-level contrasts.

Author: Yanqi Li
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "code/experiments/alpagd_rl/formal_20260810"
CONTROLLERS = (2026081911, 2026081912, 2026081913, 2026081914)
T_975_DF99 = 1.9842169515086827


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_and_unpack(destination):
    manifest = read(ROOT / "data/raw/manifest.json")
    reports = []
    for name, entry in manifest["archives"].items():
        source = ROOT / "data/raw" / entry["file"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Archive checksum differs: {source.name}")
        target = destination / name
        expected = {item["path"]: item["sha256"] for item in entry["members"]}
        actual = set()
        with tarfile.open(source, "r:gz") as archive:
            for member in archive:
                if not member.isfile() or member.name not in expected:
                    raise ValueError(f"Unexpected archive member: {member.name}")
                path = (target / member.name).resolve()
                if not path.is_relative_to(target.resolve()):
                    raise ValueError("Unsafe archive path")
                payload = archive.extractfile(member).read()
                if hashlib.sha256(payload).hexdigest() != expected[member.name]:
                    raise ValueError(f"Record checksum differs: {member.name}")
                json.loads(payload)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                actual.add(member.name)
        if actual != set(expected):
            raise ValueError(f"Incomplete archive: {name}")
        reports.append({"dataset": name, "records": len(actual), "checksums": "pass"})
    return reports


def effects(folder, left, right):
    by_world = defaultdict(list)
    identities = set()
    rows = list((folder / "evaluation").glob("controller_*/stream_*/world_*.json"))
    for path in rows:
        row = read(path)
        stored_identity = tuple(row["identity"])
        identity = stored_identity[1:] if len(stored_identity) == 4 else stored_identity
        if identity in identities:
            raise ValueError("Duplicate paired row")
        identities.add(identity)
        if identity[0] not in CONTROLLERS or identity[1] not in (0, 1):
            raise ValueError("Unexpected controller or stream")
        if row.get("optimizer_steps", 0) != 0 or row.get("graphs_frozen") is not True:
            raise ValueError("Evaluation does not use a frozen graph and controller")
        arms = row["arms"]
        difference = float(arms[left]["R"]) - float(arms[right]["R"])
        if not math.isfinite(difference):
            raise ValueError("Nonfinite paired effect")
        by_world[identity[2]].append(difference)
    if len(rows) != 800 or len(by_world) != 100 or any(len(items) != 8 for items in by_world.values()):
        raise ValueError("Expected 800 rows: eight paired runs for each of 100 worlds")
    values = [fmean(items) for _, items in sorted(by_world.items())]
    mean, se = fmean(values), stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "lower": mean - T_975_DF99 * se,
            "upper": mean + T_975_DF99 * se, "worlds": len(values), "paired_rows": len(rows)}


def order_effects(raw):
    source = read(raw / "study1/fixed_contrast_results/fixed_contrast_result.json")
    by_world = defaultdict(list)
    for row in source["paired_rows"]:
        arms = row["arms"]
        effect = float(arms["Correct"]["R"]) - fmean(float(arms[arm]["R"]) for arm in ("P1", "P2", "P3", "P4"))
        by_world[int(row["world_seed"])].append(effect)
    if len(source["paired_rows"]) != 800 or len(by_world) != 100 or any(len(items) != 8 for items in by_world.values()):
        raise ValueError("Incomplete Study 1 paired records")
    values = [fmean(items) for items in by_world.values()]
    mean, se = fmean(values), stdev(values) / math.sqrt(len(values))
    retained = read(ROOT / "data/summaries/study1.json")["primary"]
    if not math.isclose(mean, retained["estimate_R"], abs_tol=1e-12):
        raise ValueError("Study 1 differs from retained primary estimate")
    return {"study": "Study 1", "contrast": "Canonical-mean_swaps", "mean": mean,
            "lower": mean - T_975_DF99 * se, "upper": mean + T_975_DF99 * se,
            "worlds": 100, "paired_rows": 800}


def compare_csv(actual, retained):
    with actual.open(newline="", encoding="utf-8") as handle:
        first = list(csv.DictReader(handle))
    with retained.open(newline="", encoding="utf-8") as handle:
        second = list(csv.DictReader(handle))
    if len(first) != len(second):
        raise ValueError(f"Row counts differ in {retained.name}")
    for left, right in zip(first, second):
        if left.keys() != right.keys():
            raise ValueError(f"Columns differ in {retained.name}")
        for key in left:
            if left[key] == right[key]:
                continue
            try:
                matches = math.isclose(float(left[key]), float(right[key]), rel_tol=1e-10, abs_tol=1e-12)
            except ValueError:
                matches = False
            if not matches:
                raise ValueError(f"Retained value differs in {retained.name}: {key}")


def recompute_comparisons(raw, output):
    path = FORMAL / "pc3_v19_comparative_analysis/analyze_comparative_results.py"
    spec = importlib.util.spec_from_file_location("comparison_analysis", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    reports = []
    for case in ("p1", "p2"):
        destination = output / "comparisons" / case
        result = module.analyze(
            selector_root=raw / "baseline_selectors" / case,
            fixture_root=raw / "study4" / case.upper(),
            sea_evaluation_root=raw / "baseline_sea" / f"{case}_full",
            sea_artifact_root=raw / "baseline_sea/recovered" / case,
            output_root=destination, fixture=case)
        retained = ROOT / "data/comparisons" / case
        for csv_path in sorted(retained.glob("*.csv")):
            compare_csv(destination / csv_path.name, csv_path)
        reports.append({"corruption": case, "retained_csvs_match": True,
                        "validation": result["validation"]})
    return reports


def projection_check():
    targets = ("place_table", "make_wood_pickaxe", "collect_stone", "make_stone_pickaxe", "collect_iron")
    proposals = []
    for index in (1, 2, 3):
        value = read(ROOT / f"data/llm/parsed_outputs/repair_prior_{index}.json")
        proposals.append({(edge["from_achievement"], edge["to_achievement"]): float(edge["prior_probability"])
                          for edge in value["edge_priors"]})
    protocol = read(FORMAL / "pc3_v14_l5_bridge/protocol.json")
    for left in targets:
        for right in targets:
            if left == right:
                continue
            probability = fmean(proposal.get((left, right), 0.01) for proposal in proposals)
            if not math.isclose(probability, protocol["qL"][f"{left}->{right}"], abs_tol=1e-12):
                raise ValueError("Retained LLM projection differs from Study 5")
    return {"retained_proposals": 3, "ordered_pairs": 20, "projection": "pass", "api_calls": 0}


def annotation_check():
    files = []
    for index in (1, 2):
        with (ROOT / f"data/annotations/annotator_{index}.csv").open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 462:
            raise ValueError("Incomplete pairwise annotation file")
        files.append(rows)
    columns = list(files[0][0])
    key_fields = [name for name in columns if name in {"from_achievement", "to_achievement", "source", "target"}]
    if not key_fields:
        key_fields = columns[:2]
    maps = [{tuple(row[field] for field in key_fields): row["edge_present"] in {"1", "true", "True"} for row in rows}
            for rows in files]
    if maps[0].keys() != maps[1].keys():
        raise ValueError("Annotation pair domains differ")
    counts = [sum(value.values()) for value in maps]
    shared = sum(maps[0][pair] and maps[1][pair] for pair in maps[0])
    if counts != [73, 81] or shared != 21:
        raise ValueError("Annotation agreement differs from retained result")
    return {"pairs_per_file": 462, "positive_edges": counts, "shared_positives": shared}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduction")
    parser.add_argument("--unpack-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    raw = output / "raw"
    data = verify_and_unpack(raw)
    if args.unpack_only:
        print(json.dumps({"archives": data, "raw_root": str(raw)}))
        return
    specifications = (
        ("Study 2 P1", "study2/P1", "U", "F"),
        ("Study 2 P2", "study2/P2", "U", "F"),
        ("Study 3", "study3", "E", "N"),
        ("Study 4 P1", "study4/P1", "ES", "N"),
        ("Study 4 P2", "study4/P2", "ES", "N"),
        ("Study 5 natural", "study5", "L5_N", "L5_0"),
        ("Study 5", "study5", "L5_ES", "L5_N"),
    )
    contrasts = [order_effects(raw)] + [{"study": name, "contrast": f"{left}-{right}", **effects(raw / folder, left, right)}
                 for name, folder, left, right in specifications]
    with (output / "world_level_contrasts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)
    result = {"creator": "Yanqi Li", "archives": data, "world_level_contrasts": contrasts,
              "comparisons": recompute_comparisons(raw, output), "llm_projection": projection_check(),
              "annotations": annotation_check(),
              "environment_rollouts": 0}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "records": sum(item["records"] for item in data),
                      "world_level_contrasts": contrasts, "retained_baseline_csvs_match": True,
                      "llm_projection": "pass", "environment_rollouts": 0}))


if __name__ == "__main__":
    main()
