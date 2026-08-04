#!/usr/bin/env python3
"""Filter high-confidence annotation defects from a prepared run snapshot."""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import re
import shutil
import sqlite3
import time


DATASETS = {
    "train": "text2sql_split/train/items.json",
    "val": "text2sql_split/val/items.json",
    "bird": "test_sets/bird/items.json",
    "ehrsql": "test_sets/ehrsql/items.json",
    "spider2.0": "test_sets/spider2.0/items.json",
}

# Manually reviewed, high-confidence semantic conflicts. These IDs are tied to
# this prepared snapshot; every removal is still preserved in the audit log.
CURATED_REMOVALS = {
    "train": {
        "train_00096": "question asks for publisher name, but gold also projects book title",
        "train_00259": "evidence grounds the station name as a city value",
        "train_00271": "question/evidence filter high school, but gold filters birth city",
        "train_00276": "question/evidence require posthumous awards, but gold accepts every non-NULL note",
        "train_00294": "evidence date conflicts with the question and gold date",
        "train_00382": "evidence adds an unrelated short-tip condition",
        "train_00414": "gold CASE expression does not implement the requested city/store filter",
        "train_00437": "evidence date conflicts with the question date",
        "train_00471": "evidence month/day conflicts with the question and gold",
        "train_00490": "evidence date conflicts with the question date",
        "train_00537": "evidence year conflicts with the question year",
        "train_00725": "evidence title pattern conflicts with gold matching semantics",
        "train_00757": "evidence adds an unrelated Food attribute",
    },
    "bird": {
        "test_00104": "evidence language literal conflicts with gold/database spelling",
        "test_00179": "evidence end date is two years after the requested day",
        "test_00496": "evidence adds an unrelated Agility predicate",
        "test_00501": "gold OR/AND precedence does not enforce admission for both normal RNP encodings",
        "test_00634": "evidence year conflicts with the question year",
        "test_01049": "evidence restricts January while the question requests all of 2014",
        "test_01298": "evidence color conflicts with the question color",
        "test_01491": "question contains a malformed start year",
    },
}


def execute_preview(db_path: str, sql: str, timeout_seconds: float):
    try:
        con = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, timeout=30
        )
        con.execute("PRAGMA query_only=ON")
        deadline = time.monotonic() + timeout_seconds
        con.set_progress_handler(
            lambda: int(time.monotonic() >= deadline), 10_000
        )
        try:
            rows = con.execute(sql).fetchmany(2)
            return True, rows, ""
        finally:
            con.close()
    except Exception as exc:
        return False, [], f"{type(exc).__name__}: {exc}"


def evidence_literals(evidence: str) -> set[str]:
    """Extract quoted SQL literals only from explicit `refers to` mappings."""
    values = set()
    for clause in re.split(r"[;\n]", evidence or ""):
        match = re.search(r"\brefers?\s+to\b(.*)", clause, re.I)
        if not match:
            continue
        rhs = match.group(1)
        # Do not treat every quoted phrase in explanatory prose as a SQL
        # literal. Require an adjacent SQL comparison operator.
        patterns = [
            r"(?:=|!=|<>|>=|<=|>|<|\bLIKE\b)\s*'([^']+)'",
            r"\bBETWEEN\s*'([^']+)'\s+AND\s*'([^']+)'",
            r"\bIN\s*\(([^)]*)\)",
        ]
        for pattern in patterns:
            for match_values in re.findall(pattern, rhs, re.I):
                groups = (
                    match_values
                    if isinstance(match_values, tuple)
                    else (match_values,)
                )
                for group in groups:
                    quoted = re.findall(r"'([^']+)'", group)
                    if quoted:
                        values.update(value.strip() for value in quoted)
                    elif group.strip() and pattern != patterns[2]:
                        values.add(group.strip())
    return values


def audit_item(item: dict, dataset: str,
               timeout_seconds: float) -> tuple[list[dict], list[dict]]:
    sql = str(item.get("gold_sql") or "").strip()
    removal_reasons = []
    review_reasons = []
    curated_detail = CURATED_REMOVALS.get(dataset, {}).get(
        str(item.get("id"))
    )
    if curated_detail:
        removal_reasons.append({
            "code": "curated_semantic_conflict",
            "detail": curated_detail,
        })
    if not sql:
        return removal_reasons, review_reasons
    ok, rows, error = execute_preview(
        str(item.get("db_path") or ""), sql, timeout_seconds
    )
    if not ok:
        target = (
            review_reasons if "interrupted" in error.lower()
            else removal_reasons
        )
        target.append({
            "code": (
                "gold_execution_timeout"
                if target is review_reasons else "gold_execution_error"
            ),
            "detail": error,
        })
        return removal_reasons, review_reasons
    if not rows:
        removal_reasons.append({
            "code": "gold_empty_result",
            "detail": "gold SQL returns zero rows",
        })
    elif all(
        all(value is None for value in row)
        for row in rows
    ):
        removal_reasons.append({
            "code": "gold_all_null_result",
            "detail": "gold SQL result is entirely NULL",
        })

    # SQLite does not support strftime('%y', ...); it returns NULL rather than
    # raising, which can make an unrelated wrong query pass by also returning
    # NULL. Four-digit years require %Y.
    formats = re.findall(
        r"strftime\s*\(\s*['\"]([^'\"]+)", sql, re.I
    )
    if any("%y" in date_format for date_format in formats):
        removal_reasons.append({
            "code": "unsupported_sqlite_strftime_y",
            "detail": "SQLite strftime uses %Y for a four-digit year, not %y",
        })

    lower_sql = sql.lower()
    missing_literals = sorted(
        value for value in evidence_literals(str(item.get("evidence") or ""))
        if value.lower() not in lower_sql
    )
    if missing_literals:
        review_reasons.append({
            "code": "evidence_gold_literal_conflict",
            "detail": "explicit evidence literal(s) absent from gold SQL",
            "missing_literals": missing_literals,
        })
    return removal_reasons, review_reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    parser.add_argument(
        "--output-data-dir",
        help="Write filtered snapshot here; omit for dry-run audit.",
    )
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    source = os.path.join(os.path.abspath(args.run_root), "data")
    report = {
        "source_data_dir": source,
        "generated_at": datetime.datetime.now().astimezone().isoformat(),
        "mode": "write" if args.output_data_dir else "dry-run",
        "datasets": {},
    }
    filtered_by_dataset = {}
    removed_records = []
    review_records = []

    for dataset, relative_path in DATASETS.items():
        path = os.path.join(source, relative_path)
        with open(path, encoding="utf-8") as handle:
            items = json.load(handle)
        kept = []
        counts = collections.Counter()
        review_counts = collections.Counter()
        examples = []
        for item in items:
            reasons, review_reasons = audit_item(
                item, dataset, args.timeout
            )
            if review_reasons:
                for reason in review_reasons:
                    review_counts[reason["code"]] += 1
                review_records.append({
                    "dataset": dataset,
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "evidence": item.get("evidence"),
                    "gold_sql": item.get("gold_sql"),
                    "review_reasons": review_reasons,
                })
            if reasons:
                for reason in reasons:
                    counts[reason["code"]] += 1
                record = {
                    "dataset": dataset,
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "evidence": item.get("evidence"),
                    "gold_sql": item.get("gold_sql"),
                    "reasons": reasons,
                }
                removed_records.append(record)
                if len(examples) < args.max_examples:
                    examples.append(record)
            else:
                kept.append(item)
        filtered_by_dataset[dataset] = kept
        report["datasets"][dataset] = {
            "original": len(items),
            "kept": len(kept),
            "removed": len(items) - len(kept),
            "reason_counts": dict(counts),
            "review_candidate_count": sum(review_counts.values()),
            "review_reason_counts": dict(review_counts),
            "examples": examples,
            "note": (
                "No gold SQL: retained unless another high-confidence rule applies."
                if dataset == "spider2.0" else None
            ),
        }

    report["total_removed"] = len(removed_records)
    report["total_review_candidates"] = len(review_records)

    if args.output_data_dir:
        target = os.path.abspath(args.output_data_dir)
        if os.path.exists(target):
            raise FileExistsError(
                f"refusing to overwrite output data directory: {target}"
            )
        os.makedirs(target)
        for dataset, relative_path in DATASETS.items():
            destination = os.path.join(target, relative_path)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with open(destination, "w", encoding="utf-8") as handle:
                json.dump(
                    filtered_by_dataset[dataset], handle,
                    ensure_ascii=False, indent=2,
                )
        source_manifest = os.path.join(source, "test_sets", "manifest.json")
        with open(source_manifest, encoding="utf-8") as handle:
            manifest = json.load(handle)
        for dataset in ("bird", "ehrsql", "spider2.0"):
            manifest[dataset]["samples"] = len(filtered_by_dataset[dataset])
        manifest_path = os.path.join(target, "test_sets", "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        shutil.copyfile(
            os.path.join(source, "DATA_READY"),
            os.path.join(target, "SOURCE_DATA_READY"),
        )
        with open(
            os.path.join(target, "FILTERED_DATA_READY"),
            "w", encoding="utf-8",
        ) as handle:
            handle.write(
                f"filtered_at={report['generated_at']}\n"
                f"source={source}\n"
                f"removed={len(removed_records)}\n"
            )
        with open(
            os.path.join(target, "removed_annotations.json"),
            "w", encoding="utf-8",
        ) as handle:
            json.dump(
                removed_records, handle, ensure_ascii=False, indent=2
            )
        with open(
            os.path.join(target, "review_candidates.json"),
            "w", encoding="utf-8",
        ) as handle:
            json.dump(
                review_records, handle, ensure_ascii=False, indent=2
            )
        report["output_data_dir"] = target

    report_path = (
        os.path.join(os.path.abspath(args.output_data_dir), "filter_report.json")
        if args.output_data_dir else None
    )
    if report_path:
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
