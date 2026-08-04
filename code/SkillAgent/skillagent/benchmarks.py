from __future__ import annotations

import csv
import json
import math
import os
from typing import Callable

from .evaluator import evaluate, execute, extract_sql


def _load_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_test_suites(config: dict, bird_items: list[dict]) -> dict[str, dict]:
    """Load configured test suites without changing the BIRD train/val data."""
    suites = {}
    for name, spec in config.get("test_sets", {}).items():
        kind = spec.get("type", name)
        if kind == "bird":
            items = [{**row, "dataset": "bird"} for row in bird_items]
        elif kind == "ehrsql":
            root = spec["data_root"]
            items = []
            for index, row in enumerate(_load_json(spec["data_file"])):
                db_id = row["db_id"]
                items.append({
                    "id": row.get("id", f"ehrsql_{index:05d}"),
                    "dataset": "ehrsql",
                    "db_id": db_id,
                    "db_path": row.get("db_path") or os.path.abspath(
                        os.path.join(root, "database", db_id, f"{db_id}.sqlite")
                    ),
                    "question": row["question"],
                    "evidence": row.get("evidence", ""),
                    "gold_sql": row.get("gold_sql") or row.get("query"),
                })
        elif kind in {"spider2", "spider2-lite"}:
            root = spec["data_root"]
            items = []
            for row in _load_json(spec["data_file"]):
                db_id = row["db_id"]
                items.append({
                    "id": row["instance_id"],
                    "dataset": "spider2.0",
                    "db_id": db_id,
                    "db_path": os.path.abspath(
                        os.path.join(root, "databases", db_id, f"{db_id}.sqlite")
                    ),
                    "question": row["question"],
                    "evidence": row.get("external_knowledge", ""),
                })
        else:
            raise ValueError(f"unknown test-set type {kind!r} for {name!r}")
        limit = spec.get("limit")
        suites[name] = {"type": kind, "items": items if limit is None else items[:int(limit)], "config": spec}
    return suites


def load_test_suites_from_dir(root: str) -> dict[str, dict]:
    """Load the exact test manifests materialized for one training run."""
    manifest_path = os.path.join(root, "manifest.json")
    manifest = _load_json(manifest_path)
    required = {"bird", "ehrsql", "spider2.0"}
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"test-set manifest is missing {sorted(missing)}: {manifest_path}")
    suites = {}
    for name in ("bird", "ehrsql", "spider2.0"):
        entry = manifest[name]
        items_path = os.path.join(root, name, "items.json")
        items = _load_json(items_path)
        expected = int(entry.get("samples", len(items)))
        if len(items) != expected:
            raise ValueError(
                f"test-set size mismatch for {name}: manifest={expected}, items={len(items)}"
            )
        suites[name] = {
            "type": entry["type"],
            "items": items,
            "config": entry.get("config", {}),
        }
    return suites


def _metadata(path: str) -> dict:
    result = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[row["instance_id"]] = row
    return result


def _cell(value: str):
    if value == "":
        return 0
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return value


def _read_csv(path: str) -> tuple[list[str], list[list]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [], []
    return rows[0], [[_cell(value) for value in row] for row in rows[1:]]


def _vectors_match(left, right, ignore_order=False, tolerance=1e-2):
    def normalize(value):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return 0
        return value
    left = [normalize(value) for value in left]
    right = [normalize(value) for value in right]
    if ignore_order:
        key = lambda value: (value is None, str(value), isinstance(value, (int, float)))
        left, right = sorted(left, key=key), sorted(right, key=key)
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(float(a), float(b), abs_tol=tolerance):
                return False
        elif a != b:
            return False
    return True


def _columns(rows: list[list]) -> list[list]:
    width = max((len(row) for row in rows), default=0)
    return [[row[index] if index < len(row) else None for row in rows] for index in range(width)]


def _table_match(pred_rows, gold_rows, condition_cols=None, ignore_order=False):
    gold_columns = _columns(gold_rows)
    if condition_cols:
        gold_columns = [gold_columns[index] for index in condition_cols if index < len(gold_columns)]
    pred_columns = _columns(pred_rows)
    return int(all(
        any(_vectors_match(gold, pred, ignore_order) for pred in pred_columns)
        for gold in gold_columns
    ))


def _gold_paths(instance_id: str, root: str):
    exact = os.path.join(root, f"{instance_id}.csv")
    if os.path.exists(exact):
        return [exact]
    prefix = instance_id + "_"
    return sorted(os.path.join(root, name) for name in os.listdir(root)
                  if name.startswith(prefix) and name.endswith(".csv"))


def spider2_evaluator(spec: dict, timeout_seconds: float = 120) -> Callable[[str, dict], dict]:
    metadata = _metadata(spec["eval_metadata"])
    gold_dir = spec["gold_dir"]

    def evaluate_one(response: str, item: dict) -> dict:
        sql = extract_sql(response)
        ok, rows, error = execute(
            item["db_path"], sql, timeout_seconds=timeout_seconds
        )
        if not ok:
            return {"hard": 0, "predicted_sql": sql, "error": error}
        paths = _gold_paths(item["id"], gold_dir)
        if not paths:
            return {"hard": 0, "predicted_sql": sql, "error": "gold CSV not found"}
        standard = metadata.get(item["id"], {})
        conditions = standard.get("condition_cols", [])
        orders = standard.get("ignore_order", False)
        if len(paths) > 1:
            if conditions in (None, [], [[]], [None]):
                conditions = [[] for _ in paths]
            elif not all(isinstance(value, list) for value in conditions):
                conditions = [conditions for _ in paths]
            if not isinstance(orders, list):
                orders = [orders for _ in paths]
        else:
            conditions, orders = [conditions], [orders]
        hard = 0
        for index, path in enumerate(paths):
            _, gold_rows = _read_csv(path)
            if _table_match(rows, gold_rows,
                            conditions[min(index, len(conditions) - 1)],
                            orders[min(index, len(orders) - 1)]):
                hard = 1
                break
        return {"hard": hard, "predicted_sql": sql,
                "error": "" if hard else "result mismatch"}

    return evaluate_one


def evaluator_for(suite: dict, timeout_seconds: float = 120) -> Callable[[str, dict], dict]:
    if suite["type"] in {"spider2", "spider2-lite"}:
        return spider2_evaluator(suite["config"], timeout_seconds)
    return lambda response, item: evaluate(
        response, item["gold_sql"], item["db_path"],
        timeout_seconds=timeout_seconds,
    )
