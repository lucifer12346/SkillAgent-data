from __future__ import annotations
import re
import sqlite3
import time
from collections import Counter

def extract_sql(text: str) -> str:
    if not text:
        return ""
    blocks = re.findall(r"```sql\s*(.*?)```", text, re.I | re.S)
    if not blocks:
        blocks = re.findall(r"```\s*(.*?)```", text, re.S)
    sql = blocks[-1] if blocks else text
    return sql.strip().split(";", 1)[0].strip()

def _readonly_connection(db_path: str, timeout_seconds: float):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    con.execute("PRAGMA query_only=ON")
    deadline = time.monotonic() + max(0.001, float(timeout_seconds))
    con.set_progress_handler(
        lambda: int(time.monotonic() >= deadline),
        10_000,
    )
    return con


def execute(db_path: str, sql: str, max_rows: int | None = None,
            timeout_seconds: float = 120):
    if not sql.strip():
        return False, [], "empty SQL"
    try:
        con = _readonly_connection(db_path, timeout_seconds)
        try:
            cur = con.execute(sql)
            rows = cur.fetchmany(max_rows) if max_rows else cur.fetchall()
            return True, rows, ""
        finally:
            con.close()
    except Exception as exc:
        detail = "SQLTimeout: execution exceeded " + str(timeout_seconds) + "s" \
            if isinstance(exc, sqlite3.OperationalError) and "interrupted" in str(exc).lower() \
            else f"{type(exc).__name__}: {exc}"
        return False, [], detail


def execute_with_columns(db_path: str, sql: str, max_rows: int | None = None,
                         timeout_seconds: float = 120):
    """Execute read-only SQL and retain cursor column names for tool display."""
    if not sql.strip():
        return False, [], [], "empty SQL"
    try:
        con = _readonly_connection(db_path, timeout_seconds)
        try:
            cur = con.execute(sql)
            columns = [
                str(description[0]) for description in (cur.description or [])
            ]
            rows = cur.fetchmany(max_rows) if max_rows else cur.fetchall()
            return True, columns, rows, ""
        finally:
            con.close()
    except Exception as exc:
        detail = "SQLTimeout: execution exceeded " + str(timeout_seconds) + "s" \
            if isinstance(exc, sqlite3.OperationalError) and "interrupted" in str(exc).lower() \
            else f"{type(exc).__name__}: {exc}"
        return False, [], [], detail

def _norm(rows):
    def cell(x):
        return str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
    return [tuple(cell(x) for x in row) for row in rows]

def evaluate(text: str, gold_sql: str, db_path: str,
             timeout_seconds: float = 120) -> dict:
    pred_sql = extract_sql(text)
    pok, pred, perr = execute(db_path, pred_sql, timeout_seconds=timeout_seconds)
    gok, gold, gerr = execute(db_path, gold_sql, timeout_seconds=timeout_seconds)
    if not gok or not pok:
        return {"hard": 0,  "predicted_sql": pred_sql,
                "error": f"gold: {gerr}" if not gok else perr}
    if set(pred) == set(gold):
        hard = 1
    else:
        hard = 0
    return {"hard": hard, "predicted_sql": pred_sql, "error": ""}

def gold_schema(gold_sql: str, db_path: str, timeout_seconds: float = 120) -> dict:
    """Capture tables/columns actually read by gold SQL using SQLite itself."""
    reads: dict[str, set[str]] = {}
    con = _readonly_connection(db_path, timeout_seconds)

    def authorize(action, arg1, arg2, _db_name, _trigger):
        if action == sqlite3.SQLITE_READ and arg1:
            reads.setdefault(str(arg1), set())
            if arg2:
                reads[str(arg1)].add(str(arg2))
        return sqlite3.SQLITE_OK

    try:
        con.set_authorizer(authorize)
        # Preparing the query is sufficient for SQLite's authorizer to report
        # referenced tables/columns; do not execute a potentially expensive
        # gold query merely to derive schema supervision.
        con.execute("EXPLAIN QUERY PLAN " + gold_sql).fetchall()
    finally:
        con.close()
    return {"tables": [
        {"table": table, "columns": sorted(columns, key=str.lower)}
        for table, columns in sorted(reads.items(), key=lambda pair: pair[0].lower())
    ]}

def evaluate_schema(predicted: dict, gold: dict) -> dict:
    """Report containment plus precision-aware quality for schema linking."""
    pred = {
        str(entry.get("table", "")).lower(): {
            str(column).lower() for column in entry.get("columns", [])
        }
        for entry in predicted.get("tables", [])
        if isinstance(entry, dict) and entry.get("table")
    }
    gold_table_keys = {
        str(entry.get("table", "")).lower()
        for entry in gold.get("tables", [])
        if isinstance(entry, dict) and entry.get("table")
    }
    missing_tables, missing_columns = [], {}
    total_tables = total_columns = found_tables = found_columns = 0
    for entry in gold.get("tables", []):
        table = str(entry.get("table", ""))
        key = table.lower()
        columns = [str(column) for column in entry.get("columns", [])]
        total_tables += 1
        total_columns += len(columns)
        if key not in pred:
            missing_tables.append(table)
            missing_columns[table] = columns
            continue
        found_tables += 1
        missing = [column for column in columns if column.lower() not in pred[key]]
        found_columns += len(columns) - len(missing)
        if missing:
            missing_columns[table] = missing
    predicted_tables = len(pred)
    predicted_columns = sum(len(columns) for columns in pred.values())
    table_recall = found_tables / max(1, total_tables)
    column_recall = found_columns / max(1, total_columns)
    table_precision = (
        len(set(pred) & gold_table_keys) / predicted_tables
        if predicted_tables else 0.0
    )
    column_precision = found_columns / predicted_columns if predicted_columns else 0.0

    def f_beta(precision, recall, beta=2.0):
        beta_squared = beta * beta
        denominator = beta_squared * precision + recall
        return (
            (1 + beta_squared) * precision * recall / denominator
            if denominator else 0.0
        )

    table_f2 = f_beta(table_precision, table_recall)
    if total_columns:
        column_f2 = f_beta(column_precision, column_recall)
        size_ratio = predicted_columns / total_columns
        size_penalty = min(0.30, 0.05 * max(0.0, size_ratio - 1.0))
        quality = max(0.0, 0.4 * table_f2 + 0.6 * column_f2 - size_penalty)
    else:
        # COUNT(*) and similar gold queries may expose table reads without a
        # gold column. In that case column precision is not observable.
        column_precision = 1.0
        column_f2 = 1.0
        size_ratio = 1.0
        size_penalty = 0.0
        quality = table_f2
    hard = int(not missing_tables and not missing_columns)
    return {
        "schema_hard": hard,
        "missing_tables": missing_tables,
        "missing_columns": missing_columns,
        "schema_table_recall": table_recall,
        "schema_column_recall": column_recall,
        "schema_table_precision": table_precision,
        "schema_column_precision": column_precision,
        "schema_table_f2": table_f2,
        "schema_column_f2": column_f2,
        "schema_predicted_tables": predicted_tables,
        "schema_predicted_columns": predicted_columns,
        "schema_gold_tables": total_tables,
        "schema_gold_columns": total_columns,
        "schema_size_ratio": size_ratio,
        "schema_size_penalty": size_penalty,
        "schema_quality_score": quality,
    }
