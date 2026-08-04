from __future__ import annotations
from .evaluator import execute, execute_with_columns

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": (
                "Execute read-only SQLite SELECT/WITH/PRAGMA or EXPLAIN QUERY PLAN SQL. "
                "Use sqlite_master, pragma_table_info(...), and "
                "pragma_foreign_key_list(...) to inspect schema; use ordinary SELECT "
                "queries to inspect representative values or test candidate SQL."
            ),
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
    },
]

SCHEMA_LINK_TOOLS = TOOLS
SQL_TOOLS = TOOLS

def _validate_sql(db_path: str, sql: str, timeout_seconds: float = 120) -> str:
    try:
        ok, _, error = execute(
            db_path, "EXPLAIN QUERY PLAN " + sql,
            timeout_seconds=timeout_seconds,
        )
        if not ok:
            return "INVALID: " + error
        return "VALID: SQLite parsed the query and resolved its referenced schema"
    except Exception as exc:
        return f"INVALID: {type(exc).__name__}: {exc}"


def _table_cell(value) -> str:
    text = "NULL" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("\r", "\\r").replace(
        "\n", "\\n"
    ).replace("|", "\\|")


def _format_table(columns: list[str], rows: list[tuple],
                  output_limit: int = 4000) -> str:
    lines = [f"{len(rows)} row(s)"]
    if columns:
        lines.extend([
            "| " + " | ".join(_table_cell(column) for column in columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ])
    omitted = 0
    for index, row in enumerate(rows):
        line = "| " + " | ".join(_table_cell(value) for value in row) + " |"
        candidate = "\n".join(lines + [line])
        if len(candidate) > output_limit:
            omitted = len(rows) - index
            break
        lines.append(line)
    if omitted:
        lines.append(f"... {omitted} additional row(s) omitted by output limit")
    return "\n".join(lines)


def dispatch(db_path: str, name: str, args: dict,
             timeout_seconds: float = 120) -> str:
    if name == "execute_sql":
        sql = str(args.get("sql", ""))
        if not sql.lstrip().upper().startswith(
            ("SELECT", "WITH", "PRAGMA", "EXPLAIN QUERY PLAN")
        ):
            return (
                "error: only read-only SELECT/WITH/PRAGMA/"
                "EXPLAIN QUERY PLAN is allowed"
            )
        ok, columns, rows, err = execute_with_columns(
            db_path, sql, max_rows=50, timeout_seconds=timeout_seconds
        )
        return err if not ok else _format_table(columns, rows)
    # Internal validation is deliberately not exposed in TOOLS. The runtime
    # invokes it automatically before accepting a policy model's final answer.
    if name == "validate_sql":
        return _validate_sql(db_path, str(args.get("sql", "")), timeout_seconds)
    return f"error: unknown tool {name}"
