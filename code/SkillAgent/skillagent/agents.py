from __future__ import annotations
import json
import re
import sqlite3
from functools import lru_cache
from .evaluator import evaluate, evaluate_schema, extract_sql, gold_schema
from .tools import TOOLS, SCHEMA_LINK_TOOLS, dispatch

def tool_call_stats(trajectory: list[dict]) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    for event in trajectory:
        if event.get("type") != "tool_call":
            continue
        name = str(event.get("name", "unknown"))
        counts[name] = counts.get(name, 0) + 1
    return sum(counts.values()), counts

def schema_text(db_path: str, limit: int = 6000) -> str:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name").fetchall()
    con.close()
    return ";\n\n".join(r[0] for r in rows)[:limit]


@lru_cache(maxsize=256)
def complete_schema_catalog(db_path: str) -> dict:
    """Return every user table/view and column, plus PK/FK relationships.

    This compact catalog is intentionally untruncated.  Schema-linking output
    is only a relevance hint; the SQL policy must always have access to the
    complete database structure when recovering from a missed link.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        objects = con.execute(
            "SELECT name, type, sql FROM sqlite_master "
            "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' "
            "ORDER BY type, name"
        ).fetchall()
        catalog = []
        for name, object_type, definition in objects:
            quoted = str(name).replace('"', '""')
            introspection_error = None
            try:
                columns = [{
                    "name": str(row[1]),
                    "type": str(row[2] or ""),
                    "not_null": bool(row[3]),
                    "primary_key_position": int(row[5] or 0),
                } for row in con.execute(
                    f'PRAGMA table_info("{quoted}")'
                ).fetchall()]
            except sqlite3.Error as exc:
                # Some Spider2 databases contain views whose stored DDL refers
                # to a removed column. Preserve their DDL without allowing one
                # invalid view to hide the rest of the database schema.
                columns = []
                introspection_error = f"{type(exc).__name__}: {exc}"
            foreign_keys = []
            if object_type == "table":
                foreign_keys = [{
                    "from_column": str(row[3]),
                    "to_table": str(row[2]),
                    "to_column": str(row[4]),
                } for row in con.execute(
                    f'PRAGMA foreign_key_list("{quoted}")'
                ).fetchall()]
            catalog.append({
                "name": str(name),
                "object_type": str(object_type),
                "columns": columns,
                "foreign_keys": foreign_keys,
                "definition": str(definition or "") if object_type == "view" else None,
                "introspection_error": introspection_error,
            })
        return {"objects": catalog}
    finally:
        con.close()

def _quote_ddl_identifier(value: str) -> str:
    return f'"{str(value).replace(chr(34), chr(34) * 2)}"'

def _render_relation_ddl(obj: dict, selected_columns: set[str] | None = None) -> str:
    """Render one catalog object as compact CREATE TABLE-style schema context."""
    columns = obj.get("columns", [])
    if selected_columns is not None:
        selected = {str(name).lower() for name in selected_columns}
        columns = [
            column for column in columns
            if str(column.get("name", "")).lower() in selected
        ]
    if not columns:
        definition = str(obj.get("definition") or "").strip()
        return definition.rstrip(";") + ";" if definition else (
            f"-- Schema unavailable for {_quote_ddl_identifier(obj['name'])}"
        )

    lines = []
    primary_key_columns = sorted(
        (
            (int(column.get("primary_key_position", 0)), str(column["name"]))
            for column in columns
            if int(column.get("primary_key_position", 0)) > 0
        ),
        key=lambda pair: pair[0],
    )
    for column in columns:
        parts = [_quote_ddl_identifier(column["name"])]
        if column.get("type"):
            parts.append(str(column["type"]))
        if column.get("not_null"):
            parts.append("NOT NULL")
        lines.append("  " + " ".join(parts))
    if primary_key_columns:
        lines.append(
            "  PRIMARY KEY ("
            + ", ".join(_quote_ddl_identifier(name) for _, name in primary_key_columns)
            + ")"
        )
    visible_columns = {str(column["name"]).lower() for column in columns}
    for foreign_key in obj.get("foreign_keys", []):
        if str(foreign_key.get("from_column", "")).lower() not in visible_columns:
            continue
        lines.append(
            "  FOREIGN KEY ("
            + _quote_ddl_identifier(foreign_key["from_column"])
            + ") REFERENCES "
            + _quote_ddl_identifier(foreign_key["to_table"])
            + " ("
            + _quote_ddl_identifier(foreign_key["to_column"])
            + ")"
        )
    object_note = " -- source object is a view" if obj.get("object_type") == "view" else ""
    return (
        f"CREATE TABLE {_quote_ddl_identifier(obj['name'])} (\n"
        + ",\n".join(lines)
        + f"\n);{object_note}"
    )

def schema_catalog_ddl(db_path: str, schema_links: dict | None = None) -> str:
    """Format complete or schema-linked relations as CREATE TABLE statements."""
    objects = complete_schema_catalog(db_path).get("objects", [])
    if schema_links is None:
        return "\n\n".join(_render_relation_ddl(obj) for obj in objects)
    selected_by_table = {
        str(entry.get("table", "")).lower(): set(entry.get("columns", []))
        for entry in schema_links.get("tables", [])
        if isinstance(entry, dict)
    }
    return "\n\n".join(
        _render_relation_ddl(obj, selected_by_table[str(obj["name"]).lower()])
        for obj in objects
        if str(obj["name"]).lower() in selected_by_table
    ) or "-- No schema-linking hint was produced."

def _expand_schema_links_with_keys(db_path: str, links: dict) -> tuple[dict, dict]:
    """Add PKs and minimal declared-FK paths connecting selected tables."""
    objects = complete_schema_catalog(db_path).get("objects", [])
    by_key = {str(obj["name"]).lower(): obj for obj in objects}
    selected: dict[str, set[str]] = {}
    for entry in links.get("tables", []):
        key = str(entry.get("table", "")).lower()
        if key in by_key:
            selected.setdefault(key, set()).update(
                str(column) for column in entry.get("columns", [])
            )
    original_keys = list(selected)
    adjacency: dict[str, list[tuple[str, str, str]]] = {
        key: [] for key in by_key
    }
    for source_key, obj in by_key.items():
        for foreign_key in obj.get("foreign_keys", []):
            target_key = str(foreign_key.get("to_table", "")).lower()
            if target_key not in by_key:
                continue
            source_column = str(foreign_key.get("from_column", ""))
            target_column = str(foreign_key.get("to_column", ""))
            adjacency[source_key].append(
                (target_key, source_column, target_column)
            )
            adjacency[target_key].append(
                (source_key, target_column, source_column)
            )

    added_paths = []
    if len(original_keys) > 1:
        connected = {original_keys[0]}
        remaining = set(original_keys[1:])
        while remaining:
            queue = [(key, []) for key in connected]
            visited = set(connected)
            found = None
            while queue and found is None:
                node, path = queue.pop(0)
                for neighbor, node_column, neighbor_column in adjacency.get(node, []):
                    if neighbor in visited:
                        continue
                    next_path = path + [
                        (node, node_column, neighbor, neighbor_column)
                    ]
                    if neighbor in remaining:
                        found = next_path
                        break
                    visited.add(neighbor)
                    queue.append((neighbor, next_path))
            if found is None:
                break
            path_tables = []
            for source, source_column, target, target_column in found:
                selected.setdefault(source, set())
                selected.setdefault(target, set())
                if source_column:
                    selected[source].add(source_column)
                if target_column:
                    selected[target].add(target_column)
                path_tables.extend([source, target])
            connected.update(path_tables)
            remaining.difference_update(connected)
            added_paths.append(path_tables)

    for key in list(selected):
        for column in by_key[key].get("columns", []):
            if int(column.get("primary_key_position", 0)) > 0:
                selected[key].add(str(column["name"]))

    expanded = {"tables": []}
    for key, columns in selected.items():
        obj = by_key[key]
        column_order = {
            str(column["name"]).lower(): index
            for index, column in enumerate(obj.get("columns", []))
        }
        expanded["tables"].append({
            "table": str(obj["name"]),
            "columns": sorted(
                columns,
                key=lambda name: column_order.get(str(name).lower(), 10**9),
            ),
        })
    return expanded, {
        "original_tables": len(original_keys),
        "expanded_tables": len(expanded["tables"]),
        "added_path_count": len(added_paths),
        "added_paths": added_paths,
    }

def _normalized_sql(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").split())

def tool_loop(pool, messages, db_path: str, max_turns: int, max_tokens: int,
              require_final_validation: bool = False, available_tools=None,
              temperature=None, sql_timeout: float = 120):
    # Keep a complete, audit-friendly trace rather than tool calls only. Copy
    # the input messages so later mutation of `messages` cannot alter history.
    trajectory = [
        {
            "type": "message",
            "turn": 0,
            "role": str(message.get("role", "")),
            "content": message.get("content", ""),
        }
        for message in messages
    ]
    last_text, last_sql = "", ""
    for turn in range(1, max_turns + 1):
        msg = pool.complete(messages, tools=available_tools or TOOLS,
                            max_tokens=max_tokens, temperature=temperature)
        content = (msg.content or "").strip()
        # Some OpenAI-compatible reasoning models expose the hidden/generated
        # reasoning trace as `reasoning_content`; others place it in model_extra.
        model_extra = getattr(msg, "model_extra", None) or {}
        reasoning = (
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning", None)
            or model_extra.get("reasoning_content")
            or model_extra.get("reasoning")
        )
        assistant_record = {
            "type": "message",
            "turn": turn,
            "role": "assistant",
            "content": content,
        }
        if reasoning:
            assistant_record["reasoning_content"] = str(reasoning)
        trajectory.append(assistant_record)
        if content:
            last_text = content
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            proposed = extract_sql(content or last_text)
            if not require_final_validation:
                return content or last_text, trajectory
            validation = dispatch(
                db_path, "validate_sql", {"sql": proposed},
                timeout_seconds=sql_timeout,
            )
            trajectory.append({
                "type": "validation", "turn": turn,
                "phase": "sql_generation", "sql": proposed,
                "observation": validation, "automatic_final_check": True,
            })
            if validation.startswith("VALID:"):
                return content or last_text, trajectory
            reminder = (
                "The runtime rejected the proposed final SQL: "
                f"{validation}. Repair it. You may use execute_sql with "
                "EXPLAIN QUERY PLAN on the exact candidate before answering again."
            )
            messages.append({"role": "user", "content": reminder})
            trajectory.append({
                "type": "message", "turn": turn, "role": "user",
                "content": reminder, "validation_gate": True,
            })
            continue
        for call in msg.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except Exception:
                args = {}
            obs = dispatch(
                db_path, call.function.name, args,
                timeout_seconds=sql_timeout,
            )
            trajectory.append({
                "type": "tool_call",
                "turn": turn,
                "role": "assistant",
                "name": call.function.name,
                "arguments": args,
                "observation": obs,
            })
            if call.function.name == "execute_sql" and not obs.startswith("error"):
                last_sql = str(args.get("sql", ""))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": obs})
    final_text = last_text or (f"```sql\n{last_sql}\n```" if last_sql else "")
    if require_final_validation:
        final_sql = extract_sql(final_text)
        obs = dispatch(
            db_path, "validate_sql", {"sql": final_sql},
            timeout_seconds=sql_timeout,
        )
        trajectory.append({
            "type": "validation", "turn": max_turns + 1,
            "phase": "sql_generation", "sql": final_sql,
            "observation": obs, "automatic_final_check": True,
        })
        if not obs.startswith("VALID:"):
            return "", trajectory
    return final_text, trajectory

SCHEMA_LINK_SYSTEM = """You are a dedicated SQLite schema-linking agent.
Identify only the database tables and columns needed to answer the question.
You do not write SQL for the user. The only available tool is `execute_sql`.
Use it for multiple turns: query `sqlite_master` to discover tables and DDL,
query `pragma_table_info('table_name')` and
`pragma_foreign_key_list('table_name')` to inspect plausible relations, then
query small representative/distinct values only when needed to resolve semantic
ambiguity. Minimize false positives while retaining join keys that are required
to connect relevant tables.

Finish with JSON only, without Markdown or explanation:
{{"tables":[{{"table":"exact_table_name","columns":["exact_column_name"]}}]}}
The final JSON must contain table and column names only.

## Reusable Schema-Linking Skill
{schema_skill}"""

def _validated_schema_links(db_path: str, text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        proposed = json.loads(match.group(0) if match else text)
    except Exception:
        proposed = {}
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        actual_tables = {
            str(row[0]).lower(): str(row[0])
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        result = []
        seen = set()
        for entry in proposed.get("tables", []) if isinstance(proposed, dict) else []:
            if not isinstance(entry, dict):
                continue
            table = actual_tables.get(str(entry.get("table", "")).lower())
            if not table or table in seen:
                continue
            actual_columns = {
                str(row[1]).lower(): str(row[1])
                for row in con.execute(
                    f'PRAGMA table_info("{table.replace(chr(34), chr(34) * 2)}")'
                ).fetchall()
            }
            columns = []
            for name in entry.get("columns", []):
                column = actual_columns.get(str(name).lower())
                if column and column not in columns:
                    columns.append(column)
            if columns:
                result.append({"table": table, "columns": columns})
                seen.add(table)
        return {"tables": result}
    finally:
        con.close()

def schema_link_one(pool, item: dict, cfg: dict, schema_skill: str,
                    temperature=None):
    messages = [
        {"role": "system", "content": SCHEMA_LINK_SYSTEM.format(schema_skill=schema_skill)},
        {"role": "user", "content": (
            f"External knowledge:\n{item.get('evidence', '')}\n\n"
            f"Question:\n{item['question']}"
        )},
    ]
    text, trajectory = tool_loop(
        pool, messages, item["db_path"], cfg.get("max_schema_link_turns", 6),
        cfg.get("schema_link_max_completion_tokens", 4096),
        available_tools=SCHEMA_LINK_TOOLS, temperature=temperature,
        sql_timeout=cfg.get("exec_timeout", 120),
    )
    links = _validated_schema_links(item["db_path"], text)
    if not links["tables"]:
        reminder = (
            "Using the schema evidence already collected, return the final "
            "JSON now. Include only exact relevant table and column names: "
            '{"tables":[{"table":"...","columns":["..."]}]}.'
        )
        messages.append({"role": "user", "content": reminder})
        final_msg = pool.complete(
            messages,
            max_tokens=cfg.get("schema_link_max_completion_tokens", 4096),
            temperature=temperature,
        )
        text = (final_msg.content or "").strip()
        trajectory.append({
            "type": "message", "turn": cfg.get("max_schema_link_turns", 6) + 1,
            "role": "assistant", "content": text, "schema_link_finalizer": True,
        })
        links = _validated_schema_links(item["db_path"], text)
    for event in trajectory:
        event["phase"] = "schema_linking"
    expanded_links, expansion = _expand_schema_links_with_keys(
        item["db_path"], links
    )
    trajectory.append({
        "type": "schema_expansion",
        "turn": max((int(event.get("turn", 0)) for event in trajectory), default=0),
        "phase": "schema_linking",
        "name": "deterministic_pk_fk_expansion",
        **expansion,
    })
    return expanded_links, trajectory

def schema_rollout_one(pool, item: dict, schema_skill: str, cfg: dict,
                       temperature=None) -> dict:
    schema_links, trajectory = schema_link_one(
        pool, item, cfg, schema_skill, temperature=temperature
    )
    gt_schema = gold_schema(
        item["gold_sql"], item["db_path"],
        timeout_seconds=cfg.get("exec_timeout", 120),
    )
    metrics = evaluate_schema(schema_links, gt_schema)
    trajectory.append({
        "type": "metric",
        "turn": max((int(event.get("turn", 0)) for event in trajectory), default=0),
        "phase": "schema_linking",
        "name": "schema_linking_accuracy",
        "available": True,
        **metrics,
    })
    return {
        **item, **metrics, "schema_linking": schema_links,
        "gold_schema": gt_schema, "schema_linking_trajectory": trajectory,
        "schema_linking_tool_call_count": tool_call_stats(trajectory)[0],
        "schema_linking_tool_call_counts": tool_call_stats(trajectory)[1],
    }

def schema_context_one(pool, item: dict, schema_skill: str, cfg: dict,
                       temperature=None) -> dict:
    """Build the reusable schema context for one policy rollout."""
    # Spider2-lite publishes result tables rather than gold SQL.  It still uses
    # the exact same schema-linking and SQL-generation path, but cannot compute
    # gold-schema containment.  Datasets with gold SQL retain the original path.
    if item.get("gold_sql"):
        return schema_rollout_one(
            pool, item, schema_skill, cfg, temperature=temperature
        )
    schema_links, schema_trajectory = schema_link_one(
        pool, item, cfg, schema_skill, temperature=temperature
    )
    schema_trajectory.append({
        "type": "metric",
        "turn": max((int(event.get("turn", 0)) for event in schema_trajectory), default=0),
        "phase": "schema_linking",
        "name": "schema_linking_accuracy",
        "available": False,
        "reason": "dataset does not provide gold SQL or gold schema",
    })
    return {
        **item,
        "schema_linking": schema_links,
        "schema_linking_trajectory": schema_trajectory,
        "schema_linking_tool_call_count": tool_call_stats(schema_trajectory)[0],
        "schema_linking_tool_call_counts": tool_call_stats(schema_trajectory)[1],
    }


def rollout_one(pool, item: dict, skill: str, cfg: dict, schema_skill: str,
                temperature=None, evaluator=None,
                cached_schema_result: dict | None = None) -> dict:
    if cached_schema_result is not None:
        # SQL-policy comparisons must share an identical schema-linking result.
        # Copy the outer mapping so rollout-specific result fields cannot alter
        # the reusable cached record.
        schema_result = dict(cached_schema_result)
    else:
        schema_result = schema_context_one(
            pool, item, schema_skill, cfg, temperature=temperature
        )
    schema_links = schema_result["schema_linking"]
    schema_trajectory = schema_result["schema_linking_trajectory"]
    schema_reference = schema_catalog_ddl(item["db_path"], schema_links)
    full_schema_reference = schema_catalog_ddl(item["db_path"])
    system = f"""You are a SQLite Text2SQL agent. Follow the reusable skill below. You may inspect the database with tools.

MANDATORY WORKFLOW:
1. You receive both (a) a schema-linking relevance hint and (b) the complete
   database schema. Treat linked tables/columns as higher-relevance candidates,
   not as a restrictive allowlist or guaranteed-correct answer.
2. Independently verify coverage against the question. Use any object in the
   complete schema when a required output, join key, filter, grouping, ordering,
   computation, or evidence-defined concept is absent from the linked subset.
3. `execute_sql` is the only tool. Use it for schema/value inspection and
   candidate diagnosis; use `EXPLAIN QUERY PLAN <exact candidate SQL>` when a
   syntax/schema check is needed.
4. Before answering, ensure the exact final SQL is syntactically valid and
   references existing schema. The runtime automatically validates it and will
   request a repair if SQLite rejects it.
5. Return exactly one final SQL query in a sql fenced block only.

## Skill
{skill}"""
    user = f"""## High-Relevance Schema-Linking Hint
The following tables/columns were selected as especially relevant. Prioritize
checking them, but do not assume they are complete or exclusively usable:
{schema_reference}

## Complete Database Schema
This DDL catalog contains every user table/view and every column, including
primary-key and foreign-key relationships. Views are represented as CREATE TABLE
relations for a uniform schema-reading format. It is authoritative for coverage:
{full_schema_reference}

## External Knowledge
{item.get('evidence','')}

## Question
{item['question']}"""
    response, sql_trajectory = tool_loop(pool, [{"role": "system", "content": system}, {"role": "user", "content": user}], item["db_path"], cfg["max_agent_turns"], cfg["max_completion_tokens"], require_final_validation=True, available_tools=TOOLS, temperature=temperature, sql_timeout=cfg.get("exec_timeout", 120))
    for event in sql_trajectory:
        event["phase"] = "sql_generation"
    score = (
        evaluator(response, item)
        if evaluator is not None
        else evaluate(
            response, item["gold_sql"], item["db_path"],
            timeout_seconds=cfg.get("exec_timeout", 120),
        )
    )
    sql_tool_count, sql_tool_counts = tool_call_stats(sql_trajectory)
    schema_tool_counts = schema_result.get("schema_linking_tool_call_counts", {})
    total_tool_counts = dict(schema_tool_counts)
    for name, count in sql_tool_counts.items():
        total_tool_counts[name] = total_tool_counts.get(name, 0) + count
    return {**schema_result, **score, "response": response,
            # Keep the two phases separate: schema-linking is already stored in
            # schema_linking_trajectory, so the top-level trajectory is SQL-only.
            "trajectory": sql_trajectory,
            "sql_generation_tool_call_count": sql_tool_count,
            "sql_generation_tool_call_counts": sql_tool_counts,
            "tool_call_count": sum(total_tool_counts.values()),
            "tool_call_counts": total_tool_counts,
            "fail_reason": score.get("error") or ("result mismatch" if not score["hard"] else "")}

REPAIR_AGENT_SYSTEM = """You are a SQLite SQL repair agent. Repair a failed SQL
attempt by following the question, external knowledge, complete database DDL,
execution observations, and guidance supplied by a privileged guide agent.

You cannot see the gold SQL. The guide may describe semantic differences and
suggest checks, but you must independently construct and verify the repair.
`execute_sql` is the only tool. You may use it repeatedly to inspect schema,
values, intermediate results, and candidate queries. Do not modify the database.

When ready, return exactly one repaired SQL query in a sql fenced block and no
other text. Do not stop at an explanation."""

REPAIR_GUIDE_SYSTEM = """You are the privileged GT SQL repair guide used only
during training. You can see the gold SQL, the failed SQL, and the repair
agent's execution history. Diagnose why the current candidate is not
execution-equivalent to the gold SQL and give the repair agent the most useful
next action.

Do not simply reveal or copy the gold SQL. Point out semantic operators,
relations, joins, filters, aggregation grain, grouping, ordering, limits, null
behavior, or value transformations that differ. Use concrete evidence for
repair, then state a reusable lesson. Return JSON only:
{"diagnosis":"...","next_actions":["..."],"reusable_lesson":"..."}"""


def _reasoning_from_message(msg) -> str:
    model_extra = getattr(msg, "model_extra", None) or {}
    return str(
        getattr(msg, "reasoning_content", None)
        or getattr(msg, "reasoning", None)
        or model_extra.get("reasoning_content")
        or model_extra.get("reasoning")
        or ""
    )


def _json_from_text(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        value = json.loads(match.group(0) if match else text)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def guided_repair_one(pool, failed: dict, cfg: dict) -> dict:
    """Repair one failed SQL with an alternating GT-guide/repair-agent loop.

    Gold SQL is sent only to the guide. The repair agent receives the guide's
    diagnosis and can interact with the same read-only SQL environment used by
    the policy. The original rollout score is never overwritten.
    """
    max_rounds = max(1, int(cfg.get("max_repair_rounds", 6)))
    max_tokens = int(cfg.get(
        "repair_max_completion_tokens",
        cfg.get("max_completion_tokens", 16384),
    ))
    guide_tokens = int(cfg.get(
        "repair_guide_max_completion_tokens",
        cfg.get("reflect_max_completion_tokens", 16384),
    ))
    timeout = float(cfg.get("exec_timeout", 120))
    original_sql = str(failed.get("predicted_sql", ""))
    current_sql = original_sql
    last_score = {"hard": 0, "error": failed.get("fail_reason", "")}
    repair_trajectory = []
    guide_trajectory = []
    reusable_lessons = []
    repair_messages = [
        {"role": "system", "content": REPAIR_AGENT_SYSTEM},
        {"role": "user", "content": (
            f"## Complete Database Schema\n{schema_catalog_ddl(failed['db_path'])}\n\n"
            f"## External Knowledge\n{failed.get('evidence', '')}\n\n"
            f"## Question\n{failed['question']}\n\n"
            f"## Failed SQL\n```sql\n{original_sql}\n```\n\n"
            f"## Failure\n{failed.get('fail_reason', 'result mismatch')}"
        )},
    ]

    for repair_round in range(1, max_rounds + 1):
        recent_events = repair_trajectory[-8:]
        guide_payload = {
            "question": failed["question"],
            "external_knowledge": failed.get("evidence", ""),
            "gold_sql": failed.get("gold_sql", ""),
            "original_failed_sql": original_sql,
            "current_candidate_sql": current_sql,
            "current_evaluation": last_score,
            "recent_repair_events": recent_events,
        }
        guide_msg = pool.complete(
            [
                {"role": "system", "content": REPAIR_GUIDE_SYSTEM},
                {"role": "user", "content": json.dumps(
                    guide_payload, ensure_ascii=False, indent=2
                )},
            ],
            max_tokens=guide_tokens,
            temperature=0,
        )
        guide_text = (guide_msg.content or "").strip()
        guide = _json_from_text(guide_text)
        lesson = str(guide.get("reusable_lesson", "")).strip()
        if lesson:
            reusable_lessons.append(lesson)
        guide_event = {
            "type": "guide_message",
            "phase": "sql_repair",
            "round": repair_round,
            "content": guide_text,
            "diagnosis": guide.get("diagnosis", ""),
            "next_actions": guide.get("next_actions", []),
            "reusable_lesson": lesson,
        }
        guide_reasoning = _reasoning_from_message(guide_msg)
        if guide_reasoning:
            guide_event["reasoning_content"] = guide_reasoning
        guide_trajectory.append(guide_event)

        repair_messages.append({
            "role": "user",
            "content": (
                f"Privileged guide feedback for repair round {repair_round}:\n"
                f"{guide_text}\n\nUse execute_sql as needed. Return the repaired "
                "SQL only when you are ready for evaluation."
            ),
        })
        repair_msg = pool.complete(
            repair_messages,
            tools=TOOLS,
            max_tokens=max_tokens,
            temperature=0,
        )
        content = (repair_msg.content or "").strip()
        repair_event = {
            "type": "message",
            "phase": "sql_repair",
            "round": repair_round,
            "role": "assistant",
            "content": content,
        }
        repair_reasoning = _reasoning_from_message(repair_msg)
        if repair_reasoning:
            repair_event["reasoning_content"] = repair_reasoning
        repair_trajectory.append(repair_event)
        repair_messages.append(repair_msg.model_dump(exclude_none=True))

        if repair_msg.tool_calls:
            for call in repair_msg.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except Exception:
                    args = {}
                observation = dispatch(
                    failed["db_path"], call.function.name, args,
                    timeout_seconds=timeout,
                )
                repair_trajectory.append({
                    "type": "tool_call",
                    "phase": "sql_repair",
                    "round": repair_round,
                    "role": "assistant",
                    "name": call.function.name,
                    "arguments": args,
                    "observation": observation,
                })
                repair_messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": observation,
                })
            continue

        proposed_sql = extract_sql(content)
        if proposed_sql:
            current_sql = proposed_sql
        validation = dispatch(
            failed["db_path"], "validate_sql", {"sql": current_sql},
            timeout_seconds=timeout,
        )
        if validation.startswith("VALID:"):
            response = f"```sql\n{current_sql}\n```"
            try:
                last_score = evaluate(
                    response, failed["gold_sql"], failed["db_path"],
                    timeout_seconds=timeout,
                )
            except Exception as exc:
                last_score = {
                    "hard": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            last_score = {"hard": 0, "error": validation}
        repair_trajectory.append({
            "type": "repair_evaluation",
            "phase": "sql_repair",
            "round": repair_round,
            "sql": current_sql,
            "syntax_validation": validation,
            "hard": int(last_score.get("hard", 0) or 0),
            "error": last_score.get("error", ""),
        })
        if last_score.get("hard"):
            break
        repair_messages.append({
            "role": "user",
            "content": (
                "The candidate is syntactically valid but is not execution-"
                "equivalent to the target. Continue repairing it in the next "
                "round; use the next guide diagnosis and database evidence."
            ),
        })

    repair_tool_count, repair_tool_counts = tool_call_stats(repair_trajectory)
    return {
        "sample_id": failed.get("id"),
        "original_sql": original_sql,
        "repaired_sql": current_sql,
        "repair_success": bool(last_score.get("hard")),
        "repair_rounds": len(guide_trajectory),
        "repair_error": last_score.get("error", ""),
        "repair_reusable_lessons": reusable_lessons,
        "repair_trajectory": repair_trajectory,
        "repair_guide_trajectory": guide_trajectory,
        "repair_tool_call_count": repair_tool_count,
        "repair_tool_call_counts": repair_tool_counts,
    }

REPAIR_TRACE_DISTILL_SYSTEM = """You distill a completed guided SQL-repair
trajectory into reusable Text2SQL skill improvements. The trajectory is richer
evidence than a one-shot reflection: identify the decisive semantic correction,
which guide advice or database observation enabled it, which attempted actions
were unhelpful, and which verification would have prevented the original error.

Gold SQL and concrete schema details are privileged diagnostic evidence only.
The output will be used to rewrite a deployable skill, so:
- do not copy the gold or repaired SQL;
- do not mention task IDs, database/table/column names, literal values, or
  sample-specific entities in suggestions;
- convert the successful repair into explicit, transferable decision
  procedures, trigger conditions, fallback branches, and final checks;
- if repair failed, explain the unresolved gap and distill only lessons
  supported by the trace;
- do not claim a lesson when the apparent repair was only stochastic.

Return JSON only:
{"diagnosis":"...","repair_key":"...","generalizable":true,
 "suggestions":["..."]}
Set generalizable=false and suggestions=[] if the trace supports no safe,
transferable improvement."""


def distill_repair_trace_one(pool, failed: dict, current_skill: str,
                             cfg: dict) -> dict:
    """Replace one-shot reflection with evidence from the guided repair loop."""
    evidence = {
        "question": failed.get("question", ""),
        "external_knowledge": failed.get("evidence", ""),
        "gold_sql": failed.get("gold_sql", ""),
        "original_predicted_sql": failed.get("predicted_sql", ""),
        "original_failure": failed.get("fail_reason", ""),
        "repair_success": failed.get("repair_success", False),
        "repaired_sql": failed.get("repaired_sql", ""),
        "repair_rounds": failed.get("repair_rounds", 0),
        "repair_error": failed.get("repair_error", ""),
        "repair_trajectory": failed.get("repair_trajectory", []),
        "guide_trajectory": failed.get("repair_guide_trajectory", []),
        "guide_reusable_lessons": failed.get(
            "repair_reusable_lessons", []
        ),
        "current_skill": current_skill,
    }
    msg = pool.complete(
        [
            {"role": "system", "content": REPAIR_TRACE_DISTILL_SYSTEM},
            {"role": "user", "content": json.dumps(
                evidence, ensure_ascii=False, indent=2
            )},
        ],
        max_tokens=cfg.get(
            "repair_distill_max_completion_tokens",
            cfg.get("reflect_max_completion_tokens", 16384),
        ),
        temperature=0,
    )
    result = _json_from_text(msg.content or "")
    if not result:
        result = {
            "diagnosis": "unparseable repair-trace distillation",
            "repair_key": "",
            "generalizable": False,
            "suggestions": [],
        }
    if not result.get("generalizable"):
        result["suggestions"] = []
    if not isinstance(result.get("suggestions"), list):
        result["suggestions"] = [str(result["suggestions"])]
    result["sample_id"] = failed.get("id")
    result["source"] = "guided_repair_trace"
    result["repair_success"] = bool(failed.get("repair_success"))
    result["repair_rounds"] = int(failed.get("repair_rounds", 0) or 0)
    reasoning = _reasoning_from_message(msg)
    if reasoning:
        result["reasoning_content"] = reasoning
    return result

REFLECT_SYSTEM = """You are a Text2SQL reflection agent. Diagnose one failed attempt and propose reusable improvements to the current skill.

You can see the gold SQL only as privileged diagnostic evidence. You may use read-only database tools to compare the failed query, gold query, schema, and data semantics before concluding.

GENERALIZATION CONTRACT:
- Do not copy the gold SQL into the skill.
- Do not mention task IDs, database names, table names, column names, literal values, or entities from this sample in suggestions.
- Do not produce a rule that only solves this sample or merely paraphrases its answer.
- Express the underlying failure class, decision procedure, verification check, or broadly reusable SQL reasoning rule.
- Prefer operational guidance that transfers to unseen schemas.
- If the failure is an execution lapse already covered by the skill, return no suggestions.

Finish with JSON only in this schema:
{"diagnosis":"...", "generalizable":true, "suggestions":["..."]}
Set generalizable=false and suggestions=[] when no safe transferable lesson exists."""

def reflect_one(pool, failed: dict, current_skill: str, cfg: dict) -> dict:
    evidence = {
        "question": failed["question"], "schema": schema_text(failed["db_path"]),
        "external_knowledge": failed.get("evidence", ""), "gold_sql": failed["gold_sql"],
        "predicted_sql": failed.get("predicted_sql", ""), "model_response": failed.get("response", ""),
        "error_or_mismatch": failed.get("fail_reason", ""), "target_tool_trajectory": failed.get("trajectory", []),
        "guided_repair": {
            "success": failed.get("repair_success"),
            "repaired_sql": failed.get("repaired_sql", ""),
            "rounds": failed.get("repair_rounds", 0),
            "guide_lessons": failed.get("repair_reusable_lessons", []),
            "repair_error": failed.get("repair_error", ""),
        },
        "current_skill": current_skill,
    }
    messages = [{"role": "system", "content": REFLECT_SYSTEM},
                {"role": "user", "content": json.dumps(evidence, ensure_ascii=False, indent=2)}]
    text, trajectory = tool_loop(pool, messages, failed["db_path"], cfg["max_agent_turns"], cfg["reflect_max_completion_tokens"], sql_timeout=cfg.get("exec_timeout", 120))
    match = re.search(r"\{.*\}", text, re.S)
    try:
        result = json.loads(match.group(0) if match else text)
    except Exception:
        result = {"diagnosis": "unparseable reflection", "generalizable": False, "suggestions": []}
    if not result.get("generalizable"):
        result["suggestions"] = []
    result["reflect_tool_trajectory"] = trajectory
    tool_count, tool_counts = tool_call_stats(trajectory)
    result["reflect_tool_call_count"] = tool_count
    result["reflect_tool_call_counts"] = tool_counts
    result["sample_id"] = failed["id"]
    return result

SCHEMA_REFLECT_SYSTEM = """You improve a reusable SQLite schema-linking skill.
Diagnose missing required schema and excessive irrelevant tables or columns.
Gold schema is privileged training evidence only. Never put sample IDs, database,
table or column names, literal values, entities, or dataset-specific answers in
suggestions. Produce transferable search, join-closure, pruning, and verification
procedures. Preserve high recall without selecting every DDL column.
If there is no safe general lesson, return no suggestions.

Finish with JSON only:
{"diagnosis":"...","generalizable":true,"suggestions":["..."]}"""

def reflect_schema_one(pool, failed: dict, current_skill: str, cfg: dict) -> dict:
    evidence = {
        "question": failed["question"],
        "external_knowledge": failed.get("evidence", ""),
        "predicted_schema": failed.get("schema_linking", {}),
        "gold_schema": failed.get("gold_schema", {}),
        "missing_tables": failed.get("missing_tables", []),
        "missing_columns": failed.get("missing_columns", {}),
        "table_precision": failed.get("schema_table_precision"),
        "column_precision": failed.get("schema_column_precision"),
        "schema_size_ratio": failed.get("schema_size_ratio"),
        "schema_quality_score": failed.get("schema_quality_score"),
        "trajectory": failed.get("schema_linking_trajectory", []),
        "current_schema_skill": current_skill,
    }
    msg = pool.complete(
        [{"role": "system", "content": SCHEMA_REFLECT_SYSTEM},
         {"role": "user", "content": json.dumps(evidence, ensure_ascii=False, indent=2)}],
        max_tokens=cfg["reflect_max_completion_tokens"],
    )
    match = re.search(r"\{.*\}", msg.content or "", re.S)
    try:
        result = json.loads(match.group(0) if match else (msg.content or ""))
    except Exception:
        result = {"diagnosis": "unparseable reflection", "generalizable": False,
                  "suggestions": []}
    if not result.get("generalizable"):
        result["suggestions"] = []
    result["sample_id"] = failed["id"]
    return result
