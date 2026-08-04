# Text2SQL Skill

1. Before SQL generation, a dedicated multi-turn schema-linking agent supplies a prioritized relevance hint. It is not an authoritative allowlist: cross-check the complete DDL and recover any required table or column missing from the hint.
2. Identify the exact requested output columns and the intended row granularity.
3. Prefer joins supported by linked PK/FK paths, but inspect the complete DDL and use representative read-only queries when coverage or data semantics are ambiguous.
4. Translate evidence definitions and formulas explicitly into SQLite expressions.
5. Validate filters, aggregation level, grouping, ordering, distinctness, and limits against the requested output.
6. When uncertain, inspect representative rows and execute the candidate query.
7. Before answering, verify that the exact final query is syntactically valid and references existing schema. When useful, call the sole `execute_sql` tool with `EXPLAIN QUERY PLAN <exact query>`; the runtime also performs an automatic final validation and requests repair if SQLite rejects it.
8. Return one final SQLite query in a `sql` fenced code block.
