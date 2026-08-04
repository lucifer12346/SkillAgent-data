# Schema-Linking Skill

1. Decompose the question into requested outputs, filters, aggregations, ordering, and evidence-defined concepts.
2. Inspect table names first, then plausible table definitions; retain the keys needed to connect relevant tables.
3. Match concepts using names, types, relationships, and representative values rather than names alone.
4. Include columns needed for output, filtering, grouping, ordering, computation, and semantic interpretation. Do not return every column from a selected table.
5. Select the smallest plausible set of core tables. The program will deterministically add declared primary keys, foreign-key endpoints, and minimal intermediate join paths after your output.
6. Before finishing, verify that each question constraint maps to at least one selected column. When uncertainty remains, include only the specific alternative columns supported by names, types, relationships, or sampled values.
