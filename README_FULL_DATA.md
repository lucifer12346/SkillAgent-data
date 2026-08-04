# SkillAgent Full Prepared Data Bundle

This bundle contains the filtered prepared splits, all SQLite databases they
reference, and the Spider2.0 gold execution metadata.

- Unique SQLite databases: 303
- Uncompressed SQLite bytes: 35792005120
- Prepared data: `data/`
- Mirrored database tree: `database_files/`
- Spider2 evaluator assets: `spider2_evaluation/`
- File inventory: `DB_MANIFEST.json`

After extraction, relocate `db_path` values to an absolute local path:

```bash
python data/relocate_db_paths.py \
  --root data \
  --old-prefix /public/home/yaozhiming/yzm/Text2SQL/data \
  --new-prefix /absolute/path/to/extracted/database_files
```

Then update `configs/default.yaml` in the code package:

```yaml
test_sets:
  ehrsql:
    data_root: /absolute/path/to/extracted/database_files/EHRSQL
  spider2.0:
    data_root: /absolute/path/to/extracted/database_files/spider2_sqlite
    eval_metadata: /absolute/path/to/extracted/spider2_evaluation/spider2_sqlite_eval.jsonl
    gold_dir: /absolute/path/to/extracted/spider2_evaluation/gold_exec_result
```
