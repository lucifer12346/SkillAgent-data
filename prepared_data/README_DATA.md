# SkillAgent Prepared Data (2026-08-04)

This package is the filtered data snapshot prepared for SkillAgent from:

```text
/public/home/yaozhiming/yzm/Text2SQL/SkillAgent/runs/
run_20260730_113510_filtering_data_filtered/data
```

## Contents

| Split / dataset | Items |
|---|---:|
| BIRD training | 987 |
| SynSQL validation | 193 |
| BIRD test | 1,521 |
| EHRSQL test | 744 |
| Spider2.0 test | 135 |

Files:

```text
text2sql_split/train/items.json
text2sql_split/val/items.json
test_sets/bird/items.json
test_sets/ehrsql/items.json
test_sets/spider2.0/items.json
test_sets/manifest.json
FILTERED_DATA_READY
SOURCE_DATA_READY
filter_report.json
removed_annotations.json
review_candidates.json
```

`removed_annotations.json` contains the 297 removed records and reasons.
`review_candidates.json` contains 86 retained samples that need optional manual
review. The original unfiltered source snapshot was not modified.

## Database files are not included

The JSON records contain `db_path` references to the original SQLite databases.
The unique referenced database files occupy approximately:

```text
35,792,005,120 bytes (about 35.8 GB decimal)
```

They are intentionally not duplicated into this small transfer package. On the
original server, the absolute paths remain usable. On another machine, copy or
mount the original Text2SQL data tree, then run `relocate_db_paths.py`.

Example:

```bash
python relocate_db_paths.py \
  --root . \
  --old-prefix /public/home/yaozhiming/yzm/Text2SQL/data \
  --new-prefix /your/local/Text2SQL/data
```

The script creates `.bak` files before modifying any JSON. Spider2.0 evaluator
paths such as `eval_metadata` and `gold_dir` live in the SkillAgent YAML config
and must also be updated for the new machine.

## Using this snapshot

Place the extracted directory under a run root as `data/`:

```text
runs/run_NAME/
└── data/
    ├── FILTERED_DATA_READY
    ├── text2sql_split/
    └── test_sets/
```

Then start training from the SkillAgent project:

```bash
bash scripts/train_text2sql.sh runs/run_NAME
```

The training script accepts `FILTERED_DATA_READY` as the readiness marker.
