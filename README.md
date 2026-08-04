# SkillAgent Code and Prepared Data

This repository contains the SkillAgent source snapshot and the filtered data
used by the current Text-to-SQL experiments.

## Repository layout

```text
code/SkillAgent/       SkillAgent source, configs, prompts and scripts
prepared_data/         Filtered train/validation/test JSON and audit records
DB_MANIFEST.json       Inventory of the 303 referenced SQLite databases
README_FULL_DATA.md    Instructions for using the complete database bundle
```

Prepared split counts:

| Split / dataset | Items |
|---|---:|
| BIRD train | 987 |
| SynSQL validation | 193 |
| BIRD test | 1,521 |
| EHRSQL test | 744 |
| Spider2.0 test | 135 |

The 303 unique SQLite databases occupy 35,792,005,120 bytes uncompressed. They
are distributed separately from ordinary Git history. Use `DB_MANIFEST.json`
to verify the expected paths and sizes, then use
`prepared_data/relocate_db_paths.py` after extracting the database bundle.

## Training

See `code/SkillAgent/README.md`. A prepared run directory should contain the
contents of `prepared_data/` as its `data/` directory.
