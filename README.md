# SkillAgent Code and Prepared Data

This repository contains the SkillAgent source snapshot and the filtered data
used by the current Text-to-SQL experiments.

## Repository layout

```text
code/SkillAgent/       SkillAgent source, configs, prompts and scripts
prepared_data/         Filtered train/validation/test JSON and audit records
database_bundle/       Five Git LFS parts for the complete SQLite bundle
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
are stored as five Git LFS parts under `database_bundle/`, not in ordinary Git
history. Follow `database_bundle/README.md` to reassemble and verify the ZIP,
then use `prepared_data/relocate_db_paths.py` after extraction.

Cloning the complete repository requires Git LFS:

```bash
git lfs install
git clone git@github.com:lucifer12346/SkillAgent-data.git
```

## Training

See `code/SkillAgent/README.md`. A prepared run directory should contain the
contents of `prepared_data/` as its `data/` directory.
