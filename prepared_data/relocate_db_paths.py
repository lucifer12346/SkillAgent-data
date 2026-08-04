#!/usr/bin/env python3
"""Relocate absolute db_path values in prepared SkillAgent item files."""

from __future__ import annotations

import argparse
import json
import os
import shutil


ITEM_FILES = (
    "text2sql_split/train/items.json",
    "text2sql_split/val/items.json",
    "test_sets/bird/items.json",
    "test_sets/ehrsql/items.json",
    "test_sets/spider2.0/items.json",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--old-prefix", required=True)
    parser.add_argument("--new-prefix", required=True)
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create items.json.bak before rewriting.",
    )
    args = parser.parse_args()

    old = os.path.normpath(args.old_prefix)
    new = os.path.normpath(args.new_prefix)
    total_changed = 0

    for relative in ITEM_FILES:
        path = os.path.join(args.root, relative)
        with open(path, encoding="utf-8") as handle:
            rows = json.load(handle)
        changed = 0
        for row in rows:
            value = row.get("db_path")
            if not isinstance(value, str):
                continue
            normalized = os.path.normpath(value)
            try:
                suffix = os.path.relpath(normalized, old)
            except ValueError:
                continue
            if suffix == os.pardir or suffix.startswith(os.pardir + os.sep):
                continue
            row["db_path"] = os.path.abspath(os.path.join(new, suffix))
            changed += 1
        if changed:
            if not args.no_backup:
                shutil.copy2(path, path + ".bak")
            temporary = path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(rows, handle, ensure_ascii=False, indent=2)
            os.replace(temporary, path)
        total_changed += changed
        print(f"{relative}: changed={changed}/{len(rows)}")

    print(f"total changed={total_changed}")


if __name__ == "__main__":
    main()
