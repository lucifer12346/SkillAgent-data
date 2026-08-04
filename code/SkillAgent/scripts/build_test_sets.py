#!/usr/bin/env python3
"""Materialize reproducible BIRD, EHRSQL, and Spider2.0 test manifests."""

import argparse
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skillagent.benchmarks import load_test_suites
from skillagent.data import sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bird_n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    # BIRD test construction lives here (and nowhere in build_splits.py).
    bird_items = sample("test", args.bird_n, args.seed + 1)
    suites = load_test_suites(config, bird_items)
    expected = {"bird", "ehrsql", "spider2.0"}
    missing = expected.difference(suites)
    if missing:
        raise ValueError(f"missing required test sets: {sorted(missing)}")

    os.makedirs(args.out_dir, exist_ok=False)
    manifest = {}
    for name in ("bird", "ehrsql", "spider2.0"):
        suite = suites[name]
        dataset_dir = os.path.join(args.out_dir, name)
        os.makedirs(dataset_dir)
        items_path = os.path.join(dataset_dir, "items.json")
        with open(items_path, "w", encoding="utf-8") as handle:
            json.dump(suite["items"], handle, ensure_ascii=False, indent=2)
        manifest[name] = {
            "type": suite["type"],
            "samples": len(suite["items"]),
            "items": os.path.join(name, "items.json"),
            "config": suite["config"],
            "evaluation": (
                "official Spider2 CSV table match"
                if name == "spider2.0"
                else "unordered SQLite execution-result equality"
            ),
        }
        print(
            f"[test-set:{name}] samples={len(suite['items'])} "
            f"evaluation={manifest[name]['evaluation']}",
            flush=True,
        )
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
