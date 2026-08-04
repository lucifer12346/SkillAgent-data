#!/usr/bin/env python3
"""Evaluate one or more skill documents on an existing Text2SQL test split."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def parse_skill_spec(spec: str) -> tuple[str, str]:
    """Parse PATH or NAME=PATH without breaking paths that contain '='."""
    if "=" in spec:
        name, path = spec.split("=", 1)
        if name.strip() and path.strip():
            return name.strip(), path.strip()
    path = spec.strip()
    name = os.path.splitext(os.path.basename(path))[0]
    return name, path


def safe_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return cleaned.strip("_") or "skill"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate specified skill Markdown files on the test split only."
    )
    parser.add_argument(
        "skills",
        nargs="+",
        metavar="[NAME=]PATH",
        help="Skill file path; optionally prefix it with a result name.",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--test_sets_dir", help="Directory built by build_test_sets.py; overrides config.")
    parser.add_argument("--dataset", choices=["bird", "ehrsql", "spider2.0"], default="bird")
    parser.add_argument(
        "--out_dir",
        help="Output directory (default: outputs/eval_skills_<timestamp>).",
    )
    parser.add_argument("--rollout_workers", type=int)
    parser.add_argument("--max_agent_turns", type=int)
    parser.add_argument("--max_completion_tokens", type=int)
    parser.add_argument("--schema_skill", help="Schema-linking skill file; defaults to config schema_skill_init.")
    args = parser.parse_args()

    # Keep --help usable even outside the project's runtime environment.
    from skillagent.agents import rollout_one, schema_context_one
    from skillagent.benchmarks import evaluator_for, load_test_suites_from_dir
    from skillagent.model import ModelPool
    from skillagent.trainer import parallel, score

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("test_sets_dir", "rollout_workers", "max_agent_turns", "max_completion_tokens"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if not cfg.get("test_sets_dir"):
        parser.error("--test_sets_dir is required when the config does not define one")

    out_dir = args.out_dir or os.path.join(
        "outputs", "eval_skills_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=False)

    suite = load_test_suites_from_dir(cfg["test_sets_dir"])[args.dataset]
    test_items = suite["items"]
    evaluator = evaluator_for(suite, cfg.get("exec_timeout", 120))
    pool = ModelPool()
    schema_skill_path = args.schema_skill or cfg["schema_skill_init"]
    with open(schema_skill_path, encoding="utf-8") as f:
        schema_skill = f.read()
    print(
        f"[evaluate:schema-cache] precomputing once for {len(test_items)} "
        "samples; all SQL skills will share these exact schema hints",
        flush=True,
    )
    schema_rows = parallel(
        lambda item: schema_context_one(
            pool, item, schema_skill, cfg, temperature=0
        ),
        test_items,
        cfg["rollout_workers"],
        "test-schema-cache",
    )
    schema_by_id = {str(row.get("id")): row for row in schema_rows}
    summaries = []
    used_names: set[str] = set()

    for spec in args.skills:
        name, skill_path = parse_skill_spec(spec)
        result_name = safe_name(name)
        if result_name in used_names:
            parser.error(f"duplicate skill result name: {result_name}")
        used_names.add(result_name)
        with open(skill_path, encoding="utf-8") as f:
            skill = f.read()

        print(
            f"[evaluate:{result_name}] skill={skill_path} test={len(test_items)}",
            flush=True,
        )
        rows = parallel(
            lambda item, current_skill=skill: rollout_one(
                pool, item, current_skill, cfg, schema_skill,
                temperature=0, evaluator=evaluator,
                cached_schema_result=schema_by_id[str(item.get("id"))],
            ),
            test_items,
            cfg["rollout_workers"],
            f"test-{result_name}",
            track_score=True,
        )
        ex = score(rows)
        result_path = os.path.join(out_dir, f"{result_name}_test_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        summary = {
            "name": result_name,
            "skill_path": os.path.abspath(skill_path),
            "dataset": args.dataset,
            "test_sets_dir": os.path.abspath(cfg["test_sets_dir"]),
            "num_samples": len(rows),
            "correct": sum(int(row.get("hard", 0)) for row in rows),
            "ex": ex,
            "schema_containment": (
                sum(row.get("schema_hard", 0) for row in rows)
                / len([row for row in rows if "schema_hard" in row])
                if any("schema_hard" in row for row in rows) else None
            ),
            "shared_schema_cache": True,
            "schema_skill_path": os.path.abspath(schema_skill_path),
            "results_path": os.path.abspath(result_path),
        }
        summaries.append(summary)
        print(f"[evaluate:{result_name}] EX={ex:.4f}", flush=True)

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"skills": summaries}, f, ensure_ascii=False, indent=2)
    print(json.dumps({"skills": summaries}, ensure_ascii=False, indent=2))
    print(f"[evaluate] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
