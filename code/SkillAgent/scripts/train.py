#!/usr/bin/env python3
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skillagent.trainer import run_training

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--num_epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--batch_no_improvement_patience", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--rollout_workers", type=int)
    p.add_argument("--reflect_workers", type=int)
    p.add_argument("--repair_workers", type=int)
    p.add_argument("--max_agent_turns", type=int)
    p.add_argument("--max_schema_link_turns", type=int)
    p.add_argument("--max_completion_tokens", type=int)
    p.add_argument("--schema_link_max_completion_tokens", type=int)
    p.add_argument("--reflect_max_completion_tokens", type=int)
    p.add_argument("--max_repair_rounds", type=int)
    p.add_argument("--repair_max_completion_tokens", type=int)
    p.add_argument("--repair_guide_max_completion_tokens", type=int)
    p.add_argument("--repair_distill_max_completion_tokens", type=int)
    p.add_argument(
        "--enable_guided_sql_repair",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument(
        "--enable_schema_skill_evolution",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument(
        "--enable_repeated_batch_attempts",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument("--exec_timeout", type=int)
    p.add_argument("--split_dir")
    p.add_argument("--test_sets_dir")
    p.add_argument("--out_root")
    args = p.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    run_training(args.config, overrides)
