#!/usr/bin/env python3
"""Report whether a SkillAgent run is active, stalled, or complete."""

import argparse
import datetime
import json
import os
import sys


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", help="Run directory created by run_text2sql.sh")
    parser.add_argument("--stale_seconds", type=int, default=600)
    args = parser.parse_args()
    output = os.path.join(args.run_root, "output")
    complete_path = os.path.join(output, "RUN_COMPLETE.json")
    progress_path = os.path.join(output, "progress.json")

    if os.path.exists(complete_path):
        complete = load(complete_path)
        print(f"COMPLETE completed_at={complete.get('completed_at')} summary={complete.get('summary')}")
        return 0
    if not os.path.exists(progress_path):
        print(f"NOT_STARTED progress file not found: {progress_path}")
        return 2

    progress = load(progress_path)
    updated = datetime.datetime.fromisoformat(progress["updated_at"])
    now = datetime.datetime.now().astimezone()
    age = max(0, (now - updated).total_seconds())
    state = "STALLED" if age > args.stale_seconds else "RUNNING"
    done, total = progress.get("done", 0), progress.get("total", 0)
    percentage = 100 * done / total if total else 0
    print(
        f"{state} phase={progress.get('phase')} done={done}/{total} "
        f"percent={percentage:.1f}% eta_seconds={progress.get('eta_seconds')} "
        f"estimated_finish={progress.get('estimated_phase_finish_at')} "
        f"last_update_age_seconds={age:.0f}"
    )
    return 1 if state == "STALLED" else 0


if __name__ == "__main__":
    sys.exit(main())
