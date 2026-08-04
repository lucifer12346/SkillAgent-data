from __future__ import annotations
import datetime, json, math, os, random, re, shutil, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
from .agents import (
    distill_repair_trace_one,
    guided_repair_one,
    rollout_one,
    reflect_schema_one,
    schema_rollout_one,
)
from .benchmarks import evaluator_for, load_test_suites_from_dir
from .data import load_splits
from .model import ModelPool

CUMULATIVE_SUMMARY_HEADER = """# Cumulative Text2SQL Training Summary

This is private optimization evidence. Each section below summarizes exactly
one new training batch. Earlier sections are retained unchanged so every skill
rewrite can learn from all batches observed so far.
"""

# Validation and held-out tests must be deterministic regardless of the model
# service's environment-level default temperature.
VALIDATION_TEMPERATURE = 0
_PROGRESS_PATH = None


def _duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _write_progress(label, done, total, started, status="running"):
    if not _PROGRESS_PATH:
        return
    elapsed = max(0.001, time.monotonic() - started)
    rate = done / elapsed
    eta_seconds = (total - done) / rate if rate > 0 else None
    finish = (
        datetime.datetime.now().astimezone()
        + datetime.timedelta(seconds=eta_seconds)
        if eta_seconds is not None else None
    )
    payload = {
        "status": status,
        "phase": label,
        "done": done,
        "total": total,
        "elapsed_seconds": elapsed,
        "items_per_second": rate,
        "eta_seconds": eta_seconds,
        "estimated_phase_finish_at": finish.isoformat() if finish else None,
        "updated_at": datetime.datetime.now().astimezone().isoformat(),
    }
    temporary = _PROGRESS_PATH + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, _PROGRESS_PATH)

def parallel(fn, items, workers, label="work", track_score=False,
             on_result=None):
    out = []
    total = len(items)
    hard_sum = 0
    started = time.monotonic()
    print(f"[{label}] start total={total} workers={workers}", flush=True)
    _write_progress(label, 0, total, started)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fn, x) for x in items]
        for done, f in enumerate(as_completed(futures), 1):
            result = f.result()
            out.append(result)
            if on_result is not None:
                on_result(result, done, total)
            if track_score and isinstance(result, dict):
                hard_sum += int(result.get("hard", 0) or 0)
            if done == 1 or done % 10 == 0 or done == total:
                metric = f" running_EX={hard_sum / done:.3f}" if track_score else ""
                elapsed = time.monotonic() - started
                rate = done / max(0.001, elapsed)
                eta = (total - done) / rate if rate > 0 else 0
                finish = (
                    datetime.datetime.now().astimezone()
                    + datetime.timedelta(seconds=eta)
                ).strftime("%Y-%m-%d %H:%M:%S %z")
                print(
                    f"[{label}] {done}/{total}{metric} "
                    f"elapsed={_duration(elapsed)} eta={_duration(eta)} "
                    f"finish={finish}",
                    flush=True,
                )
                _write_progress(label, done, total, started)
    print(f"[{label}] complete", flush=True)
    _write_progress(label, total, total, started, status="phase_complete")
    return out

def score(rows):
    return sum(x["hard"] for x in rows) / max(1, len(rows))

def schema_score(rows):
    return sum(x["schema_hard"] for x in rows) / max(1, len(rows))

def schema_quality_score(rows):
    return sum(x["schema_quality_score"] for x in rows) / max(1, len(rows))

def _json_object(text):
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(match.group(0) if match else text)
    except Exception:
        return {}

def _trajectory_record(row):
    """Keep concrete per-sample evidence for the trajectory summarizer."""
    return {
        "id": row.get("id"),
        "outcome": "correct" if row.get("hard") else "error",
        "question": row.get("question", ""),
        "external_knowledge": row.get("evidence", ""),
        "gold_sql": row.get("gold_sql", ""),
        "predicted_sql": row.get("predicted_sql", ""),
        "fail_reason": row.get("fail_reason", ""),
        "response": row.get("response", ""),
        "trajectory": row.get("trajectory", []),
        "guided_repair": {
            "success": row.get("repair_success"),
            "rounds": row.get("repair_rounds", 0),
            "repaired_sql": row.get("repaired_sql", ""),
            "reusable_lessons": row.get("repair_reusable_lessons", []),
            "error": row.get("repair_error", ""),
        },
        "tool_call_count": row.get("tool_call_count", 0),
        "tool_call_counts": row.get("tool_call_counts", {}),
    }

def summarize_trajectories(pool, rows, chunk_size=4):
    """Hierarchically summarize every correct and incorrect trajectory.

    Intermediate summaries may intentionally retain concrete schema names,
    SQL, values, and task details. They are training evidence, not deployable
    skill text. Chunking keeps a large batch from exceeding model context.
    """
    chunk_summaries = []
    system = """You analyze Text2SQL execution trajectories for skill improvement.
Summarize both correct and incorrect examples. You MAY retain concrete table
names, columns, literal values, questions, SQL fragments, and tool observations
because this is private training evidence. Identify successful tactics, failure
causes, contrasts between predicted and gold SQL, and recurring patterns.
Surface distinct transferable skill opportunities across different task and
failure types; do not force the batch into one task-specific lesson. Do not
write the final skill. Return JSON only: {"summary":"..."}."""
    for start in range(0, len(rows), chunk_size):
        chunk = [_trajectory_record(row) for row in rows[start:start + chunk_size]]
        user = (
            f"Summarize trajectory chunk {start // chunk_size + 1}. "
            "Account for every sample in the chunk:\n"
            + json.dumps(chunk, ensure_ascii=False, indent=2)
        )
        msg = pool.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=4096,
        )
        parsed = _json_object(msg.content or "")
        chunk_summaries.append(str(parsed.get("summary") or msg.content or "").strip())
    if len(chunk_summaries) == 1:
        return chunk_summaries[0], chunk_summaries
    final_prompt = """Combine all chunk summaries into one comprehensive Text2SQL
training summary. Preserve coverage of correct and incorrect trajectories,
concrete evidence, recurring failure clusters, and successful behaviors. Do
not produce a deployable skill. Return JSON only: {"summary":"..."}.

Chunk summaries:
""" + json.dumps(chunk_summaries, ensure_ascii=False, indent=2)
    msg = pool.complete(
        [{"role": "system", "content": "You synthesize trajectory-analysis reports."},
         {"role": "user", "content": final_prompt}],
        max_tokens=8192,
    )
    parsed = _json_object(msg.content or "")
    return str(parsed.get("summary") or msg.content or "").strip(), chunk_summaries

def rewrite_skill(pool, reflections, old_skill, cumulative_summary):
    """Rewrite the complete skill from old skill + feedback; never append."""
    feedback = [
        {
            "sample_id": r.get("sample_id"),
            "source": r.get("source", "guided_repair_trace"),
            "repair_success": r.get("repair_success"),
            "repair_rounds": r.get("repair_rounds", 0),
            "diagnosis": r.get("diagnosis", ""),
            "repair_key": r.get("repair_key", ""),
            "suggestions": r.get("suggestions", []),
        }
        for r in reflections
        if r.get("suggestions")
    ]
    prompt = f"""## Previous Skill
{old_skill}

## Cumulative Summary of Previous and Current Training Batches
This private summary grows by one section per batch and may contain
task-specific data. Use it as evidence only:
{cumulative_summary}

## Per-Failure Guided Repair-Trace Distillation
Each entry summarizes the decisive correction discovered through the current
multi-round Guide/Repair/tool trajectory. Prioritize these repair keys over
speculative one-shot diagnoses:
{json.dumps(feedback, ensure_ascii=False, indent=2)}

Rewrite the ENTIRE previous skill into one improved, self-contained Text2SQL
skill document. This is a full replacement, not an appendix or patch.

Requirements:
- Produce a substantively new candidate, not a conservative paraphrase. Textual
  similarity to the previous skill is not a goal; validation will protect the
  incumbent skill if the candidate is worse.
- Reconsider the document architecture from first principles. You MAY reorder,
  merge, split, replace, or delete previous sections and rules. Preserve an old
  rule only when cumulative evidence still supports it.
- Introduce multiple independent, reusable skill modules when the evidence
  reveals distinct task families or failure classes. Do not constrain the
  rewrite to one sample, one database, or one task type.
- Replace vague reminders with explicit decision procedures, trigger
  conditions, fallback branches, and verification checks that materially change
  how the policy acts.
- Resolve contradictions and remove obsolete, redundant, overly broad, or
  harmful defaults instead of carrying them forward.
- A usable candidate must include meaningful operational changes beyond wording
  and formatting. Do not report a rename, reordering, or stylistic rewrite as a
  substantive change.
- Treat current reflections collectively: extract broadly useful skills from
  every distinct failure class instead of centering the document on a single
  wrong trace.
- Convert concrete examples into general decision procedures.
- The final skill MUST NOT contain task IDs, database/table/column names,
  literal values, copied gold SQL, or dataset-specific answers.
- Keep it concise, operational, and suitable for unseen schemas.
- Treat schema linking as a prioritized relevance hint, never as an
  authoritative allowlist; use the complete DDL to recover missing schema.
- Keep counting guidance internally consistent: decide DISTINCT from the
  requested entity grain and verified join multiplicity, not a blanket default.
- Do not add repeated "Learned Rules" sections.

Return JSON only:
{{"skill":"# Text2SQL Skill\\n...", "changes":["ADDED: ...","REMOVED: ...","RESTRUCTURED: ..."]}}"""
    msg = pool.complete(
        [{"role": "system", "content": (
            "You are an evidence-driven Text2SQL policy architect. Generate "
            "bold, structurally distinct candidate skills; a downstream "
            "validation gate, not textual conservatism, protects quality."
        )},
         {"role": "user", "content": prompt}],
        max_tokens=8192,
    )
    parsed = _json_object(msg.content or "")
    candidate = str(parsed.get("skill") or "").strip()
    changes = parsed.get("changes") or []
    if not isinstance(changes, list):
        changes = [str(changes)]
    if not candidate:
        return old_skill, [], "rewrite output was not valid JSON with a non-empty skill"
    return candidate + "\n", list(changes), ""

def summarize_schema_trajectories(pool, rows, chunk_size=4):
    records = [{
        "id": row.get("id"), "question": row.get("question", ""),
        "external_knowledge": row.get("evidence", ""),
        "predicted_schema": row.get("schema_linking", {}),
        "gold_schema": row.get("gold_schema", {}),
        "missing_tables": row.get("missing_tables", []),
        "missing_columns": row.get("missing_columns", {}),
        "table_precision": row.get("schema_table_precision"),
        "column_precision": row.get("schema_column_precision"),
        "schema_size_ratio": row.get("schema_size_ratio"),
        "schema_quality_score": row.get("schema_quality_score"),
        "outcome": "complete" if row.get("schema_hard") else "incomplete",
        "trajectory": row.get("schema_linking_trajectory", []),
    } for row in rows]
    summaries = []
    system = """Analyze SQLite schema-linking trajectories. Cover every example,
including successful tactics and causes of missing gold tables or columns.
Concrete schema evidence is allowed because this is a private summary. Identify
transferable recall and verification procedures. Do not write the final skill.
Return JSON only: {"summary":"..."}."""
    for start in range(0, len(records), chunk_size):
        msg = pool.complete(
            [{"role": "system", "content": system},
             {"role": "user", "content": json.dumps(records[start:start + chunk_size], ensure_ascii=False, indent=2)}],
            max_tokens=8192,
        )
        parsed = _json_object(msg.content or "")
        summaries.append(str(parsed.get("summary") or msg.content or "").strip())
    if len(summaries) <= 1:
        return (summaries[0] if summaries else ""), summaries
    msg = pool.complete(
        [{"role": "system", "content": "Synthesize schema-linking reports; return JSON only: {\"summary\":\"...\"}."},
         {"role": "user", "content": json.dumps(summaries, ensure_ascii=False, indent=2)}],
        max_tokens=8192,
    )
    parsed = _json_object(msg.content or "")
    return str(parsed.get("summary") or msg.content or "").strip(), summaries

def rewrite_schema_skill(pool, reflections, old_skill, cumulative_summary):
    feedback = [{"diagnosis": r.get("diagnosis", ""),
                 "suggestions": r.get("suggestions", [])}
                for r in reflections if r.get("suggestions")]
    prompt = f"""## Previous Schema-Linking Skill
{old_skill}

## Private Cumulative Training Summary
{cumulative_summary}

## Generalizable Failure Feedback
{json.dumps(feedback, ensure_ascii=False, indent=2)}

Rewrite the complete reusable schema-linking skill as a substantively new
candidate, not a conservative paraphrase. Validation will reject regressions,
so do not optimize for textual similarity to the previous skill.

Reconsider the architecture from first principles: reorder, merge, split,
replace, or delete old rules; introduce explicit search stages, trigger
conditions, fallback branches, pruning rules, and final audits. Remove
redundant, conflicting, vague, or containment-gaming guidance. The candidate
must materially change operational behavior, not merely rename or reorder
sections.

Optimize recall of every required table, output/filter/group/order/computation
column, and join key.
Also optimize precision: never retain every DDL column by default. Select
semantic columns plus keys needed by minimal join paths; the program
deterministically expands declared PK/FK paths after model output.
Do not include sample IDs, database/table/column names, literal values, entities,
gold answers, or dataset-specific rules. Keep it concise and operational.
Return JSON only:
{{"skill":"# Schema-Linking Skill\\n...","changes":["ADDED: ...","REMOVED: ...","RESTRUCTURED: ..."]}}"""
    msg = pool.complete(
        [{"role": "system", "content": (
            "You are an evidence-driven schema-linking policy architect. "
            "Generate bold, structurally distinct candidate skills; downstream "
            "validation protects the incumbent from regressions."
        )},
         {"role": "user", "content": prompt}], max_tokens=8192,
    )
    parsed = _json_object(msg.content or "")
    candidate = str(parsed.get("skill") or "").strip()
    changes = parsed.get("changes") or []
    if not isinstance(changes, list):
        changes = [str(changes)]
    if not candidate:
        return old_skill, [], "rewrite output was not valid JSON with a non-empty skill"
    return candidate + "\n", changes, ""

def persist_json(path, obj):
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)

def append_batch_summary(path, phase, batch_summary):
    """Append exactly one immutable batch section to the cumulative summary."""
    section = f"\n\n## {phase}\n\n{batch_summary.strip()}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(section)
    return open(path, encoding="utf-8").read()

def evaluate_test_suites(pool, suites, skill, schema_skill, cfg, out, stage):
    """Evaluate one skill pair on every configured held-out test suite."""
    stage_dir = os.path.join(out, "tests", stage)
    os.makedirs(stage_dir, exist_ok=True)
    summaries = {}
    for name, suite in suites.items():
        items = suite["items"]
        evaluator = evaluator_for(suite, cfg.get("exec_timeout", 120))
        jsonl_path = os.path.join(stage_dir, f"{name}_results.jsonl")
        completed = {}
        if os.path.exists(jsonl_path):
            with open(jsonl_path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        completed[str(row.get("id"))] = row
        pending = [item for item in items if str(item.get("id")) not in completed]
        if completed:
            print(
                f"[{stage}-test-{name}] resume completed={len(completed)} "
                f"pending={len(pending)}",
                flush=True,
            )

        def work(item):
            try:
                return rollout_one(
                    pool, item, skill, cfg, schema_skill,
                    temperature=VALIDATION_TEMPERATURE, evaluator=evaluator,
                )
            except Exception as exc:
                return {
                    **item, "hard": 0, "predicted_sql": "",
                    "error": f"inference error: {type(exc).__name__}: {exc}",
                    "fail_reason": f"inference error: {type(exc).__name__}: {exc}",
                    "tool_call_count": 0, "tool_call_counts": {},
                }

        def persist_result(result, _done, _total):
            with open(jsonl_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()

        new_rows = parallel(
            work, pending, cfg["rollout_workers"],
            f"{stage}-test-{name}", track_score=True,
            on_result=persist_result,
        )
        rows = list(completed.values()) + new_rows
        rows.sort(key=lambda row: str(row.get("id", "")))
        dataset_score = score(rows)
        errors = sum(str(row.get("error", "")).startswith("inference error:") for row in rows)
        schema_rows = [row for row in rows if "schema_hard" in row]
        schema_correct = sum(int(row.get("schema_hard", 0) or 0) for row in schema_rows)
        dataset_summary = {
            "dataset": name,
            "samples": len(rows),
            "correct": sum(int(row.get("hard", 0) or 0) for row in rows),
            "ex": dataset_score,
            "inference_errors": errors,
            "schema_accuracy_available": bool(schema_rows),
            "schema_evaluated_samples": len(schema_rows),
            "schema_correct": schema_correct if schema_rows else None,
            "schema_containment": schema_score(schema_rows) if schema_rows else None,
            "schema_table_recall": (
                sum(float(row.get("schema_table_recall", 0)) for row in schema_rows)
                / len(schema_rows) if schema_rows else None
            ),
            "schema_column_recall": (
                sum(float(row.get("schema_column_recall", 0)) for row in schema_rows)
                / len(schema_rows) if schema_rows else None
            ),
            "schema_table_precision": (
                sum(float(row.get("schema_table_precision", 0)) for row in schema_rows)
                / len(schema_rows) if schema_rows else None
            ),
            "schema_column_precision": (
                sum(float(row.get("schema_column_precision", 0)) for row in schema_rows)
                / len(schema_rows) if schema_rows else None
            ),
            "schema_quality": (
                schema_quality_score(schema_rows) if schema_rows else None
            ),
            "schema_avg_predicted_columns": (
                sum(float(row.get("schema_predicted_columns", 0)) for row in schema_rows)
                / len(schema_rows) if schema_rows else None
            ),
            "schema_accuracy_unavailable_reason": (
                None if schema_rows else "dataset does not provide gold SQL or gold schema"
            ),
            "tool_calls": sum(int(row.get("tool_call_count", 0) or 0) for row in rows),
        }
        summaries[name] = dataset_summary
        persist_json(os.path.join(stage_dir, f"{name}_results.json"), rows)
        print(
            f"[{stage}-test-{name}] EX={dataset_score:.4f} "
            f"correct={dataset_summary['correct']}/{len(rows)} errors={errors} "
            f"schema_accuracy={dataset_summary['schema_containment']}",
            flush=True,
        )
    persist_json(os.path.join(stage_dir, "summary.json"), summaries)
    return summaries

def run_training(config_path, overrides=None):
    global _PROGRESS_PATH
    cfg = yaml.safe_load(open(config_path, encoding="utf-8")); cfg.update(overrides or {})
    if not cfg.get("out_root"):
        cfg["out_root"] = os.path.abspath("outputs/run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out = cfg["out_root"]; os.makedirs(out, exist_ok=True); persist_json(os.path.join(out, "config.json"), cfg)
    _PROGRESS_PATH = os.path.join(out, "progress.json")
    cumulative_summary_path = os.path.join(out, "cumulative_summary.md")
    if os.path.exists(cumulative_summary_path):
        raise FileExistsError(f"refusing to reuse cumulative summary: {cumulative_summary_path}")
    with open(cumulative_summary_path, "w", encoding="utf-8") as f:
        f.write(CUMULATIVE_SUMMARY_HEADER)
    splits = load_splits(cfg["split_dir"]); pool = ModelPool(); skill = open(cfg["skill_init"], encoding="utf-8").read()
    schema_skill = open(cfg["schema_skill_init"], encoding="utf-8").read()
    test_sets_dir = cfg.get("test_sets_dir")
    if not test_sets_dir:
        raise ValueError(
            "test_sets_dir is required; build it with scripts/build_test_sets.py"
        )
    test_suites = load_test_suites_from_dir(test_sets_dir)
    random.Random(cfg["seed"]).shuffle(splits["train"])
    rollout = lambda item, sk=skill, ssk=schema_skill: rollout_one(
        pool, item, sk, cfg, ssk, temperature=VALIDATION_TEMPERATURE
    )
    print("[baseline] evaluating initial skill on validation split", flush=True)
    baseline_val = parallel(rollout, splits["val"], cfg["rollout_workers"], "baseline-val", track_score=True); best_score = score(baseline_val); best_skill = skill
    best_schema_score = schema_quality_score(baseline_val); best_schema_skill = schema_skill
    best_schema_containment = schema_score(baseline_val)
    best_schema_val_by_id = {
        str(row.get("id")): row for row in baseline_val
    }
    print(f"[baseline] validation EX={best_score:.4f} schema_quality={best_schema_score:.4f} schema_containment={best_schema_containment:.4f}", flush=True)
    print("[gate-policy] accept SQL candidates by EX and schema candidates by precision-aware quality when candidate >= current best", flush=True)
    batch_patience = int(cfg.get("batch_no_improvement_patience", 10) or 0)
    repeated_batch_attempts = cfg.get(
        "enable_repeated_batch_attempts", True
    )
    if repeated_batch_attempts:
        print(
            "[batch-policy] repeated attempts enabled; advance after train EX "
            "reaches 1.0"
            + (
                f" or after {batch_patience} consecutive attempt(s) without "
                "strict train EX improvement"
                if batch_patience > 0
                else "; no-improvement early advance is disabled"
            ),
            flush=True,
        )
    else:
        print(
            "[batch-policy] repeated attempts disabled; run exactly one "
            "rollout/repair/rewrite attempt, apply validation gate, then "
            "advance to the next batch",
            flush=True,
        )
    
    print("[baseline] evaluating initial skills on all configured test sets", flush=True)
    baseline_tests = evaluate_test_suites(
        pool, test_suites, best_skill, best_schema_skill, cfg, out, "baseline"
    )
    baseline_bird = os.path.join(out, "tests", "baseline", "bird_results.json")
    if os.path.exists(baseline_bird):
        shutil.copyfile(baseline_bird, os.path.join(out, "test_results_original.json"))
    history = []; schema_history = []
    for epoch in range(cfg["num_epochs"]):
        for step in range(math.ceil(len(splits["train"]) / cfg["batch_size"])):
            # Keep this exact batch fixed until a policy skill solves every
            # example.  Each new batch starts from the globally best policy,
            # not from a train-perfect candidate that lost on validation.
            batch = splits["train"][step*cfg["batch_size"]:(step+1)*cfg["batch_size"]]
            batch_phase = f"epoch-{epoch+1}-step-{step+1}"
            working_skill = best_skill
            attempt = 0
            batch_best_train_score = -1.0
            no_improvement_attempts = 0
            while True:
                attempt += 1
                phase = f"{batch_phase}-attempt-{attempt}"
                print(
                    f"[{phase}] begin fixed_batch_size={len(batch)} "
                    f"target_train_EX=1.0000",
                    flush=True,
                )
                train_rows = parallel(
                    lambda x: rollout_one(
                        pool, x, working_skill, cfg, best_schema_skill,
                        temperature=VALIDATION_TEMPERATURE,
                    ),
                    batch, cfg["rollout_workers"], phase + "-rollout",
                    track_score=True,
                )
                current_train_score = score(train_rows)
                incumbent_train_score = current_train_score

                # Schema linking deliberately keeps its original per-attempt
                # validation gate. It does not inherit the policy's train-100%
                # requirement.
                schema_train_rows = train_rows
                schema_failed = [
                    x for x in train_rows
                    if not x["schema_hard"]
                    or x.get("schema_column_precision", 1.0) < 0.5
                    or x.get("schema_size_ratio", 1.0) > 3.0
                ]
                schema_reflections = []
                schema_summary, schema_chunks = "", []
                schema_candidate = best_schema_skill
                schema_changes = []
                schema_rewrite_error = ""
                schema_action = "disabled"
                candidate_schema_score = best_schema_score
                schema_evolution_enabled = cfg.get(
                    "enable_schema_skill_evolution", True
                )
                print(
                    f"[{phase}-schema] quality="
                    f"{schema_quality_score(train_rows):.3f} containment="
                    f"{schema_score(train_rows):.3f} review={len(schema_failed)} "
                    f"evolution={'enabled' if schema_evolution_enabled else 'disabled'}",
                    flush=True,
                )
                if schema_evolution_enabled:
                    schema_reflections = parallel(
                        lambda x: reflect_schema_one(
                            pool, x, schema_skill, cfg
                        ),
                        schema_failed, cfg["reflect_workers"],
                        phase + "-schema-reflect",
                    )
                    schema_summary, schema_chunks = (
                        summarize_schema_trajectories(pool, train_rows)
                    )
                    schema_cumulative_path = os.path.join(
                        out, "schema_cumulative_summary.md"
                    )
                    if not os.path.exists(schema_cumulative_path):
                        open(
                            schema_cumulative_path, "w", encoding="utf-8"
                        ).write(
                            "# Cumulative Schema-Linking Training Summary\n"
                        )
                    schema_cumulative = append_batch_summary(
                        schema_cumulative_path, phase, schema_summary
                    )
                    (
                        schema_candidate,
                        schema_changes,
                        schema_rewrite_error,
                    ) = rewrite_schema_skill(
                        pool,
                        schema_reflections,
                        best_schema_skill,
                        schema_cumulative,
                    )
                    schema_action = "skip"
                    if (
                        not schema_rewrite_error
                        and schema_candidate.strip()
                        != best_schema_skill.strip()
                    ):
                        schema_val = parallel(
                            lambda x: schema_rollout_one(
                                pool,
                                x,
                                schema_candidate,
                                cfg,
                                temperature=VALIDATION_TEMPERATURE,
                            ),
                            splits["val"],
                            cfg["rollout_workers"],
                            phase + "-schema-candidate-val",
                        )
                        candidate_schema_score = schema_quality_score(schema_val)
                        if candidate_schema_score >= best_schema_score:
                            schema_skill = best_schema_skill = schema_candidate
                            best_schema_score = candidate_schema_score
                            best_schema_containment = schema_score(schema_val)
                            best_schema_val_by_id = {
                                str(row.get("id")): row
                                for row in schema_val
                            }
                            schema_action = "accept"
                    else:
                        schema_action = "reject"
                print(
                    f"[{phase}-schema-gate] candidate="
                    f"{candidate_schema_score:.4f} best="
                    f"{best_schema_score:.4f} "
                    f"decision={schema_action.upper()}",
                    flush=True,
                )
                if schema_action == "accept":
                    # Keep the policy train comparison fair: both incumbent and
                    # candidate must run with the newly selected schema skill.
                    train_rows = parallel(
                        lambda x: rollout_one(
                            pool, x, working_skill, cfg, best_schema_skill,
                            temperature=VALIDATION_TEMPERATURE,
                        ),
                        batch, cfg["rollout_workers"],
                        phase + "-incumbent-train-after-schema",
                        track_score=True,
                    )
                    current_train_score = score(train_rows)
                    incumbent_train_score = current_train_score
                failed = [x for x in train_rows if not x["hard"]]
                print(f"[{phase}-rollout] EX={score(train_rows):.3f} correct={len(train_rows)-len(failed)} failed={len(failed)}", flush=True)
                for row in failed[:3]:
                    question = str(row.get("question", "")).replace("\n", " ")[:140]
                    predicted = str(row.get("predicted_sql", "")).replace("\n", " ")[:180]
                    reason = str(row.get("fail_reason", "")).replace("\n", " ")[:160]
                    print(f"[{phase}-failure] id={row.get('id')} question={question!r}", flush=True)
                    print(f"[{phase}-failure] predicted={predicted!r} reason={reason!r}", flush=True)
                repairs = []
                if cfg.get("enable_guided_sql_repair", True) and failed:
                    print(
                        f"[{phase}-repair] starting guided multi-round repair "
                        f"for {len(failed)} failed trace(s)",
                        flush=True,
                    )
                    repairs = parallel(
                        lambda x: guided_repair_one(pool, x, cfg),
                        failed,
                        int(cfg.get("repair_workers", cfg["reflect_workers"])),
                        phase + "-repair",
                    )
                    repairs_by_id = {
                        str(repair.get("sample_id")): repair
                        for repair in repairs
                    }
                    repair_summaries_by_id = {
                        sample_id: {
                            "repair_success": repair.get("repair_success"),
                            "repaired_sql": repair.get("repaired_sql", ""),
                            "repair_rounds": repair.get("repair_rounds", 0),
                            "repair_error": repair.get("repair_error", ""),
                            "repair_reusable_lessons": repair.get(
                                "repair_reusable_lessons", []
                            ),
                            "repair_tool_call_count": repair.get(
                                "repair_tool_call_count", 0
                            ),
                            "repair_tool_call_counts": repair.get(
                                "repair_tool_call_counts", {}
                            ),
                        }
                        for sample_id, repair in repairs_by_id.items()
                    }
                    # Keep full repair/guide trajectories only in
                    # guided_repairs.json. Train results and trajectory
                    # summaries receive a compact repair outcome, avoiding
                    # duplicate trajectory logging.
                    train_rows = [
                        {
                            **row,
                            **repair_summaries_by_id.get(
                                str(row.get("id")), {}
                            ),
                        }
                        for row in train_rows
                    ]
                    failed = [
                        {
                            **row,
                            **repairs_by_id.get(str(row.get("id")), {}),
                        }
                        for row in failed
                    ]
                    repair_successes = sum(
                        int(repair.get("repair_success", False))
                        for repair in repairs
                    )
                    print(
                        f"[{phase}-repair] success={repair_successes}/"
                        f"{len(repairs)}",
                        flush=True,
                    )
                reflections = parallel(
                    lambda x: distill_repair_trace_one(
                        pool, x, working_skill, cfg
                    ),
                    failed, cfg["reflect_workers"], phase + "-reflect",
                )
                useful = [r for r in reflections if r.get("suggestions")]
                print(f"[{phase}-reflect] useful={len(useful)}/{len(reflections)}", flush=True)
                for reflection in useful[:5]:
                    diagnosis = str(reflection.get("diagnosis", "")).replace("\n", " ")[:220]
                    print(f"[{phase}-reflect] sample={reflection.get('sample_id')} diagnosis={diagnosis!r}", flush=True)
                    for suggestion in reflection.get("suggestions", [])[:3]:
                        print(f"[{phase}-reflect]   suggestion: {str(suggestion)[:300]}", flush=True)
                print(f"[{phase}-summary] summarizing all {len(train_rows)} correct/error trajectories", flush=True)
                trajectory_summary, chunk_summaries = summarize_trajectories(pool, train_rows)
                print(f"[{phase}-summary] complete chunks={len(chunk_summaries)} chars={len(trajectory_summary)}", flush=True)
                cumulative_summary = append_batch_summary(
                    cumulative_summary_path, phase, trajectory_summary
                )
                print(
                    f"[{phase}-summary] appended cumulative chars={len(cumulative_summary)}",
                    flush=True,
                )
                print(f"[{phase}-rewrite] rewriting from cumulative summary + old skill + current reflections", flush=True)
                candidate, changes, rewrite_error = rewrite_skill(
                    pool, reflections, working_skill, cumulative_summary
                )
                rewrite_ok = not rewrite_error and candidate.strip() != working_skill.strip()
                if rewrite_ok:
                    print(f"[{phase}-rewrite] proposed {len(changes)} change(s):", flush=True)
                    for idx, change in enumerate(changes, 1):
                        print(f"[{phase}-rewrite]   {idx}. {str(change)[:400]}", flush=True)
                else:
                    reason = rewrite_error or "rewritten skill is unchanged"
                    print(f"[{phase}-rewrite] no usable skill rewrite; gate skipped ({reason})", flush=True)
                action = "skip"
                candidate_train_score = current_train_score
                candidate_score = None
                incumbent_val_score = None
                candidate_train_rows = []
                if rewrite_ok:
                    train_schema_by_id = {
                        str(row.get("id")): row for row in train_rows
                    }
                    print(
                        f"[{phase}-train-gate] evaluating candidate on the same fixed "
                        f"batch; incumbent_train_EX={current_train_score:.4f}",
                        flush=True,
                    )
                    candidate_train_rows = parallel(
                        lambda x: rollout_one(
                            pool, x, candidate, cfg, best_schema_skill,
                            temperature=VALIDATION_TEMPERATURE,
                            cached_schema_result=train_schema_by_id[
                                str(x.get("id"))
                            ],
                        ),
                        batch, cfg["rollout_workers"],
                        phase + "-candidate-train", track_score=True,
                    )
                    candidate_train_score = score(candidate_train_rows)
                    if candidate_train_score >= current_train_score:
                        working_skill = candidate
                        train_rows = candidate_train_rows
                        current_train_score = candidate_train_score
                        action = "train_accept"
                    else:
                        action = "train_reject"
                    print(
                        f"[{phase}-train-gate] candidate_train_EX="
                        f"{candidate_train_score:.4f} working_train_EX="
                        f"{current_train_score:.4f} decision={action.upper()}",
                        flush=True,
                    )

                # Equal-score rewrites may still become the working skill, but
                # patience measures strict EX improvement so a batch cannot loop
                # forever through score-equivalent rewrites.
                if current_train_score > batch_best_train_score + 1e-12:
                    batch_best_train_score = current_train_score
                    no_improvement_attempts = 0
                    print(
                        f"[{phase}-batch-progress] new_best_train_EX="
                        f"{batch_best_train_score:.4f}; patience reset",
                        flush=True,
                    )
                else:
                    no_improvement_attempts += 1
                    print(
                        f"[{phase}-batch-progress] no strict EX improvement "
                        f"{no_improvement_attempts}/{batch_patience if batch_patience > 0 else 'disabled'} "
                        f"best_train_EX={batch_best_train_score:.4f}",
                        flush=True,
                    )
                train_perfect = current_train_score >= 1.0
                patience_exhausted = (
                    batch_patience > 0
                    and no_improvement_attempts >= batch_patience
                )
                batch_exit_reason = (
                    "single_attempt"
                    if not repeated_batch_attempts
                    else "train_perfect" if train_perfect
                    else "no_improvement_patience" if patience_exhausted
                    else None
                )

                # Validate when repeated mode reaches its normal stopping
                # condition, or immediately after the sole attempt in
                # single-attempt mode. The global gate still prevents a local
                # rewrite from causing a regression before the next batch.
                if batch_exit_reason:
                    print(
                        f"[{phase}-val-gate] batch_exit_reason={batch_exit_reason} "
                        f"train_EX={current_train_score:.4f}; re-evaluating the "
                        f"working policy and previous best policy with the same "
                        f"current best schema skill",
                        flush=True,
                    )
                    incumbent_val_rows = parallel(
                        lambda x: rollout_one(
                            pool, x, best_skill, cfg, best_schema_skill,
                            temperature=VALIDATION_TEMPERATURE,
                            cached_schema_result=best_schema_val_by_id[
                                str(x.get("id"))
                            ],
                        ),
                        splits["val"], cfg["rollout_workers"],
                        phase + "-previous-best-val", track_score=True,
                    )
                    incumbent_val_score = score(incumbent_val_rows)
                    val_rows = parallel(
                        lambda x: rollout_one(
                            pool, x, working_skill, cfg, best_schema_skill,
                            temperature=VALIDATION_TEMPERATURE,
                            cached_schema_result=best_schema_val_by_id[
                                str(x.get("id"))
                            ],
                        ),
                        splits["val"], cfg["rollout_workers"],
                        phase + "-candidate-val", track_score=True,
                    )
                    candidate_score = score(val_rows)
                    if candidate_score >= incumbent_val_score:
                        skill = best_skill = working_skill
                        best_score = candidate_score
                        action = "val_accept"
                    else:
                        skill = best_skill
                        best_score = incumbent_val_score
                        action = "val_reject"
                    print(
                        f"[{phase}-val-gate] candidate_val_EX={candidate_score:.4f} "
                        f"previous_best_val_EX={incumbent_val_score:.4f} "
                        f"retained_val_EX={best_score:.4f} "
                        f"decision={action.upper()}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{phase}-val-gate] skipped: working_train_EX="
                        f"{current_train_score:.4f}, no_improvement="
                        f"{no_improvement_attempts}/{batch_patience if batch_patience > 0 else 'disabled'}",
                        flush=True,
                    )
    
                step_dir = os.path.join(
                    out, "steps",
                    f"epoch_{epoch+1:02d}_step_{step+1:04d}_attempt_{attempt:04d}",
                ); os.makedirs(step_dir, exist_ok=True)
                persist_json(os.path.join(step_dir, "train_results.json"), train_rows); persist_json(os.path.join(step_dir, "reflections.json"), reflections)
                persist_json(os.path.join(step_dir, "guided_repairs.json"), repairs)
                if candidate_train_rows:
                    persist_json(
                        os.path.join(step_dir, "candidate_train_results.json"),
                        candidate_train_rows,
                    )
                persist_json(os.path.join(step_dir, "trajectory_summary.json"), {"batch_summary": trajectory_summary, "chunk_summaries": chunk_summaries, "cumulative_summary_path": os.path.abspath(cumulative_summary_path)})
                open(os.path.join(step_dir, "candidate_skill.md"), "w", encoding="utf-8").write(candidate)
                open(os.path.join(step_dir, "working_skill.md"), "w", encoding="utf-8").write(working_skill)
                persist_json(os.path.join(step_dir, "decision.json"), {
                    "changes": changes,
                    "rewrite_error": rewrite_error,
                    "incumbent_train_score": incumbent_train_score,
                    "candidate_train_score": candidate_train_score,
                    "working_train_score": current_train_score,
                    "train_target": 1.0,
                    "batch_best_train_score": batch_best_train_score,
                    "no_improvement_attempts": no_improvement_attempts,
                    "batch_no_improvement_patience": batch_patience,
                    "repeated_batch_attempts_enabled": (
                        repeated_batch_attempts
                    ),
                    "batch_exit_reason": batch_exit_reason,
                    "candidate_comparison_shared_schema_cache": True,
                    "candidate_val_score": candidate_score,
                    "previous_best_val_score": incumbent_val_score,
                    "best_val_score": best_score,
                    "gate_rule": (
                        "candidate_train_score >= working_train_score; "
                        "run validation after one attempt when repeated mode "
                        "is disabled, otherwise when working_train_score == "
                        "1.0 or no-improvement patience is exhausted; "
                        "then accept when candidate_val_score >= previous_best_val_score"
                    ),
                    "action": action,
                })
                persist_json(os.path.join(step_dir, "schema_train_results.json"), schema_train_rows)
                persist_json(os.path.join(step_dir, "schema_reflections.json"), schema_reflections)
                persist_json(os.path.join(step_dir, "schema_trajectory_summary.json"), {"batch_summary": schema_summary, "chunk_summaries": schema_chunks})
                open(os.path.join(step_dir, "candidate_schema_skill.md"), "w", encoding="utf-8").write(schema_candidate)
                persist_json(os.path.join(step_dir, "schema_decision.json"), {
                    "evolution_enabled": schema_evolution_enabled,
                    "changes": schema_changes,
                    "rewrite_error": schema_rewrite_error,
                    "candidate_quality": candidate_schema_score,
                    "best_quality": best_schema_score,
                    "best_containment": best_schema_containment,
                    "gate_rule": (
                        "candidate_schema_quality >= previous_best_schema_quality"
                        if schema_evolution_enabled
                        else "disabled by enable_schema_skill_evolution=false"
                    ),
                    "action": schema_action,
                })
                history.append({
                    "epoch": epoch+1,
                    "step": step+1,
                    "attempt": attempt,
                    "repeated_batch_attempts_enabled": (
                        repeated_batch_attempts
                    ),
                    "train_ex": current_train_score,
                    "failures": sum(not x["hard"] for x in train_rows),
                    "candidate_train_ex": candidate_train_score,
                    "candidate_val_ex": candidate_score,
                    "best_val_ex": best_score,
                    "no_improvement_attempts": no_improvement_attempts,
                    "batch_exit_reason": batch_exit_reason,
                    "action": action,
                })
                schema_history.append({
                    "epoch": epoch+1,
                    "step": step+1,
                    "attempt": attempt,
                    "evolution_enabled": schema_evolution_enabled,
                    "train_quality": schema_quality_score(schema_train_rows),
                    "train_containment": schema_score(schema_train_rows),
                    "reviewed": len(schema_failed),
                    "candidate_val_quality": candidate_schema_score,
                    "best_val_quality": best_schema_score,
                    "best_val_containment": best_schema_containment,
                    "action": schema_action,
                })
                persist_json(os.path.join(out, "history.json"), history); open(os.path.join(out, "best_skill.md"), "w", encoding="utf-8").write(best_skill)
                persist_json(os.path.join(out, "schema_history.json"), schema_history); open(os.path.join(out, "best_schema_skill.md"), "w", encoding="utf-8").write(best_schema_skill)
                print(
                    f"epoch={epoch+1} step={step+1} attempt={attempt} "
                    f"train_ex={current_train_score:.3f} "
                    f"val={candidate_score if candidate_score is not None else 'SKIPPED'} "
                    f"{action}",
                    flush=True,
                )
                if batch_exit_reason:
                    print(
                        f"[{phase}-batch-complete] advancing to next batch; "
                        f"reason={batch_exit_reason} train_EX="
                        f"{current_train_score:.4f}",
                        flush=True,
                    )
                    break
    print("[final-test] evaluating best skills on all configured test sets", flush=True)
    final_tests = evaluate_test_suites(
        pool, test_suites, best_skill, best_schema_skill, cfg, out, "final"
    )
    final_bird_path = os.path.join(out, "tests", "final", "bird_results.json")
    if os.path.exists(final_bird_path):
        shutil.copyfile(final_bird_path, os.path.join(out, "test_results.json"))
    bird_final = final_tests.get("bird", {})
    summary = {
        "best_val_ex": best_score,
        "best_val_schema_quality": best_schema_score,
        "best_val_schema_containment": best_schema_containment,
        "test_ex": bird_final.get("ex"),
        "test_schema_containment": bird_final.get("schema_containment"),
        "baseline_tests": baseline_tests,
        "final_tests": final_tests,
        "steps": len(history),
    }
    persist_json(os.path.join(out, "summary.json"), summary)
    persist_json(os.path.join(out, "schema_summary.json"), {"best_val_schema_quality": best_schema_score, "best_val_schema_containment": best_schema_containment, "test_schema_containment": bird_final.get("schema_containment"), "steps": len(schema_history)})
    persist_json(os.path.join(out, "RUN_COMPLETE.json"), {
        "status": "complete",
        "completed_at": datetime.datetime.now().astimezone().isoformat(),
        "summary": os.path.abspath(os.path.join(out, "summary.json")),
    })
    persist_json(_PROGRESS_PATH, {
        "status": "complete",
        "phase": "all",
        "updated_at": datetime.datetime.now().astimezone().isoformat(),
        "completion_marker": os.path.abspath(os.path.join(out, "RUN_COMPLETE.json")),
    })
    print(summary)
