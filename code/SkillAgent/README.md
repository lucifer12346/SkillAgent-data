# SkillAgent: Repair-Guided Text-to-SQL Skill Optimization

SkillAgent is an experimental Text-to-SQL training framework that improves a
reusable policy prompt (a *Policy Skill*) from execution failures. The current
version combines complete-schema SQL generation, optional schema-linking skill
evolution, a privileged GT Guide Agent, a multi-round SQL Repair Agent, and
train/validation gates.

## 1. What the current method does

For every training batch, the framework:

1. Runs a schema-linking agent to identify high-relevance tables and columns.
2. Gives the policy model both the high-relevance schema and the complete DDL.
3. Lets the policy model call the read-only `execute_sql` tool over multiple
   turns and validates the final SQL syntax automatically.
4. Sends every incorrect SQL trace to a privileged GT Guide Agent.
5. Lets a Repair Agent, which cannot see the gold SQL, repair the query through
   multiple rounds of guide feedback and database interaction.
6. Distills each completed repair trajectory into transferable Text-to-SQL
   decision procedures.
7. Rewrites the complete Policy Skill from the previous skill, cumulative batch
   summaries, and the current repair-trace distillations.
8. Compares the candidate and incumbent on the same batch with identical cached
   schema-linking results, then applies a validation gate.
9. Evaluates the initial and final skills on BIRD, EHRSQL, and Spider2.0 using
   the evaluator configured for each dataset.

Repair success does **not** overwrite the original rollout score. Repair traces
are privileged training evidence; the Policy Skill must still improve the
policy model's independent SQL generation.

## 2. Main components

| Path | Responsibility |
|---|---|
| `skillagent/agents.py` | Schema linking, policy rollout, tool loop, GT-guided SQL repair, repair-trace distillation |
| `skillagent/trainer.py` | Baseline tests, batch loop, skill rewriting, train/validation gates, final tests and persistence |
| `skillagent/tools.py` | Read-only `execute_sql`, table-formatted observations, internal final SQL validation |
| `skillagent/evaluator.py` | SQLite execution and schema metrics |
| `skillagent/benchmarks.py` | Dataset-specific BIRD, EHRSQL, and Spider2.0 evaluation dispatch |
| `skillagent/data.py` | Train/validation split loading |
| `skillagent/model.py` | OpenAI-compatible endpoint pool, failover, retry, timeout and temperature handling |
| `skillagent/prompts/` | Initial Policy Skill and initial Schema-Linking Skill |
| `scripts/build_text2sql_data.sh` | Stage 1: construct training, validation and test data |
| `scripts/train_text2sql.sh` | Stage 2: train from an already-built run directory |
| `scripts/run_text2sql.sh` | Convenience wrapper for the complete workflow |
| `scripts/evaluate_skills.py` | Evaluate explicitly selected skill files without training |
| `scripts/check_run_status.py` | Read progress, ETA and completion state |

## 3. Requirements

- Linux with Python 3.10+ (the project has been run with Python 3.12)
- SQLite databases for the selected datasets
- An OpenAI-compatible chat-completions endpoint with function calling
- Python packages listed in `requirements.txt`

Install dependencies:

```bash
cd /path/to/SkillAgent
python -m pip install -r requirements.txt
```

Configure the local model service:

```bash
export QWEN_CHAT_BASE_URL="http://127.0.0.1:8000/v1"
export QWEN_CHAT_API_KEY="EMPTY"
export QWEN_CHAT_MODEL="qwen3.6-35b-a3b"
export QWEN_CHAT_TEMPERATURE="0"
export QWEN_CHAT_TIMEOUT="600"
export QWEN_CHAT_RETRY_ROUNDS="2"
export QWEN_CHAT_RETRY_BACKOFF="2"
```

Multiple endpoints may be supplied as a comma-separated list in
`QWEN_CHAT_BASE_URL`.

## 4. Data layout

A prepared run directory has the following form:

```text
runs/run_NAME/
└── data/
    ├── DATA_READY or FILTERED_DATA_READY
    ├── text2sql_split/
    │   ├── train/items.json
    │   └── val/items.json
    └── test_sets/
        ├── manifest.json
        ├── bird/items.json
        ├── ehrsql/items.json
        └── spider2.0/items.json
```

The three held-out test sets are built only by the test-set builder and keep
their own evaluation logic. Do not duplicate BIRD construction in the generic
train/validation split builder.

Dataset roots and Spider2.0 evaluation metadata are configured in
`configs/default.yaml`. Update the absolute paths before moving the project to
another server.

## 5. Recommended two-stage execution

### Stage 1: build data

```bash
cd /path/to/SkillAgent
RUN_ROOT=runs/run_NAME bash scripts/build_text2sql_data.sh
```

Inspect the generated `data/` directory and readiness marker before training.

### Stage 2: train

```bash
bash scripts/train_text2sql.sh runs/run_NAME
```

The training script accepts either `DATA_READY` or `FILTERED_DATA_READY`.
Keeping data construction separate prevents an existing run's prepared splits
from being rebuilt accidentally.

The convenience wrapper can be used when both stages should be run together:

```bash
bash scripts/run_text2sql.sh
```

Review that script's run name and dataset paths before launching a long job.

## 6. Important switches

The current recommended defaults in `configs/default.yaml` are:

```yaml
enable_guided_sql_repair: true
enable_schema_skill_evolution: false
enable_repeated_batch_attempts: false

max_repair_rounds: 6
repair_workers: 16
batch_no_improvement_patience: 10
```

### `enable_guided_sql_repair`

When enabled, every incorrect rollout enters the GT Guide / Repair Agent loop,
and Policy Skill feedback is produced by repair-trace distillation instead of
the legacy one-shot reflection.

### `enable_schema_skill_evolution`

When disabled, schema linking and schema metrics still run, but schema
reflection, schema-skill rewriting, and schema candidate validation are
skipped. The initial schema-linking skill remains fixed.

### `enable_repeated_batch_attempts`

When disabled, each batch receives exactly one rollout/repair/rewrite attempt,
followed by the validation gate, then training advances to the next batch.

When enabled, the same batch is retried until train EX reaches 1.0 or strict
train EX has not improved for `batch_no_improvement_patience` attempts.

## 7. Skill gates

The train gate evaluates the candidate and incumbent on the same batch and
reuses an identical cached schema-linking result:

```text
candidate_train_EX >= incumbent_train_EX
```

Equal-score rewrites are allowed. The validation gate then compares a fresh
candidate and incumbent evaluation at temperature 0:

```text
candidate_val_EX >= previous_best_val_EX
```

Only a candidate that passes the validation gate becomes the global best
Policy Skill.

## 8. Outputs

The main output directory contains:

```text
output/
├── config.json
├── progress.json
├── history.json
├── schema_history.json
├── best_skill.md
├── best_schema_skill.md
├── cumulative_summary.md
├── schema_cumulative_summary.md          # only when schema evolution runs
├── tests/
│   ├── baseline/
│   └── final/
├── steps/
└── RUN_COMPLETE.json                     # written only after full completion
```

Each step directory contains:

| File | Contents |
|---|---|
| `train_results.json` | Results for the final working policy of this attempt |
| `candidate_train_results.json` | Candidate policy rollout results |
| `guided_repairs.json` | Full Guide Agent and Repair Agent trajectories |
| `reflections.json` | Compact repair-trace distillations used for skill rewriting |
| `trajectory_summary.json` | Batch trajectory summary and cumulative-summary path |
| `candidate_skill.md` | Rewritten Policy Skill candidate |
| `working_skill.md` | Skill retained by the train gate |
| `decision.json` | Train/validation scores, gate decision and exit reason |
| `schema_*.json` / `candidate_schema_skill.md` | Schema-linking evidence and decision |

Full repair trajectories are stored only in `guided_repairs.json`; other files
use compact repair fields to avoid duplicate logging.

One implementation detail matters during analysis: after a candidate passes the
train gate, `train_results.json` represents the accepted working result. Use
`decision.json` for the recorded incumbent and candidate aggregate scores.

## 9. Progress and completion

Use:

```bash
python scripts/check_run_status.py runs/run_NAME
```

`progress.json` reports phase, completed items, rate, elapsed time, ETA, and the
estimated finish time. A run is fully complete only when
`output/RUN_COMPLETE.json` exists.

## 10. Evaluating selected skills

Use `scripts/evaluate_skills.py` to test explicitly selected Policy and Schema
Skill files without starting training. Check its `--help` output for the current
CLI options:

```bash
python scripts/evaluate_skills.py --help
```

Validation and held-out evaluation use temperature 0.

## 11. Tool and trajectory accounting

The model-facing tool surface contains only `execute_sql`. It accepts read-only
`SELECT`, `WITH`, `PRAGMA`, and `EXPLAIN QUERY PLAN` statements. Internal
`validate_sql` is invoked automatically and is not exposed to the model.

Tool calls are recorded separately for schema linking, SQL generation, repair,
and legacy reflection fields. Schema-linking and SQL-generation trajectories
are also persisted separately rather than concatenated.

## 12. Common issues

### `RuntimeError: all model endpoints failed: Request timed out`

Check endpoint health, `QWEN_CHAT_BASE_URL`, worker counts, and
`QWEN_CHAT_TIMEOUT`. Too many rollout, repair, and reflection workers can
overload a local serving cluster. Reduce concurrency before increasing timeout.

### Training appears stuck after a completed phase

Inspect `progress.json`, the run log, and the latest step directory. Calls that
summarize trajectories or rewrite a skill do not have item-level progress, so a
long model request may follow a phase marked `phase_complete`.

### Repair succeeds but train EX barely improves

Repair uses privileged, sample-specific, multi-round feedback. The deployable
Policy Skill contains only generalized instructions. Track both repair success
and repair-transfer rate; a successful repaired SQL does not imply that the
rewritten Policy Skill will reproduce it independently.

### A running job ignores a newly added switch

Configuration and Python code are loaded when the process starts. Restart in a
new run directory to apply new switches; existing processes do not hot reload.

## 13. Packaging policy

Source distributions should include code, prompts, configs, scripts,
`requirements.txt`, and this README. Exclude `runs/`, generated outputs, logs,
database files, dataset copies, `__pycache__/`, and `.pyc` files. Dataset and
model credentials must be supplied separately.
