from __future__ import annotations
import argparse, json, os, random
from .evaluator import execute

ROOT = "/public/home/yaozhiming/yzm/Text2SQL/data"
SOURCES = {
    "train": (f"{ROOT}/bird/bird23-train-filtered/data/train-00000-of-00001.jsonl", f"{ROOT}/bird/train/train_databases", "bird"),
    # Validation is deliberately sampled from SynSQL instead of the BIRD
    # training pool, so skill gates measure transfer to unseen synthetic
    # schemas rather than another slice of the training distribution.
    "val": (f"{ROOT}/bird/preprocessed_json_files/train_synsql_complex_1000_harder.json", f"{ROOT}/SynSQL-2.5M/databases", "syn"),
    "test": (f"{ROOT}/bird/dev_20240627/dev.json", f"{ROOT}/bird/dev_20240627/dev_databases", "bird"),
}

def load(path):
    text = open(path, encoding="utf-8").read().strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else data.get("data", list(data.values()))
    except json.JSONDecodeError:
        return [json.loads(x) for x in text.splitlines() if x.strip()]

def sample(name, n, seed):
    src, dbroot, kind = SOURCES[name]
    print(f"[split:{name}] loading {src}", flush=True)
    rows = load(src); random.Random(seed).shuffle(rows); out = []
    print(f"[split:{name}] pool={len(rows)} requested={n}; validating gold SQL...", flush=True)
    for scanned, raw in enumerate(rows, 1):
        dbid, q = raw.get("db_id"), raw.get("question")
        sql = raw.get("sql") or raw.get("SQL") or raw.get("query")
        if not (dbid and q and sql): continue
        db = os.path.join(dbroot, str(dbid), f"{dbid}.sqlite")
        if not os.path.exists(db) or not execute(db, str(sql))[0]: continue
        evidence = raw.get("external_knowledge") if kind == "syn" else raw.get("evidence")
        out.append({"id": f"{name}_{len(out):05d}", "db_id": str(dbid), "question": str(q).strip(),
                    "evidence": str(evidence or "").strip(), "gold_sql": str(sql).strip(), "db_path": os.path.abspath(db)})
        if len(out) % 100 == 0 or len(out) >= n:
            print(f"[split:{name}] kept={len(out)}/{n} scanned={scanned}/{len(rows)}", flush=True)
        if len(out) >= n: break
    print(f"[split:{name}] complete kept={len(out)}/{n}", flush=True)
    return out

def sample_bird_train_val(train_n, val_n, seed):
    """Build BIRD training and independently sampled SynSQL validation sets."""
    train = sample("train", train_n, seed)
    val = sample("val", val_n, seed + 1)
    print(
        f"[split:train+val] complete train={len(train)} source=bird "
        f"val={len(val)} source=synsql",
        flush=True,
    )
    return train, val

def build_cli():
    p = argparse.ArgumentParser(); p.add_argument("--out_dir", default="data/text2sql_split")
    p.add_argument("--train_n", type=int, default=1000); p.add_argument("--val_n", type=int, default=200)
    # Accepted for compatibility with older launch commands, but test data is
    # now owned exclusively by scripts/build_test_sets.py.
    p.add_argument("--test_n", type=int, help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    train, val = sample_bird_train_val(a.train_n, a.val_n, a.seed)
    for name, items in (("train", train), ("val", val)):
        d = os.path.join(a.out_dir, name); os.makedirs(d, exist_ok=True)
        json.dump(items, open(os.path.join(d, "items.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[split:{name}] wrote {len(items)} items to {d}/items.json", flush=True)

def load_splits(root):
    return {name: json.load(open(os.path.join(root, name, "items.json"), encoding="utf-8")) for name in ("train", "val")}
