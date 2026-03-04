import json
import os
import shlex
import subprocess
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# ---------- Helpers to load OLMES outputs ----------

def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def find_run_dirs(base: Path) -> List[Path]:
    """Find directories that look like OLMES output dirs (contain metrics.json)."""
    if not base.exists():
        return []
    hits = []
    for p in base.rglob("metrics.json"):
        hits.append(p.parent)
    # newest first
    hits.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return hits

def find_task_files(out_dir: Path, suffix: str) -> List[Path]:
    """Find task-*-<suffix> files (e.g., task-000-...-predictions.jsonl)."""
    return sorted(out_dir.glob(f"task-*-{suffix}"))

def load_summary(out_dir: Path) -> Tuple[Optional[dict], Optional[pd.DataFrame]]:
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        return None, None

    metrics = _read_json(metrics_path)
    tasks = metrics.get("tasks", [])
    if tasks:
        rows = []
        for t in tasks:
            m = t.get("metrics", {}) or {}
            row = {
                "task": t.get("alias") or t.get("task_name"),
                "primary_score": m.get("primary_score"),
                **{k: v for k, v in m.items() if k != "extra_metrics"},
            }
            extra = (m.get("extra_metrics") or {})
            for ek, ev in extra.items():
                row[f"extra.{ek}"] = ev
            row["num_instances"] = t.get("num_instances")
            row["processing_time_sec"] = t.get("processing_time")
            rows.append(row)
        df = pd.DataFrame(rows)
    else:
        df = None
    return metrics, df

def load_predictions(out_dir: Path) -> pd.DataFrame:
    pred_files = find_task_files(out_dir, "predictions.jsonl")
    all_rows = []
    for pf in pred_files:
        # filename pattern: task-000-gsm8k-predictions.jsonl
        parts = pf.name.split("-")
        task_alias = "unknown"
        if len(parts) >= 4:
            task_alias = parts[2]  # e.g., gsm8k
        for r in _read_jsonl(pf):
            mo = (r.get("model_output") or [{}])[0] or {}
            metrics = r.get("metrics") or {}
            all_rows.append({
                "task": task_alias,
                "doc_id": r.get("doc_id"),
                "native_id": r.get("native_id"),
                "gold": r.get("label"),
                "model_answer": mo.get("model_answer"),
                "continuation_raw": mo.get("continuation_raw") or mo.get("continuation"),
                "exact_match": metrics.get("exact_match"),
                "num_tokens": metrics.get("num_tokens"),
                "max_tokens_reached": metrics.get("max_tokens_reached"),
                "sum_logits": mo.get("sum_logits"),
            })
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    # Sort: errors first, then max_tokens_reached
    df["is_error"] = (df["exact_match"] == 0.0)
    df = df.sort_values(by=["is_error", "max_tokens_reached", "num_tokens"], ascending=[False, False, False])
    return df

def load_recorded_inputs(out_dir: Path) -> Dict[Tuple[str, int], dict]:
    """Map (task, doc_id) -> recorded input doc (question/prompt) if available."""
    rec_files = find_task_files(out_dir, "recorded-inputs.jsonl")
    mapping = {}
    for rf in rec_files:
        parts = rf.name.split("-")
        task_alias = "unknown"
        if len(parts) >= 4:
            task_alias = parts[2]
        for r in _read_jsonl(rf):
            doc_id = r.get("doc_id")
            if doc_id is None:
                continue
            mapping[(task_alias, int(doc_id))] = r
    return mapping


# ---------- Runner ----------

@dataclass
class RunConfig:
    model: str
    model_type: str
    task: str
    limit: str
    num_shots: int
    output_dir: str
    cache_dir: str
    batch_size: int
    extra_args: str
    save_raw_requests: bool


def run_olmes(cfg: RunConfig) -> int:
    cmd = [
        "olmes",
        "--model", cfg.model,
        "--model-type", cfg.model_type,
        "--task", cfg.task,
        "--limit", str(cfg.limit),
        "--num-shots", str(cfg.num_shots),
        "--batch-size", str(cfg.batch_size),
        "--output-dir", cfg.output_dir,
        "--cached-output-dir", cfg.cache_dir,
    ]
    if cfg.save_raw_requests:
        cmd += ["--save-raw-requests"]
    if cfg.extra_args.strip():
        cmd += shlex.split(cfg.extra_args.strip())

    st.code(" ".join(shlex.quote(c) for c in cmd), language="bash")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    log_box = st.empty()
    lines = []
    while True:
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            lines.append(line.rstrip("\n"))
            # keep last ~200 lines in UI
            lines = lines[-200:]
            log_box.text("\n".join(lines))
        if proc.poll() is not None:
            break
        time.sleep(0.02)

    return int(proc.returncode or 0)


# ---------- UI ----------

st.set_page_config(page_title="OLMES UI", layout="wide")

st.title("OLMES UI – Task auswählen → ausführen → Ergebnisse interaktiv ansehen")

base_work = Path(os.environ.get("WORKDIR", "/work"))
default_out_base = base_work / "out"
default_cache = base_work / "cache"

mode = st.sidebar.radio("Modus", ["Run Evaluation", "Open Existing Results"], index=0)

if mode == "Run Evaluation":
    st.sidebar.subheader("Run-Konfiguration")

    model = st.sidebar.text_input("Model (HF repo oder Pfad)", "Qwen/Qwen2.5-0.5B-Instruct")
    model_type = st.sidebar.selectbox("Model Type", ["hf", "vllm"], index=0)
    task = st.sidebar.text_input("Task", "gsm8k")
    limit = st.sidebar.text_input("Limit (z.B. 20 oder 0.1)", "20")
    num_shots = st.sidebar.number_input("num_shots", min_value=0, max_value=50, value=0, step=1)
    batch_size = st.sidebar.number_input("batch_size", min_value=1, max_value=256, value=1, step=1)
    

    # Put each run in a task-named folder so the user doesn't care about task-000- files
    run_name = st.sidebar.text_input("Run Name (Ordnername)", f"{task}-{int(time.time())}")
    output_dir = st.sidebar.text_input("output_dir", str(default_out_base / run_name))
    cache_dir = st.sidebar.text_input("cache_dir", str(default_cache))

    save_raw = st.sidebar.checkbox("save_raw_requests", value=True)
    extra_args = st.sidebar.text_input("Extra args (optional)", "")

    colA, colB = st.columns([1, 2])
    with colA:
        run_clicked = st.button("▶ Run OLMES", type="primary")

    if run_clicked:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cfg = RunConfig(
            model=model,
            model_type=model_type,
            task=task,
            limit=limit,
            num_shots=int(num_shots),
            output_dir=output_dir,
            cache_dir=cache_dir,
            batch_size=int(batch_size),
            extra_args=extra_args,
            save_raw_requests=save_raw,
        )

        st.info("Eval läuft im Container/Runner (nicht im Browser). Logs unten:")
        rc = run_olmes(cfg)
        if rc != 0:
            st.error(f"OLMES beendet mit Return Code {rc}")
        else:
            st.success("Fertig! Ergebnisse werden geladen…")
            st.session_state["open_dir"] = output_dir
            st.rerun()

    # Show results if a run dir is set
    out_dir = Path(st.session_state.get("open_dir", output_dir))

else:
    st.sidebar.subheader("Ergebnisse öffnen")
    out_base = st.sidebar.text_input("Suche in Basisordner", str(default_out_base))
    candidates = find_run_dirs(Path(out_base))
    selected = st.sidebar.selectbox(
        "Gefundene Runs (metrics.json)",
        options=[str(p) for p in candidates] if candidates else [],
    )
    out_dir = Path(selected) if selected else Path(".")

# ---------- Results view ----------
st.divider()
st.subheader("Ergebnisse")

if out_dir.exists():
    metrics, task_df = load_summary(out_dir)
    if not metrics:
        st.warning(f"Kein metrics.json gefunden in: {out_dir}")
        st.stop()

    # Top summary
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Output Dir", str(out_dir))
    with c2:
        model_cfg = metrics.get("model_config", {})
        st.metric("Model", model_cfg.get("model", ""))
    with c3:
        st.metric("Tasks", str(len(metrics.get("tasks", []) or [])))

    if task_df is not None and not task_df.empty:
        st.write("### Task-Metriken")
        st.dataframe(task_df, use_container_width=True)

    # Predictions table
    st.write("### Predictions (instanz-level)")
    preds = load_predictions(out_dir)
    if preds.empty:
        st.info("Keine *predictions.jsonl gefunden.")
        st.stop()

    rec_map = load_recorded_inputs(out_dir)

    # Filters
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        only_errors = st.checkbox("nur Fehler (exact_match=0)", value=True)
    with f2:
        only_maxed = st.checkbox("nur max_tokens_reached", value=False)
    with f3:
        search = st.text_input("Suche (gold/model_answer enthält…)", "")

    view = preds.copy()
    if only_errors:
        view = view[view["exact_match"] == 0.0]
    if only_maxed:
        view = view[view["max_tokens_reached"] == True]
    if search.strip():
        s = search.strip().lower()
        view = view[
            view["gold"].astype(str).str.lower().str.contains(s, na=False)
            | view["model_answer"].astype(str).str.lower().str.contains(s, na=False)
        ]

    st.dataframe(
        view[["task", "doc_id", "native_id", "gold", "model_answer", "exact_match", "num_tokens", "max_tokens_reached"]],
        use_container_width=True,
        height=420,
    )

    st.write("### Detailansicht")
    pick = st.number_input("doc_id auswählen", min_value=0, max_value=int(preds["doc_id"].max()), value=0, step=1)
    # If multiple tasks exist, pick first match
    row = preds[preds["doc_id"] == pick].head(1)
    if row.empty:
        st.info("doc_id nicht gefunden.")
        st.stop()

    r = row.iloc[0].to_dict()
    st.write(f"**Task:** {r['task']}  |  **doc_id:** {r['doc_id']}  |  **gold:** `{r['gold']}`  |  **model_answer:** `{r['model_answer']}`")

    rec = rec_map.get((r["task"], int(r["doc_id"])))
    if rec:
        doc = rec.get("doc", {})
        st.write("**Question (recorded):**")
        st.code(doc.get("question", ""), language="text")
        st.write("**Prompt (query):**")
        st.code(doc.get("query", ""), language="text")
    else:
        st.info("Kein recorded-input für dieses Beispiel vorhanden (bei dir waren nur die ersten N aufgezeichnet).")

    st.write("**Model output (raw):**")
    st.code(r.get("continuation_raw") or "", language="text")

else:
    st.warning("Bitte einen gültigen Output-Ordner auswählen/erzeugen.")
