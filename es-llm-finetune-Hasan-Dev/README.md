# es-llm-finetune (Docker + OLMES)

Dieses Repository stellt eine reproduzierbare Umgebung bereit, um Sprachmodelle (LLMs) **zu evaluieren** – aktuell über das **OLMES**-CLI-Tool (AllenAI).
Der Fokus liegt auf einem einfachen **Baseline-Workflow**: Modell laden → Tasks laufen lassen → Ergebnisse als JSON/Logs speichern (ohne ES/Backprop).

> Hinweis: Das Docker-Image ist standardmäßig auf **CPU** ausgelegt. GPU-Support kann ergänzt werden (siehe Abschnitt „GPU“).

---

## Features

- ✅ Reproduzierbare Evaluation in Docker (Python 3.11 slim)
- ✅ Installiert Torch (CPU), Transformers, Datasets, Accelerate
- ✅ OLMES-CLI verfügbar (`olmes --help`, `olmes --list-tasks`, ...)
- ✅ Output- und Cache-Verzeichnisse mountbar, damit Ergebnisse auf dem Host landen
- ✅ Geeignet für Baseline-Experimente (0-shot/ few-shot, limit, seed, batch)

---

## Voraussetzungen

### Lokal
- Docker (Engine) installiert
- Optional: `buildx` (empfohlen), da der Legacy-Builder deprecated ist

### Optional (für GPU)
- NVIDIA GPU + Treiber
- NVIDIA Container Toolkit (`nvidia-docker2`)

---

## Quickstart (CPU)

### 1) Image bauen
```bash
docker build -t es-llm:cpu-olmes .

```

### 2) Smoke-Test: Imports
```bash
docker run --rm -it es-llm:cpu-olmes \
  python -c "import torch, transformers, datasets; print(torch.__version__)"
```

### 3) OLMES prüfen
```bash
docker run --rm -it es-llm:cpu-olmes olmes --help
docker run --rm -it es-llm:cpu-olmes olmes --list-tasks
docker run --rm -it es-llm:cpu-olmes olmes --list-models
```

### 4) Typische Nutzung: Evaluation mit OLMES

OLMES führt definierte Tasks/Benchmarks gegen ein Modell aus und schreibt strukturierte Ergebnisse in ein Output-Verzeichnis.

### Empfehlung: Output + Cache auf Host mounten

Damit Ergebnisse erhalten bleiben, mounte ein Projektverzeichnis in den Container:
```bash
docker run --rm -it \
  -v "$PWD:/work" \
  es-llm:cpu-olmes \
  olmes \
    --model <MODEL> \
    --model-type hf \
    --task <TASKNAME> \
    --num-shots 0 \
    --limit 100 \
    --output-dir /work/out \
    --cached-output-dir /work/cache
```
--num-shots 0: Zero-shot Baseline

--limit 100: schneller Testlauf (statt kompletter Benchmark)

--cached-output-dir: spart Zeit bei Wiederholungen (Caching)

--output-dir: hier landen Metriken/Predictions/Logs

Wichtig: <MODEL> kann z. B. ein HF-Repo (Qwen/Qwen2.5-0.5B-Instruct) oder ein lokaler Pfad sein.

### Beispiele
# A) Sehr kleiner Debug-Run (schnell, zum Checken)
--inspect weglassen für qwen eval
```bash
docker run --rm -it -v "$PWD:/work" es-llm:cpu-olmes \
  olmes \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --model-type hf \
    --task gsm8k \
    --num-shots 0 \
    --limit 20 \
    --output-dir /work/out/debug \
    --cached-output-dir /work/cache \
    --inspect   
```

# B) Reproduzierbarer Run (Seed + Revision)
```bash
docker run --rm -it -v "$PWD:/work" es-llm:cpu-olmes \
  olmes \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --model-type hf \
    --revision main \
    --task gsm8k \
    --num-shots 0 \
    --random-subsample-seed 123 \
    --limit 500 \
    --batch-size 4 \
    --output-dir /work/out/gsm8k_0shot \
    --cached-output-dir /work/cache
```

# C) Modell-Args (z. B. trust_remote_code / device_map)
```bash
docker run --rm -it -v "$PWD:/work" es-llm:cpu-olmes \
  olmes \
    --model <MODEL> \
    --model-type hf \
    --task <TASKNAME> \
    --model-args "{'trust_remote_code': True, 'device_map': 'auto'}" \
    --output-dir /work/out \
    --cached-output-dir /work/cache
```

### Output-Struktur (typisch)

Je nach Task/Config erzeugt OLMES u. a.:

summary/Metriken (Accuracy, etc.)

samples/Predictions (inkl. raw_output, gold, pred, reward)

ggf. zusätzliche Logs oder Request-Dumps (--save-raw-requests)

Empfohlen:

out/ versionieren nicht, lieber als Artefakt speichern

cache/ ebenfalls nicht versionieren (in .gitignore)

### Reproduzierbarkeit / „Saubere“ Baselines

Für eine gute Baseline (ohne ES/Backprop) empfehle ich:

--revision (fixe Modellrevision statt „latest“)

--random-subsample-seed bei --limit (damit Subsampling identisch bleibt)

--num-shots explizit setzen (0, 1, 4, …)

Konfig und Kommandozeile im Output mitschreiben (z. B. in out/<run>/command.txt)

### Troubleshooting

# 1) Docker: „legacy builder is deprecated“
Das ist eine Warnung. Optional kannst du BuildKit aktivieren:
```bash
docker buildx build -t es-llm:cpu-olmes .
```

# 2) Build-Fehler: gcc: No such file or directory
Einige Python-Pakete bauen native Extensions (z. B. pytrec_eval). Lösung: Build-Tools installieren.
Falls das bei dir wieder auftaucht, ergänze im Dockerfile:
```bash
RUN apt-get update && apt-get install -y --no-install-recommends \
    git gcc g++ build-essential \
 && rm -rf /var/lib/apt/lists/*
```

# 3) Downloads/Cache werden riesig

Setze Cache-Ordner bewusst (Host mounten!) und nutze:

--cached-output-dir /work/cache

optional HF_HOME=/work/.hf via -e HF_HOME=/work/.hf

Beispiel:
```bash
docker run --rm -it \
  -e HF_HOME=/work/.hf \
  -v "$PWD:/work" \
  es-llm:cpu-olmes \
  olmes ...
```

# OLMES UI – Interaktive Evaluierung von Sprachmodellen

Eine **Streamlit-basierte Web-UI** zur Evaluierung von Sprachmodellen mit dem OLMES-Benchmark-Tool von AllenAI.

## 🚀 Quick Start

### Mit Docker (empfohlen)

```bash
docker run --rm -it \
  -p 8501:8501 \
  -v "$PWD:/work" \
  es-llm:cpu-olmes \
  bash -lc "pip install -q streamlit pandas && \
            streamlit run /work/ui/olmes_ui.py \
            --server.address 0.0.0.0 --server.port 8501"
```

Dann öffne: **http://localhost:8501**

### Lokal (ohne Docker)

```bash
pip install streamlit pandas
streamlit run ui/olmes_ui.py
```

---

## 📖 Benutzeranleitung

### Modus 1: Run Evaluation – Neue Evaluierung starten

#### Schritt 1: Konfiguration (Sidebar)

| Parameter | Beschreibung | Beispiel |
|-----------|-------------|---------|
| **Model** | HuggingFace Repo oder lokaler Pfad | `Qwen/Qwen2.5-0.5B-Instruct` |
| **Model Type** | `hf` (HuggingFace) oder `vllm` (schneller) | `hf` |
| **Task** | Benchmark-Task | `gsm8k`, `arc`, `hellaswag` |
| **Limit** | Anzahl Instanzen oder Anteil (z.B. 20 oder 0.1) | `20` |
| **num_shots** | Few-shot Prompting (0 = Zero-shot) | `0` oder `3` |
| **batch_size** | Batch-Größe | `1` (CPU-freundlich) |
| **Run Name** | Ordnername für Ergebnisse | `qwen-gsm8k-test` |
| **output_dir** | Wo Ergebnisse gespeichert werden | `/work/out/qwen-gsm8k-test` |
| **cache_dir** | Cache für HF-Modelle | `/work/cache` |
| **save_raw_requests** | Speichere vollständige Prompts | ✓ (an) |
| **Extra args** | Zusätzliche OLMES-Argumente | `--seed 42` |

#### Schritt 2: Starten

Klick auf **▶ Run OLMES** → Die Evaluierung startet im Hintergrund

**Live-Logs:** Du siehst die OLMES-Ausgabe in Echtzeit

#### Schritt 3: Warten

Abhängig von:
- Modellgröße
- Anzahl Instanzen
- Hardware

→ Kann von Minuten bis Stunden dauern

---

### Modus 2: Open Existing Results – Frühere Ergebnisse analysieren

#### Schritt 1: Run auswählen

1. Gib Basis-Ordner an (z.B. `/work/out`)
2. Wähle einen Run aus der Dropdown-Liste
   - Neueste Runs oben
   - Erkennt automatisch `metrics.json`

#### Schritt 2: Ergebnisse untersuchen

---

## 📊 Ergebnisse-Ansicht

### 1. Task-Metriken (Zusammenfassung)

```
┌─────────────────────────────────────┐
│ Task      │ Accuracy │ Time (sec)  │
├─────────────────────────────────────┤
│ gsm8k     │ 0.65     │ 234.5       │
└─────────────────────────────────────┘
```

- **primary_score**: Hauptmetrik (z.B. Accuracy)
- **num_instances**: Anzahl getesteter Beispiele
- **processing_time_sec**: Gesamtdauer

### 2. Predictions Table (Instanz-Ebene)

Tabelle mit allen Vorhersagen:

```
Task   doc_id  Gold      Model Answer    Match  Tokens  MaxReached
───────────────────────────────────────────────────────────────────
gsm8k  0       42        42              ✓      15      ✗
gsm8k  1       100       99              ✗      22      ✗
gsm8k  2       7         [ERROR]         ✗      2048    ✓
```

#### Filter

- **nur Fehler**: Zeige nur falsch beantwortete Fragen
- **nur max_tokens_reached**: Zeige nur Fälle, wo das Modell abgebrochen hat
- **Suche**: Text-Filter (gold oder model_answer enthält…)

### 3. Detail-Ansicht

Wähle eine `doc_id` aus → Sieh Frage, Prompt und vollständige Antwort:

```
Task: gsm8k | doc_id: 5 | gold: 28 | model_answer: 28

Question (recorded):
  "Sarah hat 12 Äpfel. Sie kauft 16 weitere. Wie viele hat sie jetzt?"

Prompt (query):
  "Question: Sarah hat 12 Äpfel. Sie kauft 16 weitere. Wie viele hat sie jetzt?
   Answer:"

Model output (raw):
  "Sarah hat zunächst 12 Äpfel.
   Sie kauft 16 weitere Äpfel.
   12 + 16 = 28
   Sarah hat 28 Äpfel."
```

---

## 📁 Ausgabe-Struktur

Nach einer Evaluierung:

```
out/
├── qwen-gsm8k-test/
│   ├── metrics.json                    # Zusammenfassung
│   ├── task-000-gsm8k-predictions.jsonl
│   ├── task-000-gsm8k-recorded-inputs.jsonl
│   └── task-000-gsm8k-raw-requests.jsonl (optional)
│
└── baseline-model-test/
    ├── metrics.json
    └── task-000-gsm8k-predictions.jsonl
```

### Datei-Formate

#### `metrics.json`
```json
{
  "model_config": {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "model_type": "hf"
  },
  "tasks": [
    {
      "alias": "gsm8k",
      "num_instances": 20,
      "processing_time": 234.5,
      "metrics": {
        "primary_score": 0.65,
        "accuracy": 0.65,
        "extra_metrics": {...}
      }
    }
  ]
}
```

#### `task-000-gsm8k-predictions.jsonl`
```json
{"doc_id": 0, "native_id": "...", "label": "42", "model_output": [{"model_answer": "42", "continuation_raw": "..."}], "metrics": {"exact_match": 1.0, "num_tokens": 15}}
```

#### `task-000-gsm8k-recorded-inputs.jsonl`
```json
{"doc_id": 0, "doc": {"question": "...", "query": "..."}}
```

---

## 🔧 Tipps & Tricks

### Baseline vergleichen

1. Evaluiere **Original-Modell** → speichere Ergebnisse
2. Evaluiere **Fine-tuned Modell** → speichere in separatem Ordner
3. Öffne beide Runs nacheinander → vergleiche Metriken manuell

**Beispiel:**
```bash
# Run 1: Original
olmes --model Qwen/Qwen2.5-0.5B-Instruct --output-dir /work/out/baseline-run

# Run 2: Fine-tuned
olmes --model /work/checkpoints/finetuned-model --output-dir /work/out/finetuned-run
```

### Few-shot Prompting testen

Ändere `num_shots` → Vergleiche Ergebnisse:
- `num_shots: 0` (Zero-shot)
- `num_shots: 3` (3-shot)
- `num_shots: 5` (5-shot)

### Performance-Optimierung

- **Kleine Limits zum Testen:** `--limit 20` statt `--limit 1000`
- **Batch-size erhöhen:** `--batch-size 8` oder `16` (wenn VRAM reicht)
- **vllm statt hf:** `--model-type vllm` für schnellere Inferenz

### Errors debuggen

Filtere auf **"nur Fehler"** → Sieh die problematischen Beispiele:
- Ist das Modell zu klein für die Task?
- Sind Tokens zu früh aufgebraucht?
- Ist der Prompt unklar formuliert?

---

## ⚙️ Umgebungsvariablen

In `Dockerfile` oder `.env.sh` setzen:

```bash
export WORKDIR=/work              # Standard-Basis-Ordner
export HF_HOME=/work/cache        # HF-Modell-Cache
export TRANSFORMERS_CACHE=/work/cache
```

---

## 🐛 Häufige Fehler

### Problem: "Modell nicht gefunden"
```
Error: Model 'xyz' not found on HuggingFace
```
**Lösung:** Modell-Namen korrekt schreiben, z.B. `Qwen/Qwen2.5-0.5B-Instruct`

### Problem: "CUDA out of memory"
```
RuntimeError: CUDA out of memory
```
**Lösung:**
- `batch_size` reduzieren
- `--limit` reduzieren
- Auf CPU evaluieren (`model_type: hf`)

### Problem: "metrics.json nicht gefunden"
```
Kein metrics.json gefunden in: /work/out/...
```
**Lösung:** Die Evaluierung ist noch nicht fertig oder nicht erfolgreich abgelaufen. Sieh die Logs.

---

## 📚 Weitere Ressourcen

- [OLMES GitHub](https://github.com/allenai/olmes)
- [Streamlit Doku](https://docs.streamlit.io/)
- [Projekt-README](../README.md)

---

**Kontakt:** Bei Fragen zu diesem Projekt → Sieh [../README.md](../README.md)