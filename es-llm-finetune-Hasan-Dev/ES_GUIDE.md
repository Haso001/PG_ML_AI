# ES-LLM: Layer-wise Fine-Tuning mit Evolutionary Strategies

> **Finetuning einzelner Decoder-Layer** im **Qwen2.5-0.5B** Modell  
> mittels Evolutionary Strategies (ES) – **ohne Backpropagation**.

---

## Projektstruktur

```
es-llm-finetune-Hasan-Dev/
│
├── configs/                          # ── Experiment-Konfigurationen ──
│   ├── default.yaml                  #   Defaults (wird immer gemerged)
│   ├── gsm8k_last_layer.yaml         #   Experiment: letzter Layer
│   └── gsm8k_multilayer.yaml         #   Experiment: mehrere Layer + Low-Rank
│
├── src/es_llm/                       # ── Kernbibliothek ──
│   ├── __init__.py
│   │
│   ├── model/                        # Modell-Handling
│   │   ├── loader.py                 #   HuggingFace-Modell laden
│   │   └── layer_selector.py         #   Layer auswählen, einfrieren, Params extrahieren
│   │
│   ├── es/                           # Evolutionary Strategies
│   │   ├── base.py                   #   Abstrakte Basisklasse (ask/tell Interface)
│   │   ├── openai_es.py              #   OpenAI-ES (Salimans et al., 2017)
│   │   ├── cma_es.py                 #   CMA-ES Wrapper
│   │   └── noise.py                  #   Noise-Generierung (Gauss, Antithetic, Low-Rank)
│   │
│   ├── fitness/                      # Fitness-Evaluierung
│   │   ├── base.py                   #   Abstrakte Fitness-Schnittstelle
│   │   └── gsm8k.py                  #   GSM8K Accuracy als Fitness
│   │
│   ├── training/                     # Training Loop
│   │   └── es_trainer.py             #   Haupt-Trainingsschleife (orchestriert alles)
│   │
│   └── utils/                        # Hilfsfunktionen
│       ├── config.py                 #   YAML-Config laden & mergen
│       └── logging.py                #   Logging & Run-History
│
├── scripts/                          # ── Einstiegspunkte ──
│   ├── train_es.py                   #   ES-Training starten
│   ├── inspect_layers.py             #   Modell-Architektur inspizieren
│   ├── eval_es_model.py              #   Fine-tuned Modell evaluieren
│   ├── eval_gsm8k.py                 #   Standalone GSM8K Eval (bestehend)
│   └── olmes_report.py               #   OLMES Report (bestehend)
│
├── experiments/runs/                  # ── Outputs (gitignored) ──
│   └── gsm8k_last_layer_es/
│       ├── config.json               #   Gespeicherte Konfiguration
│       ├── layer_selection.txt        #   Welche Layer selektiert wurden
│       ├── training_log.jsonl         #   Generationen-Log
│       ├── checkpoint_gen0050.pt      #   Zwischenstände
│       ├── best_layer_weights.pt      #   Beste Layer-Gewichte
│       └── model/                     #   Komplettes HF-Modell
│
├── ui/olmes_ui.py                    # Streamlit UI (bestehend)
├── notebooks/
│   └── es_training_colab.ipynb       # Colab-Notebook (Haupteinstieg)
├── Dockerfile
├── requirements.txt
└── ES_GUIDE.md                       # ← Diese Datei
```

---

## Quickstart: Google Colab (empfohlen)

### 1. Notebook öffnen

Öffne `notebooks/es_training_colab.ipynb` in [Google Colab](https://colab.research.google.com):

- **Upload:** Datei → Notebook hochladen → `es_training_colab.ipynb` wählen
- **Oder aus GitHub:** Datei → Notebook öffnen → GitHub-Tab → Repo-URL einfügen

### 2. GPU aktivieren

> **Runtime → Change runtime type → GPU (T4)**

| GPU | VRAM | Qwen 0.5B (fp16) | Empfehlung |
|-----|------|-------------------|------------|
| T4 (free) | 16 GB | ~1 GB | Reicht gut |
| A100 (Pro) | 40 GB | ~1 GB | Überdimensioniert, aber schneller |

### 3. Notebook durchlaufen

Das Notebook führt Schritt für Schritt durch:
1. GPU prüfen & Dependencies installieren
2. Modell-Architektur inspizieren
3. Layer-Selektion testen
4. Baseline messen
5. ES-Training starten
6. Ergebnisse plotten
7. Nach Google Drive speichern

### Wichtige Colab-Hinweise

- **Ergebnisse IMMER nach Google Drive speichern** – Colab-VMs sind flüchtig!
- **dtype `float16`** verwenden – halber Speicher, schnellere Inferenz
- Colab trennt nach ~90 Min. Inaktivität → Browser-Tab offen halten
- Bei langen Läufen: Checkpoints alle 10 Generationen (schon im Default)

---

## Qwen2.5-0.5B Architektur-Überblick

```
model.embed_tokens                     [151936 × 896]    ~136M params
model.layers.{0..23}                   24 Decoder-Blöcke
  ├── input_layernorm                  [896]              RMSNorm
  ├── self_attn
  │     ├── q_proj                     [896 × 896]        Query
  │     ├── k_proj                     [128 × 896]        Key   (GQA: 2 KV-heads)
  │     ├── v_proj                     [128 × 896]        Value (GQA: 2 KV-heads)
  │     └── o_proj                     [896 × 896]        Output
  ├── post_attention_layernorm         [896]              RMSNorm
  └── mlp
        ├── gate_proj                  [4864 × 896]       SwiGLU Gate
        ├── up_proj                    [4864 × 896]       Up-Projektion
        └── down_proj                  [896 × 4864]       Down-Projektion
model.norm                             [896]              Final RMSNorm
lm_head                               [151936 × 896]    ~136M params
```

**Pro Decoder-Layer: ~14.7M Parameter**  
**Gesamt: ~494M Parameter**

---

## Step-by-Step Guide

### Schritt 1: Layer inspizieren

Bevor du mit ES loslegst, verstehe welche Parameter du ändern willst:

```bash
# Alle Parameter auflisten
python scripts/inspect_layers.py --model Qwen/Qwen2.5-0.5B-Instruct

# Nur Layer 23 (letzte Decoder-Schicht)
python scripts/inspect_layers.py --model Qwen/Qwen2.5-0.5B-Instruct --filter "layers\.23"

# Zusammenfassung pro Layer
python scripts/inspect_layers.py --model Qwen/Qwen2.5-0.5B-Instruct --summary

# Top-20 größte Parameter
python scripts/inspect_layers.py --model Qwen/Qwen2.5-0.5B-Instruct --top 20
```

### Schritt 2: Experiment konfigurieren

Erstelle/bearbeite eine YAML-Datei unter `configs/`:

```yaml
# configs/mein_experiment.yaml
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"
  dtype: "float16"                    # float16 auf GPU!
  device: "cuda"                      # Colab GPU

layer_selection:
  strategy: "by_index"
  layer_indices: [23]                 # Nur der letzte Layer
  components: "mlp"                   # Nur MLP-Gewichte (gate, up, down)

es:
  algorithm: "openai_es"
  population_size: 20
  sigma: 0.01                         # Rauschstärke
  learning_rate: 0.001
  num_generations: 100
  antithetic: true                    # Mirrored Sampling → weniger Varianz
  fitness_shaping: "centered_rank"    # Robuster als raw fitness

fitness:
  task: "gsm8k"
  split: "train"
  num_samples: 30                     # Samples pro Fitness-Evaluation
```

### Schritt 3: Baseline messen

```bash
# Evaluiere das unveränderte Modell auf dem Test-Split
python scripts/eval_gsm8k.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --split test \
    --n 200 \
    --out experiments/runs/baseline_test200.json
```

### Schritt 4: ES-Training starten

```bash
# Mit Konfigurationsdatei
python scripts/train_es.py --config configs/gsm8k_last_layer.yaml

# Mit CLI-Overrides
python scripts/train_es.py \
    --config configs/gsm8k_last_layer.yaml \
    --set es.sigma=0.005 \
    --set es.num_generations=200 \
    --set fitness.num_samples=50
```

### Schritt 5: Fine-tuned Modell evaluieren

```bash
python scripts/eval_es_model.py \
    --run experiments/runs/gsm8k_last_layer_es \
    --split test \
    --num_samples 200
```

---

## Wie Evolutionary Strategies funktionieren (Layer-Level)

### Kernidee

Anstatt Gradienten via Backprop zu berechnen, schätzt ES den Gradienten durch **randomisierte finite Differenzen**:

```
Für Parameter θ der selektierten Layer:

1. Generiere N Rausch-Vektoren:   ε_i ~ N(0, σ²I)
2. Evaluiere Fitness:              F_i = F(θ + ε_i)     ← z.B. GSM8K Accuracy
3. Update:                         θ ← θ + α/(Nσ) · Σ F_i · ε_i
```

### Warum layer-weise?

| Ansatz | Parameter | Machbarkeit mit ES |
|--------|-----------|-------------------|
| Ganzes Modell (~494M) | ❌ Zu viele | Nicht sinnvoll |
| Letzter Layer (~14.7M) | ⚠️ Grenzwertig | Möglich mit Low-Rank |
| Nur LayerNorm (~1.8K/Layer) | ✅ Sehr wenige | Ideal für ES |
| Nur MLP eines Layers (~13M) | ⚠️ Machbar | Mit Low-Rank Noise |
| Nur attention q/k/v (~1M) | ✅ Handlich | Gut geeignet |

### Empfohlene Strategie

1. **Start klein**: Beginne mit LayerNorm-Parametern des letzten Layers (~1.8K Params)
2. **Skaliere hoch**: Dann attention-Gewichte oder MLP eines Layers
3. **Nutze Low-Rank**: Für große Gewichtsmatrizen (> 100K Params) aktiviere `low_rank.enabled: true`
4. **Antithetic Sampling**: Immer aktiviert lassen (halbiert Varianz)
5. **Centered Rank**: Immer `fitness_shaping: centered_rank` verwenden

---

## Layer-Auswahl-Strategien

### A) Nach Index (empfohlen zum Start)

```yaml
layer_selection:
  strategy: "by_index"
  layer_indices: [23]          # Nur letzter Layer
  components: "layernorm"      # Nur die LayerNorm Parameter
```

### B) Nach Regex (flexibel)

```yaml
layer_selection:
  strategy: "by_regex"
  layer_regex: "model\\.layers\\.(2[0-3])\\.mlp\\."    # MLP der Layer 20-23
```

### C) Nach exaktem Namen

```yaml
layer_selection:
  strategy: "by_name"
  layer_names:
    - "model.layers.23.self_attn.q_proj.weight"
    - "model.layers.23.self_attn.v_proj.weight"
```

### Components-Filter

| Wert | Beschreibung | Params/Layer |
|------|-------------|-------------|
| `"all"` | Alle Parameter im Layer | ~14.7M |
| `"attention"` | q/k/v/o-Projektionen | ~1.8M |
| `"attention_qkv"` | Nur q/k/v | ~1.1M |
| `"mlp"` | gate/up/down-Projektionen | ~13M |
| `"layernorm"` | input_layernorm + post_attention_layernorm | ~1.8K |

---

## Hyperparameter-Guide

| Parameter | Empfohlener Start | Effekt |
|-----------|------------------|--------|
| `population_size` | 20–50 | Mehr → bessere Gradientenschätzung, aber langsamer |
| `sigma` | 0.001–0.01 | Noise-Stärke. Zu hoch → zerstört Modell. Zu niedrig → kein Signal |
| `learning_rate` | 0.0001–0.01 | Schrittweite. σ und lr zusammen tunen |
| `num_generations` | 50–500 | Je nach Fitness-Konvergenz |
| `fitness.num_samples` | 20–100 | Mehr → stabilere Fitness, aber langsamer |
| `antithetic` | `true` | Immer an. Halbiert Varianz bei gleichem Compute |
| `low_rank.rank` | 4–32 | Für große Matrizen. Niedriger → weniger expressiv, schneller |

### σ / lr Faustregel

```
σ × lr ≈ 1e-5 ... 1e-4   (als Produkt)

Beispiele:
  σ=0.01,  lr=0.001   → Produkt = 1e-5  ✓
  σ=0.001, lr=0.01    → Produkt = 1e-5  ✓
  σ=0.1,   lr=0.01    → Produkt = 1e-3  ⚠️ zu aggressiv
```

---

## Eigene Fitness-Funktionen hinzufügen

Erweitere das System für andere Tasks:

```python
# src/es_llm/fitness/my_task.py
from .base import BaseFitness

class MyTaskFitness(BaseFitness):
    def name(self) -> str:
        return "my_task"

    def evaluate(self, model, tokenizer) -> float:
        # Deine Evaluierungslogik hier
        # Return: Skalar, höher = besser
        correct = 0
        total = 100
        for sample in my_dataset:
            pred = generate(model, tokenizer, sample)
            if pred == sample.gold:
                correct += 1
        return correct / total
```

Dann in `es_trainer.py` → `_build_fitness()` registrieren.

---

## Tipps zur Performance

### Google Colab GPU (T4 / A100)

- **Immer `dtype: float16`** – Qwen 0.5B belegt dann ~1 GB statt ~2 GB VRAM
- `population_size: 20–50` auf T4, `50–100` auf A100
- `num_samples: 50–100` für stabile Fitness (GPU schafft das schnell)
- `max_new_tokens: 256` ist Standard, auf `128` reduzieren wenn Speed wichtiger
- Rechne mit **~10–60 Sek pro Generation** (je nach `num_samples` und `population_size`)
- **VRAM-Verbrauch** wird automatisch zwischen Candidates bereinigt (`torch.cuda.empty_cache()`)

### Speicher-Budget (T4, 16 GB VRAM)

| Was | VRAM |
|-----|------|
| Modell (fp16) | ~1.0 GB |
| Noise-Vektoren (pop=50, 14.7M params fp32) | ~2.9 GB |
| Generation-Buffer | ~0.5 GB |
| **Gesamt** | **~4.5 GB (von 16 GB)** |

→ Viel Headroom. Selbst den ganzen letzten Layer (`components: all`) auf einem T4 zu perturbieren ist kein Problem.

### CPU (Fallback)

- Nutze `num_samples: 20` für schnelle Iterationen
- `population_size: 10–20` ist ein guter Kompromiss
- `max_new_tokens: 128` reduzieren wenn möglich
- Rechne mit ~1–5 Minuten pro Generation

### Parallelisierung (fortgeschritten)

ES ist trivial parallelisierbar — jede Candidate-Evaluation ist unabhängig:
- Multi-GPU: Verschiedene Candidates auf verschiedene GPUs
- Multi-Process: `torch.multiprocessing` für CPU-Parallelisierung
- Cluster: Candidates über mehrere Nodes verteilen

---

## Häufige Probleme

| Problem | Lösung |
|---------|--------|
| Fitness ändert sich nicht | σ erhöhen, oder anderes Component wählen |
| Fitness wird schlechter | σ senken, lr senken |
| Out of Memory | `num_samples` oder `population_size` senken |
| NaN in Parametern | σ war zu hoch → Modell destabilisiert |
| Zu langsam | `num_samples` senken, `max_new_tokens` kürzen |
| Colab Session getrennt | Ergebnisse nach Drive speichern, Checkpoint laden |
| "CUDA out of memory" | `population_size` oder `num_samples` senken, `dtype: float16` prüfen |
