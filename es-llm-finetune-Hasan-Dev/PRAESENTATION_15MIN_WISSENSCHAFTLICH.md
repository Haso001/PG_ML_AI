# ES-LLM auf GSM8K: Wissenschaftliche 15-Minuten-Praesentation

Ziel dieser Vorlage:
- wissenschaftlich saubere Storyline
- 15 Minuten genau taktiert
- direkt auf deinem Projektstand aufbauend

## Slide 1 (0:00-1:00) - Titel und Problem
Titel:
- Layer-wise Evolutionary Fine-Tuning fuer Mathematical Reasoning in Small LLMs

Untertitel:
- ES-basierte Anpassung einzelner Qwen2.5-0.5B Layer ohne Backpropagation

Sprechtext:
- Dieses Projekt untersucht, ob wir mathematische Reasoning-Leistung auf GSM8K verbessern koennen, indem wir nur wenige Layer eines 0.5B-Modells mit Evolutionary Strategies optimieren.
- Fokus ist nicht Full Finetuning, sondern ein parameter-effizienter, gradient-freier Ansatz.

Beleg im Code:
- `src/es_llm/training/es_trainer.py`
- `src/es_llm/model/layer_selector.py`
- `src/es_llm/es/openai_es.py`

## Slide 2 (1:00-2:10) - Forschungsfrage und Hypothesen
Forschungsfrage:
- Kann ES auf selektierten Layern die GSM8K-Performance von Qwen2.5-0.5B messbar verbessern, bei begrenztem Compute-Budget?

Hypothesen:
- H1: Attention-nahe Layer (spate Decoder-Layer) tragen ueberproportional zur kurzfristigen GSM8K-Verbesserung bei.
- H2: Log-Likelihood-Fitness liefert schnelleres und stabileres Optimierungssignal als diskrete 0/1-Accuracy waehrend des Trainings.
- H3: Antithetic Sampling plus Fitness Shaping reduziert die Varianz des ES-Updates.

Beleg im Code:
- Fitness-Varianten: `src/es_llm/fitness/gsm8k.py`, `src/es_llm/fitness/gsm8k_loglikelihood.py`
- Varianzreduktion: `src/es_llm/es/openai_es.py`

## Slide 3 (2:10-3:30) - Methodischer Rahmen
Setup:
- Basismodell: `Qwen/Qwen2.5-0.5B-Instruct`
- Task: GSM8K
- Layer-selektive Perturbation statt Full-Model-Update

Warum ES statt Backprop:
- Keine Gradientenberechnung notwendig
- Direkte Black-box-Optimierung auf Task-Fitness
- Flexibel fuer nicht-differenzierbare Ziele

Trade-off:
- Viele Fitness-Evaluationen pro Generation
- Daher ist Fitness-Design der zentrale Laufzeit- und Stabilitaetsfaktor

Beleg im Code:
- Trainer-Orchestrierung: `src/es_llm/training/es_trainer.py`

## Slide 4 (3:30-5:00) - Architektur und Pipeline
Ablauf pro Run:
1. Config laden und mergen (`default.yaml` + experiment-spezifisch)
2. Modell + Tokenizer laden
3. Ziel-Parameter durch LayerSelector bestimmen
4. Baseline-Fitness messen
5. ES-Loop: ask -> evaluate (Population) -> tell -> checkpoint
6. Bestes Parametervector anwenden und Modell speichern

Artefakte fuer Reproduzierbarkeit:
- `config.json`
- `training_log.jsonl`
- `checkpoint_genXXXX.pt`
- `best_layer_weights.pt`
- gespeichertes HF-Modell unter `model/`

Beleg im Code:
- `src/es_llm/utils/config.py`
- `src/es_llm/training/es_trainer.py`
- `src/es_llm/utils/logging.py`

## Slide 5 (5:00-6:20) - Mathematischer Kern (OpenAI-ES)
Kernidee:
- Sample Perturbationen: epsilon_i ~ N(0, sigma^2 I)
- Evaluiere Kandidaten: F(theta + epsilon_i)
- Update mit geshapeten Fitnesswerten

Formel:
```text
grad_est = (1 / (N * sigma)) * Sum_i [u_i * epsilon_i]
theta_{t+1} = theta_t + alpha * grad_est
```

Wichtige Stabilitaetskomponenten im Projekt:
- antithetic sampling (epsilon, -epsilon)
- centered-rank fitness shaping
- optionale Adam-Update-Variante
- sigma decay (constant, cosine, linear, adaptive)

Beleg im Code:
- `src/es_llm/es/openai_es.py`

## Slide 6 (6:20-7:40) - Fitness-Design: Binary vs Log-Likelihood
Binary-Fitness (`gsm8k.py`):
- erzeugt komplette Antwort via `model.generate(...)`
- extrahiert `#### <answer>`
- bewertet korrekt/inkorrekt (0/1)

Log-Likelihood-Fitness (`gsm8k_loglikelihood.py`):
- teacher-forcing forward pass auf Gold-Target
- Score = mittlere Log-Wahrscheinlichkeit der Ziel-Tokens
- batched und gecached moeglich

Konsequenz:
- Binary ist realitaetsnah fuer Endmetrik, aber langsam und diskret
- Log-Likelihood ist schnell und kontinuierlich, daher besseres Trainingssignal

Beleg im Code:
- `src/es_llm/fitness/gsm8k.py`
- `src/es_llm/fitness/gsm8k_loglikelihood.py`

## Slide 7 (7:40-9:00) - Experimentelles Protokoll
Beispiel-High-Budget-Konfiguration (v5):
- Layer: `[18, 19, 20, 21]`, Komponente: `attention`
- Population: 96
- Generationen: 100
- sigma: 0.003 -> 0.001 (cosine decay)
- Fitness: `gsm8k_loglikelihood`, `num_samples=120`, `target_mode=full`, `batch_size=12`

Begruendung:
- Mehr Layer = mehr Anpassungskapazitaet
- Log-likelihood + Batching = praktikable Laufzeiten
- Reshuffle reduziert Overfitting auf fixen Fitness-Subset

Beleg in Config:
- `configs/gsm8k_a100_v5.yaml`

## Slide 8 (9:00-10:20) - Empirische Ergebnisse (vorliegende Artefakte)
Direkt aus `out/`:
- GSM8K (Qwen2.5-0.5B, limit=500): exact_match = 0.224
  Quelle: `out/gsm8k-1769434405/metrics.json`
- GSM8K (Qwen2.5-0.5B, limit=100): exact_match = 0.200
  Quelle: `out/gsm8k-1769242273/metrics.json`
- GSM8K (Qwen2.5-0.5B, limit=20, OLMES alias): exact_match = 0.200
  Quelle: `out/gsm8k::olmes-1770491138/metrics.json`
- Minerva algebra (Qwen2.5-0.5B, limit=20): exact_match = 0.000, flex = 0.050
  Quelle: `out/minerva_math_algebra::olmes-1770492626/metrics.json`

Interpretation:
- GSM8K-Leistung liegt im Bereich ~0.20 bis 0.224 je nach Sample-Budget.
- Cross-task-Transfer auf Minerva algebra ist schwach.
- Es braucht striktere Vorher/Nachher-Vergleiche auf identischem Evaluationsprotokoll fuer kausale Aussagen.

## Slide 9 (10:20-11:30) - Laufzeit- und Skalierungsanalyse
Warum Training teuer ist:
- ES braucht pro Generation Population * Fitness-Evaluations
- im Trainer wird jeder Kandidat einzeln evaluiert
- bei generativer Binary-Fitness wird pro Beispiel autoregressiv decodiert

Kostenmodell (qualitativ):
```text
Zeit pro Generation ~ Population * Samples * Kosten(Fitness)
```

Praktische Beobachtung:
- Log-likelihood ist deutlich schneller als Binary, weil kein schrittweises Decoding noetig ist.

Beleg im Code:
- `src/es_llm/training/es_trainer.py`
- `src/es_llm/fitness/gsm8k.py`
- `src/es_llm/fitness/gsm8k_loglikelihood.py`

## Slide 10 (11:30-12:40) - Wissenschaftliche Guete und Risiken
Staerken:
- reproduzierbare Konfigurationsstruktur
- klar getrennte Module (ES, Fitness, Layer Selection, Eval)
- mehrere Evaluationspfade (GSM8K + OLMES Tasks)

Limitierungen:
- derzeit keine konsistente Baseline-vs-ES Tabelle mit identischem Seed/Limit pro Run
- moeglicher Mismatch zwischen Trainingsfitness (LL) und Zielmetrik (Exact Match)
- kleine Limits in manchen Runs erhoehen Messrauschen

Methodische Risiken:
- Data subset bias bei Fitness-Sampling
- Prompt-/Tokenizer-Effekte auf numerisches Parsing
- Budgetabhaengigkeit (Population, num_samples, max tokens)

## Slide 11 (12:40-13:50) - Nächste Experimente (wissenschaftlich sauber)
Priorisierte Agenda:
1. Strenger A/B-Vergleich: baseline vs ES auf identischem Test-Slice (z. B. n=500, gleicher Seed)
2. Ablation: Layer [21] vs [20,21] vs [18-21] bei konstantem Budget
3. Fitness-Ablation: binary vs LL-short vs LL-full
4. Robustheit: 3-5 Seeds, Mittelwert plus Konfidenzintervall
5. Transfer: GSM8K-optimiertes Modell auf Minerva/HellaSwag mit fixem Eval-Protokoll

Erwartetes Ergebnis:
- belastbare Aussage, welche Komponenten tatsaechlich den Effekt treiben

## Slide 12 (13:50-15:00) - Fazit
Kernaussagen:
- Das Projekt implementiert eine saubere ES-Pipeline fuer layer-weises LLM-Finetuning ohne Backprop.
- Das Fitness-Design entscheidet ueber Praktikabilitaet; Log-Likelihood ist fuer Training deutlich effizienter.
- Aktuelle Ergebnisse zeigen brauchbare GSM8K-Basiswerte, aber fuer starke wissenschaftliche Claims fehlt noch ein striktes A/B-Experiment-Set.

Take-home message:
- Methodisch ist das Fundament stark.
- Der naechste Hebel ist experimentelle Strenge und statistische Absicherung.

---

## Wie du praesentieren solltest (Delivery-Plan)

Erzaehlstil:
- Problem -> Methode -> Evidenz -> Limitationen -> Plan
- Keine Tool-Demo zuerst, sondern wissenschaftliche Frage zuerst

Sprechtempo:
- 140 bis 160 Woerter pro Minute
- pro Slide 1 Hauptbotschaft, maximal 3 Unterpunkte

Visual Design (empfohlen):
- Folien 1-3: Motivation und RQ (Text + 1 Diagramm)
- Folien 4-6: Architektur + Formel + Fitnessvergleich (2 Schaubilder)
- Folien 7-9: Setup + Ergebnistabelle + Laufzeitplot
- Folien 10-12: Threats to Validity + Next Steps + Fazit

Q&A-Vorbereitung (wahrscheinliche Fragen):
- Warum nicht LoRA/Backprop als Baseline?
- Wie sensitiv sind Ergebnisse auf Seed und Sample-Limit?
- Wie wird verhindert, dass nur Prompt-Overfitting passiert?
- Gibt es Evidenz fuer echte Generalisierung auf andere Tasks?

---

## Backup-Slide Material (falls Zeit)
- OpenAI-ES vs CMA-ES Entscheidungskriterien
- Wirkung von `fitness_shaping` (raw vs normalized vs centered_rank)
- Einfluss von `target_mode` (`short` vs `full`) auf Stabilitaet und Geschwindigkeit
- Reproduzierbarkeitspaket: welche Artefakte pro Run gespeichert werden
