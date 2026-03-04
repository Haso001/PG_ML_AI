"""ES-LLM Fine-Tuning – Haupteinstiegspunkt.

Usage:
    python main.py train --config configs/gsm8k_last_layer.yaml
    python main.py inspect --model Qwen/Qwen2.5-0.5B-Instruct
    python main.py eval --run experiments/runs/gsm8k_last_layer_es
"""

import sys
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Befehle: train, inspect, eval")
        sys.exit(0)

    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # shift args for sub-scripts

    if cmd == "train":
        from scripts.train_es import main as train_main
        train_main()
    elif cmd == "inspect":
        from scripts.inspect_layers import main as inspect_main
        inspect_main()
    elif cmd == "eval":
        from scripts.eval_es_model import main as eval_main
        eval_main()
    else:
        print(f"Unbekannter Befehl: {cmd}")
        print("Verfügbar: train, inspect, eval")
        sys.exit(1)


if __name__ == "__main__":
    main()
