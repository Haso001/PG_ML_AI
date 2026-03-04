import re
import json
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

SYSTEM = "You are a helpful assistant."
USER_TMPL = (
    "Solve the following problem step by step.\n"
    "At the end, output ONLY the final numeric answer in exactly this format:\n"
    "#### <answer>\n\n"
    "Problem:\n{q}"
)

def extract_hash_answer(text: str):
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)
    # fallback: last number (useful for debugging; slightly less strict)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None

def normalize(x: str | None):
    if x is None:
        return None
    x = x.strip().replace(",", "").replace("$", "").replace(" ", "")
    # optional: remove spaces
    x = x.replace(" ", "")
    try:
        fx = float(x)
        if fx.is_integer():
            return str(int(fx))
    except Exception:
        pass
    return x


def reward_01(pred: str | None, gold: str | None) -> int:
    return int(pred is not None and gold is not None and pred == gold)

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--save_samples", type=int, default=5)
    ap.add_argument("--out", default="experiments/runs/qwen_frozen_test5.json")
    args = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main")[args.split]
    ds = ds.select(range(min(args.n, len(ds))))

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    correct = 0
    samples = []

    for i, ex in enumerate(tqdm(ds, desc=f"GSM8K/{args.split}")):
        q = ex["question"]
        gold = normalize(extract_hash_answer(ex["answer"]))

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(q=q)},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tok([prompt], return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        out_ids = gen[0, inputs["input_ids"].shape[1]:]
        pred_text = tok.decode(out_ids, skip_special_tokens=True)

        pred = normalize(extract_hash_answer(pred_text))
        r = reward_01(pred, gold)
        correct += r

        if i < args.save_samples:
            samples.append({
                "question": q,
                "gold": gold,
                "pred": pred,
                "reward": r,
                "raw_output": pred_text,
            })

    acc = correct / len(ds)
    summary = {
        "model": args.model,
        "task": "gsm8k",
        "split": args.split,
        "n": len(ds),
        "accuracy": acc,
        "max_new_tokens": args.max_new_tokens,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "samples": samples}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
