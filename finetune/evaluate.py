"""
Evaluate fine-tuned Qwen2.5-0.5B on solar tracker planning.

Tests:
  1. JSON parse rate (must be ~100%)
  2. Budget accuracy per DNI tier
  3. Hold/track accuracy
  4. Inference speed (tokens/sec)

Usage:
    python evaluate.py
    python evaluate.py --samples 200
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = Path(__file__).resolve().parent / "output"
DATA_DIR = Path(__file__).resolve().parent / "data"
ADAPTER_DIR = OUT_DIR / "adapter"
BUDGET_TIERS = [0.0, 3.0, 6.0, 12.0, 20.0, 30.0]


def load_model(adapter_path: Path):
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


_FIELD_PATTERNS = {
    "target_tilt":       r'"target_tilt"\s*:\s*([\d.]+)',
    "target_az":         r'"target_az"\s*:\s*([\d.]+)',
    "motion_budget_deg": r'"motion_budget_deg"\s*:\s*([\d.]+)',
    "pre_position_tilt": r'"pre_position_tilt"\s*:\s*([\d.]+)',
    "pre_position_az":   r'"pre_position_az"\s*:\s*([\d.]+)',
    "hold":              r'"hold"\s*:\s*(true|false)',
    "horizon_min":       r'"horizon_min"\s*:\s*(\d+)',
    "confidence":        r'"confidence"\s*:\s*([\d.]+)',
}
_REQUIRED = {"target_tilt", "target_az", "motion_budget_deg"}


def extract_json(text: str) -> dict | None:
    # Try standard parse first
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: extract individual fields from malformed output
    fields: dict = {}
    for key, pat in _FIELD_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            v = m.group(1)
            if key == "hold":
                fields[key] = v == "true"
            elif key == "horizon_min":
                fields[key] = int(v)
            else:
                fields[key] = float(v)

    return fields if _REQUIRED.issubset(fields) else None


def infer(model, tokenizer, messages: list[dict], max_new_tokens: int = 300) -> tuple[str, float]:
    prompt = tokenizer.apply_chat_template(
        messages[:-1],   # system + user only
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — less hallucination
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0.0
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response, tok_per_sec


def evaluate(n_samples: int = 100) -> None:
    if not ADAPTER_DIR.exists():
        print(f"ERROR: Adapter not found at {ADAPTER_DIR}. Run train.py first.")
        return

    model, tokenizer = load_model(ADAPTER_DIR)

    # Load val set
    val_path = DATA_DIR / "val.jsonl"
    samples = []
    with open(val_path) as f:
        for line in f:
            samples.append(json.loads(line))
    samples = samples[:n_samples]
    print(f"\nEvaluating {len(samples)} samples...\n")

    results = []
    tok_speeds = []
    budget_correct = {b: {"correct": 0, "total": 0} for b in BUDGET_TIERS}
    hold_correct = hold_total = 0
    parse_failures = 0

    for i, sample in enumerate(samples):
        messages = sample["messages"]
        # Ground truth is in assistant message
        gt_text = messages[-1]["content"]
        gt_json = extract_json(gt_text)
        if gt_json is None:
            continue

        response, tok_s = infer(model, tokenizer, messages)
        tok_speeds.append(tok_s)
        parsed = extract_json(response)

        if parsed is None:
            parse_failures += 1
            results.append({"gt": gt_json, "pred": None, "parse_ok": False})
            continue

        results.append({"gt": gt_json, "pred": parsed, "parse_ok": True})

        # Budget accuracy
        gt_budget = float(gt_json.get("motion_budget_deg", -1))
        pred_budget = float(parsed.get("motion_budget_deg", -1))
        nearest_tier = min(BUDGET_TIERS, key=lambda t: abs(t - pred_budget))
        if gt_budget in budget_correct:
            budget_correct[gt_budget]["total"] += 1
            if abs(nearest_tier - gt_budget) < 0.5:
                budget_correct[gt_budget]["correct"] += 1

        # Hold accuracy
        gt_hold = bool(gt_json.get("hold", False))
        pred_hold = bool(parsed.get("hold", False))
        hold_total += 1
        if gt_hold == pred_hold:
            hold_correct += 1

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(samples)} — parse_ok={len(results)-parse_failures}/{i+1}")

    # ── report ────────────────────────────────────────────────────────────────
    parse_rate = (len(results) - parse_failures) / len(results) * 100
    hold_acc = hold_correct / max(hold_total, 1) * 100
    mean_tok_s = sum(tok_speeds) / max(len(tok_speeds), 1)

    print("\n" + "=" * 50)
    print(f"JSON parse rate      : {parse_rate:.1f}%")
    print(f"Hold/track accuracy  : {hold_acc:.1f}%")
    print(f"Mean inference speed : {mean_tok_s:.1f} tok/s")
    print()
    print("Budget accuracy per tier:")
    for b, counts in budget_correct.items():
        if counts["total"] == 0:
            continue
        acc = counts["correct"] / counts["total"] * 100
        print(f"  budget={b:4.0f}: {acc:5.1f}%  ({counts['correct']}/{counts['total']})")

    # Save results
    report = {
        "parse_rate_pct": parse_rate,
        "hold_accuracy_pct": hold_acc,
        "mean_tokens_per_sec": mean_tok_s,
        "budget_accuracy": {
            str(b): {
                "accuracy_pct": (c["correct"] / max(c["total"], 1)) * 100,
                "correct": c["correct"],
                "total": c["total"],
            }
            for b, c in budget_correct.items()
        },
        "parse_failures": parse_failures,
        "n_evaluated": len(results),
    }
    out = OUT_DIR / "eval_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved → {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.samples)
