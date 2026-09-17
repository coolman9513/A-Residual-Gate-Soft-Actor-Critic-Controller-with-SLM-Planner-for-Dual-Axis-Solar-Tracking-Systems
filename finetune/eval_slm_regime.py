# -*- coding: utf-8 -*-
"""Measure the oracle-tuned SLM's regime accuracy on the held-out split.

Reports JSON parse failures, which matter because under guardrail_mode
'loose' the SLM's output is no longer repaired by the physics guardrails.

Usage:
    python finetune/eval_slm_regime.py [--n 339] [--split val]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADAPTER = ROOT / "finetune" / "output_oracle" / "adapter"
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
DATA = ROOT / "finetune" / "data_oracle"
OUT = ROOT / "models" / "slm_regime_eval.json"
REGIMES = ["hold", "cloud", "clear"]


def to_regime(goal):
    """Mirror llm.regime_router.route_goal: the SLM's only real control lever."""
    if goal is None:
        return None
    if goal.get("hold") or float(goal.get("motion_budget_deg", 0.0)) <= 3.0:
        return "hold"
    return "clear" if float(goal.get("motion_budget_deg", 0.0)) >= 20.0 else "cloud"


def parse_json(text):
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in (DATA / ("%s.jsonl" % args.split)).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    # de-duplicate oversampled copies so the score is per distinct hour
    seen, uniq = set(), []
    for r in rows:
        key = r["messages"][1]["content"]
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    if args.n:
        uniq = uniq[:args.n]
    print("evaluating %d distinct %s prompts" % (len(uniq), args.split), flush=True)

    tok = AutoTokenizer.from_pretrained(str(ADAPTER), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16,
                                                 device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.eval()

    y_true, y_pred, bad_json = [], [], 0
    for i, r in enumerate(uniq):
        msgs = r["messages"][:-1]
        truth = to_regime(json.loads(r["messages"][-1]["content"]))
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=110, do_sample=False,
                                 eos_token_id=tok.eos_token_id,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        goal = parse_json(text)
        if goal is None:
            bad_json += 1
        pred = to_regime(goal)
        y_true.append(truth)
        y_pred.append(pred if pred else "hold")   # unparseable -> safest regime
        if (i + 1) % 50 == 0:
            print("  %d/%d" % (i + 1, len(uniq)), flush=True)

    # Persist raw predictions FIRST: generation is the expensive part and must
    # never be lost to a downstream metrics error.
    raw_path = OUT.parent / ("slm_regime_raw_%s.json" % args.split)
    raw_path.write_text(json.dumps({"y_true": y_true, "y_pred": y_pred,
                                    "json_parse_failures": bad_json}, indent=1),
                        encoding="utf-8")
    print("raw predictions -> %s" % raw_path, flush=True)

    # Metrics computed inline; sklearn is not installed in the fine-tune env.
    n = len(y_true)
    acc = sum(1 for t, p_ in zip(y_true, y_pred) if t == p_) / float(n)
    cm = [[sum(1 for t, p_ in zip(y_true, y_pred) if t == a and p_ == b)
           for b in REGIMES] for a in REGIMES]
    f1s = []
    for r in REGIMES:
        tp = sum(1 for t, p_ in zip(y_true, y_pred) if t == r and p_ == r)
        fp = sum(1 for t, p_ in zip(y_true, y_pred) if t != r and p_ == r)
        fn = sum(1 for t, p_ in zip(y_true, y_pred) if t == r and p_ != r)
        prec = tp / float(tp + fp) if tp + fp else 0.0
        rec = tp / float(tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    f1 = sum(f1s) / len(f1s)
    maj = max(set(y_true), key=y_true.count)
    maj_acc = float(sum(1 for t in y_true if t == maj) / n)

    print("\n=== oracle-tuned SLM (%s) ===" % args.split)
    print("  n %d | JSON parse failures %d (%.1f%%)" % (len(y_true), bad_json, 100.0 * bad_json / len(y_true)))
    print("  accuracy %.3f | macro-F1 %.3f | majority-class floor %.3f" % (acc, f1, maj_acc))
    print("  confusion (rows=true %s):" % REGIMES)
    for r, row in zip(REGIMES, cm):
        print("    %-6s %s" % (r, row))

    res = {"split": args.split, "n": len(y_true), "accuracy": round(acc, 4),
           "macro_f1": round(f1, 4), "majority_floor": round(maj_acc, 4),
           "json_parse_failures": bad_json, "confusion": cm, "labels": REGIMES}
    blob = json.loads(OUT.read_text()) if OUT.exists() else {}
    blob[args.split] = res
    OUT.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print("saved -> %s" % OUT)


if __name__ == "__main__":
    main()
