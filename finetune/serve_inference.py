"""Persistent stdin/stdout inference server for the fine-tuned LoRA adapter.

Designed to be launched as a long-lived subprocess by LocalQwenClient so that
the solar_tracker conda env (torch 1.13, no peft) can use the fine-tuned model
without any dependency conflicts.

Protocol
--------
- Reads one JSON object per line from stdin:
      {"messages": [...], "max_tokens": 300}
- Writes one JSON object per line to stdout:
      {"text": "...", "error": null}
- All startup/debug messages go to stderr so stdout stays clean.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from pathlib import Path as pathlib_Path

_ROOT = Path(__file__).resolve().parent.parent
import os
# Which fine-tuned adapter to serve. Defaults to the paper adapter;
# set SOLAR_SLM_ADAPTER to serve the oracle-tuned one instead.
_ADAPTER = pathlib_Path(os.environ["SOLAR_SLM_ADAPTER"]) if os.environ.get("SOLAR_SLM_ADAPTER") else _ROOT / "finetune" / "output" / "adapter"
_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _load():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("serve_inference: loading tokenizer ...", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(_ADAPTER), trust_remote_code=True)

    print("serve_inference: loading base model ...", file=sys.stderr, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        _BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("serve_inference: applying LoRA adapter ...", file=sys.stderr, flush=True)
    model = PeftModel.from_pretrained(model, str(_ADAPTER))
    model.eval()

    print("serve_inference: ready", file=sys.stderr, flush=True)
    return tokenizer, model


def _generate(tokenizer, model, messages, max_tokens, temperature=0.0):
    import torch

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=int(max_tokens),
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0.0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
    else:
        gen_kwargs["do_sample"] = False
    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    tokenizer, model = _load()

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
            text = _generate(
                tokenizer, model,
                req["messages"],
                req.get("max_tokens", 128),
                req.get("temperature", 0.0),
            )
            result = {"text": text, "error": None}
        except Exception as exc:
            result = {"text": "", "error": str(exc)}

        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
