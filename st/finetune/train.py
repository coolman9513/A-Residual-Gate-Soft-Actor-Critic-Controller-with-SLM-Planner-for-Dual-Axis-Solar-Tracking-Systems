"""
LoRA fine-tune Qwen2.5-0.5B-Instruct on solar tracker planning data.

Uses plain PyTorch training loop — no trl/datasets dependency,
avoiding Windows pyarrow DLL issues.

Requirements (sllm_finetune conda env):
    transformers peft torch accelerate

Usage:
    python train.py
    python train.py --epochs 5 --lr 2e-4

Outputs:
    finetune/output/adapter/   — LoRA adapter
    finetune/output/logs.json  — training loss curve
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

MAX_SEQ_LEN = 768
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data_dir", default=None,
                   help="Directory with train.jsonl/val.jsonl (default: finetune/data)")
    p.add_argument("--out_dir", default=None,
                   help="Where to write adapter/ and logs.json (default: finetune/output)")
    p.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN,
                   help="Truncation length. MUST exceed prompt+reply or the "
                        "assistant turn is cut off and nothing is supervised.")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=20)
    return p.parse_args()


class ChatDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append(obj["messages"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = enc["input_ids"]
        labels = input_ids.copy()

        # Mask everything before the last assistant turn so loss is
        # only on the assistant reply tokens.
        assistant_header = "<|im_start|>assistant\n"
        header_ids = self.tokenizer.encode(assistant_header, add_special_tokens=False)
        # Find last occurrence of assistant header token sequence
        mask_until = 0
        for i in range(len(input_ids) - len(header_ids)):
            if input_ids[i : i + len(header_ids)] == header_ids:
                mask_until = i + len(header_ids)
        labels[:mask_until] = [-100] * mask_until

        if mask_until == 0 or all(l == -100 for l in labels):
            # Truncation cut the assistant turn away: nothing is supervised and
            # the loss silently falls back onto prompt tokens. Fail loudly.
            raise ValueError(
                "Sample %d has no supervised tokens: %d tokens truncated to %d. "
                "Raise --max_seq_len above the longest prompt+reply."
                % (idx, len(enc["input_ids"]), self.max_length)
            )

        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch, pad_id: int):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad_len)
        labels.append(b["labels"] + [-100] * pad_len)
        attention_mask.append([1] * len(b["input_ids"]) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    global DATA_DIR, OUT_DIR, MAX_SEQ_LEN
    MAX_SEQ_LEN = int(args.max_seq_len)
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data  : {DATA_DIR}")
    print(f"Out   : {OUT_DIR}")
    print(f"Model : {args.model}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\nLoading dataset...")
    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Run generate_dataset.py first. Expected: {train_path}")

    train_ds = ChatDataset(train_path, tokenizer, MAX_SEQ_LEN)
    val_ds = ChatDataset(val_path, tokenizer, MAX_SEQ_LEN)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    _collate = lambda b: collate_fn(b, pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=_collate, num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\nStarting training... ({total_steps} optimizer steps, {len(train_loader)} batches/epoch)")
    log_history = []
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        accum_count = 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            running_loss += loss.item() * args.grad_accum
            accum_count += 1

            if accum_count == args.grad_accum or step == len(train_loader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                accum_count = 0

                if global_step % args.logging_steps == 0:
                    avg_loss = running_loss / args.logging_steps / args.grad_accum
                    running_loss = 0.0
                    print(f"  epoch {epoch+1} step {global_step} — loss {avg_loss:.4f}")
                    log_history.append({"step": global_step, "loss": avg_loss})

                if global_step % args.save_steps == 0:
                    # Eval
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for vb in val_loader:
                            vb = {k: v.to(model.device) for k, v in vb.items()}
                            val_loss += model(**vb).loss.item()
                    val_loss /= len(val_loader)
                    model.train()
                    print(f"  eval step {global_step} — val_loss {val_loss:.4f}")
                    log_history.append({"step": global_step, "eval_loss": val_loss})
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        adapter_path = OUT_DIR / "adapter"
                        model.save_pretrained(str(adapter_path))
                        tokenizer.save_pretrained(str(adapter_path))
                        print(f"  [saved best adapter at step {global_step}]")

    # Final save if we haven't saved at all
    adapter_path = OUT_DIR / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"  Adapter saved -> {adapter_path}")

    with open(OUT_DIR / "logs.json", "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"  Logs saved  -> {OUT_DIR / 'logs.json'}")


if __name__ == "__main__":
    main()
