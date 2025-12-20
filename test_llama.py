#!/usr/bin/env python3

import argparse
import random
import re
import string
import types
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from models.attn_patch import entropy_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

ALL_ATTENTION_FUNCTIONS.register("entropy_attn", entropy_attention_forward)


def _random_filler_words(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    words = []
    alphabet = string.ascii_lowercase
    for _ in range(n_words):
        wlen = rng.randint(3, 10)
        w = "".join(rng.choice(alphabet) for _ in range(wlen))
        words.append(w)
    return " ".join(words)


def build_passkey_prompt(tokenizer, passkey: str, filler_words: int, seed: int) -> str:
    user_text = (
        "You will see a long text. Inside it there is exactly one line of the form:\n"
        "PASSKEY: <digits>\n"
        "Your task: output ONLY the digits of the passkey (no other words).\n\n"
    )

    filler1 = _random_filler_words(filler_words // 2, seed=seed)
    filler2 = _random_filler_words(filler_words - filler_words // 2, seed=seed + 1)

    long_text = (
        f"{filler1}\n\n"
        f"PASSKEY: {passkey}\n\n"
        f"{filler2}\n"
    )

    question = "What is the passkey? Output only the digits."

    # If tokenizer supports chat templates, use them (instruct models).
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": "You are a precise assistant that follows instructions exactly."},
            {"role": "user", "content": user_text + long_text + "\n" + question},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Fallback: plain prompt
    return (
        "You are a precise assistant that follows instructions exactly.\n\n"
        + user_text
        + long_text
        + "\n"
        + question
        + "\nAnswer: "
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="HF model name or local path")
    parser.add_argument("--attn-impl", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2", "entropy_attn"])
    parser.add_argument("--max-new", type=int, default=16)
    parser.add_argument("--filler-words", type=int, default=4096)
    parser.add_argument("--passkey", type=str, default="938274")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.device="cuda:0"

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model)
    config._attn_implementation = args.attn_impl
    config.attn_implementation = args.attn_impl

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        dtype=torch.bfloat16,
        device_map=None,  # keep simple & explicit
    ).to(args.device)
    model.eval()

    prompt = build_passkey_prompt(tokenizer, args.passkey, args.filler_words, seed=args.seed)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(args.device) for k, v in inputs.items()}
    
    print(f"[info] prompt_tokens={inputs['input_ids'].shape[-1]}  device={args.device}")
    print(f"[info] attn_impl={args.attn_impl}")

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new,
            do_sample=False,           # greedy
            use_cache=True,  # patch path is cache-ignorant
            pad_token_id=tokenizer.eos_token_id,
        )

    gen = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    # Extract digits from generation (best-effort)
    m = re.search(r"(\d{3,})", gen)
    extracted = m.group(1) if m else None

    print("\n=== Generation ===")
    print(gen)
    print("==================\n")

    print(f"[check] expected_passkey={args.passkey}")
    print(f"[check] extracted_passkey={extracted}")

    ok_passkey = (extracted == args.passkey)
    print(f"[result] passkey_ok={ok_passkey}")

if __name__ == "__main__":
    main()

