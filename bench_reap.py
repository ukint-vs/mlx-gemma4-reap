#!/usr/bin/env python3
"""Benchmark REAP MLX models — matches 0xSero's evaluation methodology.

Tasks: MMLU subtasks + GSM8K, 0-shot generative with thinking enabled.
Extraction: regex matching after stripping thinking channel tokens.
Reference: https://huggingface.co/0xSero/gemma-4-21b-a4b-it-REAP
"""
import sys, os, json, time, re
from datasets import load_dataset
from mlx_vlm import load, generate


# MMLU subtasks matching 0xSero's evaluation
MMLU_TASKS = [
    "elementary_mathematics",
    "philosophy",
    "world_religions",
    "college_computer_science",
    "high_school_mathematics",
    "abstract_algebra",
    "college_mathematics",
]


def strip_thinking(text):
    """Remove Gemma 4 thinking channel to get the final response."""
    # If there's a response channel, take everything after the last one
    if "<|channel>response" in text:
        text = text.split("<|channel>response")[-1]
        text = text.replace("<channel|>", "").strip()
        return text
    # Otherwise strip thinking blocks
    text = re.sub(r'<\|channel>thought.*?<channel\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|?channel\|?>', '', text)
    text = re.sub(r'<\|?turn\|?>', '', text)
    return text.strip()


def extract_letter(text, valid="ABCD"):
    """Extract a single letter answer from model response."""
    text = strip_thinking(text)

    # Pattern 1: "The answer is X" or "Answer: X"
    m = re.search(r'(?:answer|choice)\s*(?:is|:)\s*\(?([A-D])\)?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern 2: starts with a letter (possibly bold/formatted)
    m = re.match(r'[\s*]*\(?([A-D])\)?[\s.*:)]', text)
    if m:
        return m.group(1).upper()

    # Pattern 3: standalone letter on first line
    first_line = text.split('\n')[0].strip() if text.strip() else ""
    m = re.match(r'^\**\(?([A-D])\)?\**\.?$', first_line)
    if m:
        return m.group(1).upper()

    # Pattern 4: first standalone letter A-D (word boundary)
    m = re.search(r'\b([A-D])\b', text[:200])
    if m:
        return m.group(1).upper()

    return ""


def extract_number(text):
    """Extract final numeric answer from GSM8K response."""
    text = strip_thinking(text)

    # Pattern 1: after ####
    if "####" in text:
        after = text.split("####")[-1].strip().replace(",", "")
        m = re.search(r'-?\d+\.?\d*', after)
        if m:
            return m.group()

    # Pattern 2: "answer is X" or "= X"
    m = re.search(r'(?:answer is|=)\s*\$?\s*(-?\d[\d,]*\.?\d*)', text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")

    # Pattern 3: boxed answer \boxed{X}
    m = re.search(r'\\boxed\{(-?\d[\d,]*\.?\d*)\}', text)
    if m:
        return m.group(1).replace(",", "")

    # Pattern 4: last number in the response
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def run_mmlu_task(model, processor, tokenizer, task_name, limit=50):
    """Run a single MMLU subtask with generative evaluation."""
    ds = load_dataset("cais/mmlu", task_name, split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0
    extraction_fails = 0
    labels = ["A", "B", "C", "D"]

    for sample in ds:
        question = sample["question"]
        choices = sample["choices"]
        answer_idx = sample["answer"]  # 0-3
        answer_key = labels[answer_idx]

        choice_str = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
        prompt_text = f"{question}\n\n{choice_str}\n\nWhat is the correct answer? Reply with just the letter (A, B, C, or D)."
        prompt = f"<bos><|turn>user\n{prompt_text}<turn|>\n<|turn>model\n"

        out = generate(model, processor, prompt, max_tokens=100, verbose=False, temperature=0.0)
        response = out.text if hasattr(out, "text") else str(out)

        pred = extract_letter(response)
        if not pred:
            extraction_fails += 1

        if pred == answer_key:
            correct += 1
        total += 1

    return correct, total, extraction_fails


def run_gsm8k(model, processor, tokenizer, limit=50):
    """GSM8K with flexible extraction matching 0xSero's approach."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0
    extraction_fails = 0

    for sample in ds:
        question = sample["question"]
        answer_str = sample["answer"].split("####")[-1].strip().replace(",", "")

        prompt_text = f"{question}\n\nSolve step by step. Put your final answer after ####."
        prompt = f"<bos><|turn>user\n{prompt_text}<turn|>\n<|turn>model\n"

        out = generate(model, processor, prompt, max_tokens=500, verbose=False, temperature=0.0)
        response = out.text if hasattr(out, "text") else str(out)

        pred = extract_number(response)
        if not pred:
            extraction_fails += 1

        if pred == answer_str:
            correct += 1
        total += 1

    return correct, total, extraction_fails


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=50, help="Samples per task (0=all)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None

    print(f"Model: {args.model}")
    print(f"Limit: {limit or 'all'} samples/task")
    print(f"Method: 0-shot generative, thinking enabled, regex extraction\n")

    model, processor = load(args.model)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    results = {}

    # MMLU subtasks
    for task_name in MMLU_TASKS:
        display = task_name.replace("_", " ").title()
        print(f"--- {display} ---", flush=True)
        t0 = time.time()
        correct, total, fails = run_mmlu_task(model, processor, tokenizer, task_name, limit)
        elapsed = time.time() - t0
        acc = correct / total * 100 if total > 0 else 0
        fail_note = f" ({fails} extraction fails)" if fails else ""
        print(f"  {correct}/{total} ({acc:.0f}%) in {elapsed:.0f}s{fail_note}\n", flush=True)
        results[task_name] = {
            "correct": correct, "total": total,
            "accuracy": round(acc, 1), "extraction_fails": fails,
        }

    # GSM8K
    print(f"--- GSM8K ---", flush=True)
    t0 = time.time()
    correct, total, fails = run_gsm8k(model, processor, tokenizer, limit)
    elapsed = time.time() - t0
    acc = correct / total * 100 if total > 0 else 0
    fail_note = f" ({fails} extraction fails)" if fails else ""
    print(f"  {correct}/{total} ({acc:.0f}%) in {elapsed:.0f}s{fail_note}\n", flush=True)
    results["gsm8k"] = {
        "correct": correct, "total": total,
        "accuracy": round(acc, 1), "extraction_fails": fails,
    }

    # Summary
    print(f"\n{'='*60}")
    print(f"  {args.model}")
    print(f"  0-shot generative | thinking enabled | {limit or 'all'} samples")
    print(f"{'='*60}")
    print(f"  {'Task':<25s} {'Score':>10s}  {'Extr.Fail':>10s}")
    print(f"  {'-'*25} {'-'*10}  {'-'*10}")
    for task, r in results.items():
        display = task.replace("_", " ").title()
        print(f"  {display:<25s} {r['correct']:>3}/{r['total']:<3} ({r['accuracy']:>4.0f}%)  {r['extraction_fails']:>5}")
    print()

    if args.output:
        out = {"model": args.model, "limit": limit, "method": "0-shot generative, thinking enabled", "results": results}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {args.output}")
