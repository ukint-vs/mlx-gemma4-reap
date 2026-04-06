#!/usr/bin/env python3
"""Quick tok/s comparison across models. Measures prefill + decode speed."""
import sys, time, argparse
import mlx.core as mx
from mlx_vlm import load, generate


PROMPTS = [
    ("short", "What is the capital of France?"),
    ("medium", "Write a 200 word essay about climate change and its impact on biodiversity."),
    ("long", "Explain the complete history of the Roman Empire from founding to fall, covering key emperors, military campaigns, political structures, cultural achievements, and the factors that led to its eventual decline. Be thorough and detailed."),
]


def measure(model, processor, prompt_text, max_tokens=200, warmup=False):
    prompt = f"<bos><|turn>user\n{prompt_text}<turn|>\n<|turn>model\n"
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    input_ids = tokenizer.encode(prompt)
    n_input = len(input_ids)

    t0 = time.perf_counter()
    out = generate(model, processor, prompt, max_tokens=max_tokens, verbose=False, temperature=0.0)
    elapsed = time.perf_counter() - t0

    text = out.text if hasattr(out, "text") else str(out)
    output_ids = tokenizer.encode(text)
    n_output = len(output_ids)

    prefill_est = n_input / elapsed  # rough estimate
    decode_tps = n_output / elapsed if elapsed > 0 else 0

    return n_input, n_output, elapsed, decode_tps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3, help="Runs per prompt (best of N)")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}, Best of {args.runs} runs\n")

    model, processor = load(args.model)

    # Warmup
    measure(model, processor, "Hello", max_tokens=5, warmup=True)

    results = []
    for label, prompt_text in PROMPTS:
        best_tps = 0
        best_result = None
        for r in range(args.runs):
            mx.clear_cache()
            n_in, n_out, elapsed, tps = measure(model, processor, prompt_text, args.max_tokens)
            if tps > best_tps:
                best_tps = tps
                best_result = (n_in, n_out, elapsed, tps)

        n_in, n_out, elapsed, tps = best_result
        print(f"  {label:8s} | {n_in:>4} in -> {n_out:>4} out | {elapsed:>5.1f}s | {tps:>5.1f} tok/s")
        results.append({"label": label, "input_tokens": n_in, "output_tokens": n_out, "time": round(elapsed, 2), "tok_per_sec": round(tps, 1)})

    avg_tps = sum(r["tok_per_sec"] for r in results) / len(results)
    print(f"\n  Average: {avg_tps:.1f} tok/s")


if __name__ == "__main__":
    main()
