"""
Benchmark TurboQuant (TBQ) vs baseline on MM-NIAH (Multimodal Needle-in-a-Haystack).

INSTALL
    pip install -U mlx-vlm
    # or
    uv pip install -U mlx-vlm

SETUP — Extract images (one-time)
    huggingface-cli download OpenGVLab/MM-NIAH mm_niah_val/images.tar.gz --repo-type dataset

    mkdir -p /tmp/mm_niah_val_images
    tar xzf ~/.cache/huggingface/hub/datasets--OpenGVLab--MM-NIAH/snapshots/*/mm_niah_val/images.tar.gz \\
        -C /tmp/mm_niah_val_images

USAGE
    # Full benchmark (baseline vs TBQ)
    python bench_mm_niah.py --model google/gemma-4-26b-a4b-it

    # Quick smoke test (10 samples, 4-bit KV cache)
    python bench_mm_niah.py --model google/gemma-4-26b-a4b-it --num-samples 10 --kv-bits 4

    # Custom image directory
    python bench_mm_niah.py --model google/gemma-4-26b-a4b-it --image-root /path/to/images
"""

import argparse
import importlib
import os
import time

import mlx.core as mx
import numpy as np
from datasets import load_dataset
from PIL import Image

from mlx_vlm.models.cache import make_prompt_cache

from mlx_vlm import load

mod = importlib.import_module("mlx_vlm.generate")

DEFAULT_IMAGE_ROOT = "/tmp/mm_niah_val_images/mm_niah_dev/images"

BUCKETS = [
    (0, 2000, "~1K"),
    (2000, 5000, "~3K"),
    (5000, 10000, "~7K"),
    (10000, 20000, "~15K"),
    (20000, 40000, "~30K"),
    (40000, 100000, "~60K"),
]


def select_samples(ds, tokenizer, num_per_bucket):
    """Select samples bucketed by actual token count (not meta context_length)."""
    # Tokenize all contexts to get real token counts
    all_samples = []
    for i, s in enumerate(ds):
        tok_count = len(tokenizer.encode(s["context"]))
        all_samples.append((i, tok_count))

    indices = []
    for lo, hi, label in BUCKETS:
        bucket = [(i, n) for i, n in all_samples if lo <= n < hi]
        bucket.sort(key=lambda x: x[1])
        if len(bucket) >= num_per_bucket:
            step = max(1, len(bucket) // num_per_bucket)
            picked = [bucket[j] for j in range(0, len(bucket), step)][:num_per_bucket]
        else:
            picked = bucket
        for idx, tok_count in picked:
            indices.append((idx, tok_count, label))
    return indices


def load_images(sample, image_root):
    images = []
    for img_path in sample["images_list"]:
        full_path = os.path.join(image_root, img_path)
        if os.path.exists(full_path):
            images.append(Image.open(full_path).convert("RGB"))
    return images


def measure_cache_bytes(prompt_cache) -> int:
    """Sum actual .nbytes across all layers of the prompt cache."""
    total = 0
    for entry in prompt_cache:
        total += entry.nbytes
    return total


def run_sample(input_ids, model, pv, mask, kv_args):
    n = input_ids.shape[1]

    prompt_cache = make_prompt_cache(model.language_model)
    gen = mod.generate_step(
        input_ids, model, pv, mask, max_tokens=20, temperature=0.0,
        prompt_cache=prompt_cache, **kv_args
    )

    t0 = time.perf_counter()
    token, _ = next(gen)
    mx.eval(token if isinstance(token, mx.array) else mx.array(token))
    t_prefill = time.perf_counter() - t0
    prefill_tps = n / t_prefill

    t0 = time.perf_counter()
    toks = [token.item() if isinstance(token, mx.array) else token]
    count = 0
    for tok, _ in gen:
        mx.eval(tok if isinstance(tok, mx.array) else tok)
        toks.append(tok.item() if isinstance(tok, mx.array) else tok)
        count += 1
    t_decode = time.perf_counter() - t0
    decode_tps = count / t_decode if t_decode > 0 else 0

    kv_bytes = measure_cache_bytes(prompt_cache)

    return toks, prefill_tps, decode_tps, kv_bytes


def main():
    parser = argparse.ArgumentParser(description="Benchmark TurboQuant on MM-NIAH")
    parser.add_argument("--model", type=str, default="google/gemma-4-26b-a4b-it")
    parser.add_argument("--num-samples", type=int, default=1, help="Samples per bucket")
    parser.add_argument("--kv-bits", type=float, default=3.5)
    parser.add_argument("--image-root", type=str, default=DEFAULT_IMAGE_ROOT)
    args = parser.parse_args()

    if not os.path.exists(args.image_root):
        print(f"Image root not found: {args.image_root}")
        print("Extract images first (see script docstring for instructions).")
        return

    ds = load_dataset("OpenGVLab/MM-NIAH", split="val")
    model, processor = load(args.model)

    indices = select_samples(ds, processor.tokenizer, args.num_samples)
    print(f"Model: {args.model}")
    print(f"Samples: {len(indices)} ({args.num_samples} per bucket)")
    print(f"TBQ: {args.kv_bits}-bit\n")

    modes = [
        ("BL", {}),
        ("TBQ", {"kv_bits": args.kv_bits, "kv_quant_scheme": "turboquant"}),
    ]

    results = []
    for idx, ctx_len, label in indices:
        s = ds[idx]
        images = load_images(s, args.image_root)
        if not images:
            print(f"  Skipping {label} idx={idx}, no images found", flush=True)
            continue

        prompt = f"{s['context']}\n\nQuestion: {s['question']}\nAnswer briefly:"
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )

        try:
            inputs = processor(text=text, images=images, return_tensors="np")
        except Exception as e:
            print(f"  Skipping {label} idx={idx}, processor error: {e}", flush=True)
            continue

        input_ids = mx.array(inputs["input_ids"])
        pv = inputs.get("pixel_values", None)
        if pv is not None:
            if isinstance(pv, list):
                # Multi-image: resize all to match the first image's shape, then stack
                arrays = [np.asarray(p) if not isinstance(p, np.ndarray) else p for p in pv]
                target_shape = arrays[0].shape  # (C, H, W) or (1, C, H, W)
                resized = []
                for a in arrays:
                    if a.shape == target_shape:
                        resized.append(a)
                    else:
                        # Pad or crop to target shape
                        result = np.zeros(target_shape, dtype=a.dtype)
                        slices = tuple(slice(0, min(a.shape[i], target_shape[i])) for i in range(len(target_shape)))
                        result[slices] = a[slices]
                        resized.append(result)
                pv = mx.array(np.stack(resized) if resized[0].ndim == 3 else np.concatenate(resized, axis=0))
            elif not isinstance(pv, mx.array):
                pv = mx.array(np.asarray(pv))
        mask = (
            mx.array(inputs["attention_mask"])
            if "attention_mask" in inputs
            else None
        )
        n = input_ids.shape[1]
        gold = s["answer"]

        for mode_name, kv_args in modes:
            mx.clear_cache()

            try:
                toks, prefill_tps, decode_tps, kv_bytes = run_sample(
                    input_ids, model, pv, mask, kv_args
                )
            except Exception as e:
                print(f" Skipping {label} idx={idx} {mode_name}, error: {e}", flush=True)
                continue
            kv_gb = kv_bytes / (1 << 30)

            answer = processor.tokenizer.decode(toks).strip().replace("\n", " ")[:60]
            correct = gold.lower() in answer.lower()

            results.append(
                {
                    "idx": idx,
                    "n": n,
                    "mode": mode_name,
                    "prefill": prefill_tps,
                    "decode": decode_tps,
                    "kv": kv_gb,
                    "correct": correct,
                    "gold": gold,
                    "answer": answer,
                    "label": label,
                    "num_images": len(images),
                }
            )

            mark = "Y" if correct else "N"
            print(
                f"{label:>5} {n:>6} ({len(images):>2} img) | {mode_name:<3} | "
                f"pf {prefill_tps:>7.1f} | dec {decode_tps:>5.1f} | "
                f"KV {kv_gb:>5.3f}G | {mark} | "
                f"gold={gold[:15]:>15} | {answer[:40]}",
                flush=True,
            )
        print(flush=True)

    # Summary
    bl_results = [r for r in results if r["mode"] == "BL"]
    tbq_results = [r for r in results if r["mode"] == "TBQ"]
    n_pairs = min(len(bl_results), len(tbq_results))

    if n_pairs == 0:
        print("No completed pairs.")
        return

    print(
        f"\n{'Bucket':>5} {'Tok':>6} {'Img':>3} | {'BL pf':>6} {'TBQ pf':>6} | "
        f"{'BL dec':>6} {'TBQ dec':>7} {'Ratio':>5} | "
        f"{'KV BL':>6} {'KV TBQ':>6} {'Save':>5} | {'BL':>2} {'TBQ':>3} {'Gold':>8}"
    )
    print("-" * 105)
    for b, t in zip(bl_results[:n_pairs], tbq_results[:n_pairs]):
        ratio = t["decode"] / b["decode"] if b["decode"] > 0 else 0
        save = (1 - t["kv"] / b["kv"]) * 100 if b["kv"] > 0 else 0
        bm = "Y" if b["correct"] else "N"
        tm = "Y" if t["correct"] else "N"
        print(
            f"{b['label']:>5} {b['n']:>6} {b['num_images']:>3} | "
            f"{b['prefill']:>6.0f} {t['prefill']:>6.0f} | "
            f"{b['decode']:>6.1f} {t['decode']:>7.1f} {ratio:>4.2f}x | "
            f"{b['kv']:>5.3f}G {t['kv']:>5.3f}G {save:>4.0f}% | "
            f"{bm:>2} {tm:>3} {b['gold'][:8]:>8}"
        )

    bl_correct = sum(1 for r in bl_results[:n_pairs] if r["correct"])
    tbq_correct = sum(1 for r in tbq_results[:n_pairs] if r["correct"])
    agree = sum(
        1
        for b, t in zip(bl_results[:n_pairs], tbq_results[:n_pairs])
        if b["correct"] == t["correct"]
    )
    tbq_wins = sum(
        1
        for b, t in zip(bl_results[:n_pairs], tbq_results[:n_pairs])
        if t["correct"] and not b["correct"]
    )
    bl_wins = sum(
        1
        for b, t in zip(bl_results[:n_pairs], tbq_results[:n_pairs])
        if b["correct"] and not t["correct"]
    )

    bl_kv_avg = sum(r["kv"] for r in bl_results[:n_pairs]) / n_pairs
    tbq_kv_avg = sum(r["kv"] for r in tbq_results[:n_pairs]) / n_pairs
    kv_save = (1 - tbq_kv_avg / bl_kv_avg) * 100 if bl_kv_avg > 0 else 0

    print(
        f"\nAccuracy:  BL={bl_correct}/{n_pairs} ({bl_correct/n_pairs*100:.0f}%)  "
        f"TBQ={tbq_correct}/{n_pairs} ({tbq_correct/n_pairs*100:.0f}%)"
    )
    print(f"Agreement: {agree}/{n_pairs} ({agree/n_pairs*100:.0f}%)")
    print(f"TBQ wins:  {tbq_wins}  |  BL wins: {bl_wins}  |  Net: TBQ +{tbq_wins - bl_wins}")
    print(f"KV cache:  BL={bl_kv_avg:.3f}G  TBQ={tbq_kv_avg:.3f}G  ({kv_save:.0f}% savings)")


if __name__ == "__main__":
    main()
