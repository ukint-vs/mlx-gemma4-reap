#!/usr/bin/env python3
"""Full trimodal validation: vision + audio + text for Gemma 4 MLX quantized models."""
import sys, time, os
from mlx_vlm import load, generate

AUDIO = "test_audio/rickroll_30s.mp3"
IMAGE = os.path.expanduser("~/Python/imaglius/data/test-images/016A2C8D-4845-4BB6-9FE6-22A0755BBA5B_1_105_c.jpeg")

def run(model, processor, name):
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    results = {}

    # ── 1. Vision ──
    print(f"\n  [Vision] Caption...")
    messages = [{"role": "user", "content": [
        {"type": "image", "url": IMAGE},
        {"type": "text", "text": "Write a single detailed caption for this image."},
    ]}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    t0 = time.time()
    out = generate(model, processor, prompt, [IMAGE],
        max_tokens=150, verbose=False, repetition_penalty=1.2, temperature=0.7)
    text = out.text if hasattr(out, "text") else str(out)
    el = time.time() - t0
    ok = len(text.strip()) > 30 and "oxff" not in text and text.count(text[:10]) < 3
    print(f"    {'✅' if ok else '❌'} {el:.1f}s | {text[:150]}")
    results["vision"] = {"ok": ok, "text": text, "time": round(el, 1)}

    # ── 2. Audio ──
    print(f"\n  [Audio] Describe audio...")
    messages = [{"role": "user", "content": [
        {"type": "audio", "url": AUDIO},
        {"type": "text", "text": "Describe what you hear in this audio. What song is it? Who is singing?"},
    ]}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    t0 = time.time()
    try:
        out = generate(model, processor, prompt, audio=[AUDIO],
            max_tokens=150, verbose=False, repetition_penalty=1.2, temperature=0.7)
        text = out.text if hasattr(out, "text") else str(out)
        el = time.time() - t0
        ok = len(text.strip()) > 20 and "oxff" not in text
        print(f"    {'✅' if ok else '❌'} {el:.1f}s | {text[:200]}")
    except Exception as e:
        text = str(e)
        el = time.time() - t0
        ok = False
        print(f"    ❌ {el:.1f}s | Error: {text[:200]}")
    results["audio"] = {"ok": ok, "text": text[:500], "time": round(el, 1)}

    # ── 3. Audio transcription ──
    print(f"\n  [Audio] Transcribe...")
    messages = [{"role": "user", "content": [
        {"type": "audio", "url": AUDIO},
        {"type": "text", "text": "Transcribe the lyrics you hear."},
    ]}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    t0 = time.time()
    try:
        out = generate(model, processor, prompt, audio=[AUDIO],
            max_tokens=200, verbose=False, repetition_penalty=1.2, temperature=0.7)
        text = out.text if hasattr(out, "text") else str(out)
        el = time.time() - t0
        ok = len(text.strip()) > 20
        print(f"    {'✅' if ok else '❌'} {el:.1f}s | {text[:200]}")
    except Exception as e:
        text = str(e)
        el = time.time() - t0
        ok = False
        print(f"    ❌ {el:.1f}s | Error: {text[:200]}")
    results["transcribe"] = {"ok": ok, "text": text[:500], "time": round(el, 1)}

    # ── 4. Text chat EN ──
    print(f"\n  [Text] Chat EN...")
    messages = [{"role": "user", "content": "What is the capital of France? One word answer."}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    t0 = time.time()
    out = generate(model, processor, prompt,
        max_tokens=50, verbose=False, repetition_penalty=1.2, temperature=0.0)
    text = out.text if hasattr(out, "text") else str(out)
    el = time.time() - t0
    ok = "paris" in text.lower()
    print(f"    {'✅' if ok else '❌'} {el:.1f}s | {text[:100]}")
    results["chat_en"] = {"ok": ok, "text": text, "time": round(el, 1)}

    # ── 5. Text chat ZH ──
    print(f"\n  [Text] Chat ZH-TW...")
    messages = [{"role": "user", "content": "法國的首都是哪裡？一個詞回答。"}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    t0 = time.time()
    out = generate(model, processor, prompt,
        max_tokens=50, verbose=False, repetition_penalty=1.2, temperature=0.0)
    text = out.text if hasattr(out, "text") else str(out)
    el = time.time() - t0
    ok = "巴黎" in text
    print(f"    {'✅' if ok else '❌'} {el:.1f}s | {text[:100]}")
    results["chat_zh"] = {"ok": ok, "text": text, "time": round(el, 1)}

    passed = sum(1 for v in results.values() if v["ok"])
    total = len(results)
    print(f"\n  === {name}: {passed}/{total} passed ===")
    return results

if __name__ == "__main__":
    models = sys.argv[1:] or ["gemma4-e4b-mlx-4bit", "gemma4-e4b-mlx-8bit"]
    for path in models:
        print(f"\n{'='*60}")
        print(f"  {path}")
        print(f"{'='*60}")
        model, processor = load(path)
        run(model, processor, path)
        del model, processor
