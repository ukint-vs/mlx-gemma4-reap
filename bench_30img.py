#!/usr/bin/env python3
"""30-image benchmark: Gemma 4 E4B MLX PLE-safe 4bit vs 8bit."""
import os, sys, json, time, random
from mlx_vlm import load, generate

IDIR = os.path.expanduser("~/Python/imaglius/data/test-images")
ODIR = os.path.expanduser("~/Python/imaglius/doc/20260403")

random.seed(42)
all_imgs = sorted([f for f in os.listdir(IDIR)
    if f.endswith(('.jpg','.jpeg','.png'))
    and 10000 < os.path.getsize(f"{IDIR}/{f}") < 10000000])
random.shuffle(all_imgs)
images = sorted(all_imgs[:30])
print(f"Selected {len(images)} images")

MODELS = {
    "Gemma4-E4B-MLX-4bit": "gemma4-e4b-mlx-4bit",
    "Gemma4-E4B-MLX-8bit": "gemma4-e4b-mlx-8bit",
}

CHAT_PROMPTS = [
    ("EN", "What are the main differences between JPEG and PNG image formats? Give a concise answer."),
    ("ZH-TW", "請用繁體中文解釋 JPEG 和 PNG 圖片格式的主要差異。簡潔回答。"),
    ("JA", "日本語で、JPEGとPNGの違いを簡潔に説明してください。"),
]

results = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "hw": "Apple M4 Max 128GB",
           "n_images": len(images)}

for name, path in MODELS.items():
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    model, processor = load(path)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Caption 30 images
    caps = []
    total_tps = 0; count = 0
    for i, fn in enumerate(images):
        img_path = f"{IDIR}/{fn}"
        messages = [{"role": "user", "content": [
            {"type": "image", "url": img_path},
            {"type": "text", "text": "Write a single detailed caption for this image."},
        ]}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        t0 = time.time()
        out = generate(model, processor, prompt, [img_path],
            max_tokens=200, verbose=False, repetition_penalty=1.2, temperature=0.7)
        el = time.time() - t0
        text = out.text if hasattr(out, "text") else str(out)
        tps = len(text.split()) / el if el > 0 else 0  # rough word/s

        # Get actual token stats from GenerationResult
        tok = out.generation_tokens if hasattr(out, "generation_tokens") else 0
        actual_tps = out.tokens_per_second if hasattr(out, "tokens_per_second") else (tok / el if tok and el > 0 else 0)
        if actual_tps > 0: total_tps += actual_tps; count += 1

        print(f"  [{i+1:2d}/30] {el:.1f}s {actual_tps:.0f}t/s | {fn[:35]}")
        print(f"    {text[:120]}")
        caps.append({"img": fn, "time": round(el, 2), "tps": round(actual_tps, 1), "text": text})

    avg_tps = total_tps / count if count > 0 else 0
    print(f"  Caption AVG: {avg_tps:.1f} tok/s ({count} images)")

    # Chat
    chats = []
    for lang, prompt_text in CHAT_PROMPTS:
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        t0 = time.time()
        out = generate(model, processor, prompt,
            max_tokens=200, verbose=False, repetition_penalty=1.2, temperature=0.7)
        el = time.time() - t0
        text = out.text if hasattr(out, "text") else str(out)
        actual_tps = out.tokens_per_second if hasattr(out, "tokens_per_second") else 0
        print(f"  Chat[{lang}] {el:.1f}s {actual_tps:.0f}t/s")
        print(f"    {text[:150]}")
        chats.append({"lang": lang, "time": round(el, 2), "tps": round(actual_tps, 1), "text": text})

    results[name] = {"avg_caption_tps": round(avg_tps, 1), "captions": caps, "chats": chats}
    del model, processor

# Save
os.makedirs(ODIR, exist_ok=True)
out_path = f"{ODIR}/gemma4_mlx_30img.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n📊 Saved: {out_path}")
