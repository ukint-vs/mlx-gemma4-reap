#!/usr/bin/env python3
"""Full validation: all Gemma 4 MLX variants × vision + audio + text."""
import os, sys, json, time, glob, random
from mlx_vlm import load, generate

IDIR = os.path.expanduser("~/Python/imaglius/data/test-images")
SPEECH_DIR = "test_audio/speech"

# 10 images for caption
random.seed(42)
all_imgs = sorted([f for f in os.listdir(IDIR)
    if f.endswith(('.jpg','.jpeg','.png'))
    and 10000 < os.path.getsize(f"{IDIR}/{f}") < 10000000])
random.shuffle(all_imgs)
IMAGES = sorted(all_imgs[:10])

SPEECH = [
    ("obama_30s.wav", "Transcribe this speech.", "this week i traveled to chicago"),
    ("en_greeting.wav", "Transcribe this speech.", "hello my name is john"),
    ("en_numbers.wav", "Transcribe this speech.", "72 degrees"),
    ("zh_greeting.wav", "Transcribe this speech in the original language.", "小明"),
    ("ja_greeting.wav", "Transcribe this speech in the original language.", "田中"),
]

CHAT = [
    ("EN", "What is the capital of France? One word.", "paris"),
    ("ZH", "法國的首都是哪裡？一個詞。", "巴黎"),
    ("JA", "フランスの首都は？一言で。", "パリ"),
]

MODELS = sorted(glob.glob("gemma4-*-mlx-*"))
if sys.argv[1:]:
    MODELS = [m for m in MODELS if any(a in m for a in sys.argv[1:])]

results = {}
for mdir in MODELS:
    if not os.path.isdir(mdir): continue
    cfg = json.load(open(f"{mdir}/config.json"))
    has_audio = cfg.get("audio_config") is not None

    print(f"\n{'='*60}\n  {mdir} (audio={has_audio})\n{'='*60}")
    model, processor = load(mdir)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    r = {"vision": [], "audio": [], "chat": []}

    # Vision: 10 captions
    print("\n  --- Vision (10 images) ---")
    for i, fn in enumerate(IMAGES):
        messages = [{"role": "user", "content": [
            {"type": "image", "url": f"{IDIR}/{fn}"},
            {"type": "text", "text": "Write a single detailed caption for this image."},
        ]}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        t0 = time.time()
        out = generate(model, processor, prompt, [f"{IDIR}/{fn}"],
            max_tokens=150, verbose=False, repetition_penalty=1.2, temperature=0.7)
        text = (out.text if hasattr(out, "text") else str(out)).strip()
        el = time.time() - t0
        ok = len(text) > 20 and "oxff" not in text
        print(f"    {'✅' if ok else '❌'} [{i+1:2d}] {el:.1f}s | {text[:80]}")
        r["vision"].append({"img": fn, "ok": ok, "time": round(el, 1), "text": text[:300]})

    v_pass = sum(1 for x in r["vision"] if x["ok"])
    print(f"  Vision: {v_pass}/10")

    # Audio: 5 speech (only for E2B/E4B)
    if has_audio:
        print("\n  --- Audio (5 speech) ---")
        for fname, prompt_text, keyword in SPEECH:
            wav = f"{SPEECH_DIR}/{fname}"
            if not os.path.exists(wav) or os.path.getsize(wav) < 100:
                print(f"    SKIP {fname}"); continue
            messages = [{"role": "user", "content": [
                {"type": "audio", "url": wav},
                {"type": "text", "text": prompt_text},
            ]}]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            t0 = time.time()
            try:
                out = generate(model, processor, prompt, audio=[wav],
                    max_tokens=150, verbose=False, repetition_penalty=1.2, temperature=0.1)
                text = (out.text if hasattr(out, "text") else str(out)).strip()
            except Exception as e:
                text = f"ERROR: {e}"
            el = time.time() - t0
            ok = keyword.lower() in text.lower()
            print(f"    {'✅' if ok else '❌'} {fname:20s} {el:.1f}s | {text[:100]}")
            r["audio"].append({"file": fname, "ok": ok, "time": round(el, 1), "text": text[:300]})
        a_pass = sum(1 for x in r["audio"] if x["ok"])
        print(f"  Audio: {a_pass}/{len(r['audio'])}")

    # Chat: 3 languages
    print("\n  --- Chat ---")
    for lang, prompt_text, keyword in CHAT:
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        t0 = time.time()
        out = generate(model, processor, prompt,
            max_tokens=30, verbose=False, repetition_penalty=1.2, temperature=0.0)
        text = (out.text if hasattr(out, "text") else str(out)).strip()
        el = time.time() - t0
        ok = keyword in text.lower() or keyword in text
        print(f"    {'✅' if ok else '❌'} {lang:5s} {el:.1f}s | {text[:80]}")
        r["chat"].append({"lang": lang, "ok": ok, "time": round(el, 1), "text": text[:200]})
    c_pass = sum(1 for x in r["chat"] if x["ok"])
    print(f"  Chat: {c_pass}/3")

    total = v_pass + sum(1 for x in r["audio"] if x["ok"]) + c_pass
    max_total = 10 + (5 if has_audio else 0) + 3
    print(f"\n  === {mdir}: {total}/{max_total} passed ===")
    results[mdir] = r
    del model, processor

# Save
with open("validation_all.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n📊 Saved: validation_all.json")
