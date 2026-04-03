#!/usr/bin/env python3
"""Test Gemma 4 E4B on MLX — bf16 or quantized."""
import sys
from mlx_vlm import load, generate

model_path = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-E4B-it"
model, processor = load(model_path)
tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

# Caption
messages = [{"role": "user", "content": [
    {"type": "image", "url": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/bird.png"},
    {"type": "text", "text": "Write a single detailed caption for this image."},
]}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
out = generate(model, processor, prompt,
    ["https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/bird.png"],
    max_tokens=150, verbose=True, repetition_penalty=1.2, temperature=0.7)
print(f"\nCaption: {out.text if hasattr(out, 'text') else out}")

# Chat ZH
messages = [{"role": "user", "content": "請用繁體中文解釋 JPEG 和 PNG 的差異。簡潔回答。"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
out = generate(model, processor, prompt,
    max_tokens=200, verbose=True, repetition_penalty=1.2, temperature=0.7)
print(f"\nChat: {out.text if hasattr(out, 'text') else out}")
