#!/usr/bin/env python3
"""Test Gemma 4 E4B on MLX with proper chat template."""
from mlx_vlm import load, generate

model, processor = load("google/gemma-4-E4B-it")
tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

# --- Vision caption ---
messages = [{"role": "user", "content": [
    {"type": "image", "url": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/bird.png"},
    {"type": "text", "text": "Write a single detailed caption for this image."},
]}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
out = generate(model, processor, prompt,
    ["https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/bird.png"],
    max_tokens=150, verbose=True, repetition_penalty=1.2, temperature=0.7)
print(out.text if hasattr(out, "text") else out)

# --- Text chat ---
messages = [{"role": "user", "content": "Explain JPEG vs PNG briefly."}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
out = generate(model, processor, prompt,
    max_tokens=200, verbose=True, repetition_penalty=1.2, temperature=0.7)
print(out.text if hasattr(out, "text") else out)
