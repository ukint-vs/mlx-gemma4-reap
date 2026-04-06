# REAP Gemma 4 MLX — Pruned MoE on Apple Silicon

> First PLE-safe MLX quantization of [REAP-pruned](https://arxiv.org/abs/2510.13999) Gemma 4 MoE models
> **REAP-21B 4-bit (13.9 GB) beats the original 26B 4-bit (16.4 GB) on 5 of 8 benchmarks — smaller model, better scores**

## Models

| Model | Size | HuggingFace |
|---|---|---|
| **REAP-21B** (20% pruned, 103 experts) | 13.9 GB | [ukint-vs/gemma-4-21b-a4b-it-REAP-MLX-4bit](https://huggingface.co/ukint-vs/gemma-4-21b-a4b-it-REAP-MLX-4bit) |
| **REAP-19B** (30% pruned, 90 experts) | 12.6 GB | [ukint-vs/gemma-4-19b-a4b-it-REAP-MLX-4bit](https://huggingface.co/ukint-vs/gemma-4-19b-a4b-it-REAP-MLX-4bit) |

Based on [0xSero's REAP models](https://huggingface.co/0xSero/gemma-4-21b-a4b-it-REAP) and [FakeRocket543's PLE-safe quantization](https://github.com/FakeRocket543/mlx-gemma4).

## Quick Start

```bash
pip install -U mlx-vlm
```

```python
from mlx_vlm import load, generate

model, processor = load("ukint-vs/gemma-4-21b-a4b-it-REAP-MLX-4bit")
tokenizer = processor.tokenizer

# Vision
messages = [{"role": "user", "content": [
    {"type": "image", "url": "photo.jpg"},
    {"type": "text", "text": "Describe this image."},
]}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
out = generate(model, processor, prompt, ["photo.jpg"], max_tokens=200, temperature=0.7)
print(out.text)

# Text
messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
out = generate(model, processor, prompt, max_tokens=100, temperature=0.0)
print(out.text)
```

No patching required — all fixes have been merged upstream into mlx-vlm 0.4.4+.

## Why REAP + PLE-safe Quantization?

**REAP** removes the lowest-scoring MoE experts while keeping the same active parameters per token (8 experts selected). **PLE-safe quantization** only quantizes the large decoder layers, keeping sensitive PLE/vision/audio layers in bf16.

| | Original 26B-A4B | REAP-21B | REAP-19B |
|---|---|---|---|
| Experts/layer | 128 | 103 | 90 |
| Total params | ~26B | ~21B | ~19B |
| BF16 disk | ~52 GB | ~43 GB | ~36 GB |
| **MLX 4-bit** | **16.4 GB** | **13.9 GB** | **12.6 GB** |
| Fits on | 24GB+ Mac | **16GB+ Mac** | **16GB+ Mac** |

## Accuracy Benchmarks

0-shot generative, thinking enabled, 50 samples/task, Apple M4 Max 36GB. Same eval harness across all three models.

| Task | 26B-A4B 4-bit (16.4 GB) | **REAP-21B 4-bit (13.9 GB)** | REAP-19B 4-bit (12.6 GB) |
|---|---|---|---|
| Elementary Math | 84% | **84%** | 44% |
| Philosophy | **66%** | 62% | 54% |
| World Religions | **66%** | 52% | 34% |
| College CS | 58% | **68%** | 34% |
| HS Math | 26% | **40%** | 22% |
| Abstract Algebra | 44% | **60%** | 36% |
| College Math | 36% | **52%** | 16% |
| GSM8K | 64% | **76%** | 62% |

**REAP-21B outscores the original 26B on 5 of 8 tasks** (1 tie, 2 losses) despite being 2.5 GB smaller. Biggest gains: College Math +16, Abstract Algebra +16, HS Math +14, GSM8K +12. The two losses are knowledge-heavy tasks where removing experts hurts: World Religions -14, Philosophy -4.

REAP-19B takes a bigger hit — 30% pruning + 4-bit compounds. Not recommended unless memory is the primary constraint.

> **Methodology note:** Generative evaluation with regex extraction. Extraction failures (unparseable responses) counted as incorrect. REAP-21B: 38/350 (11%), REAP-19B: 113/400 (28%), 26B-A4B: 85/400 (21%). True accuracy may be higher. BF16 reference numbers from [0xSero](https://huggingface.co/0xSero/gemma-4-21b-a4b-it-REAP) use log-likelihood scoring (lm-eval + vLLM) — not directly comparable.

## Decode Speed

Apple M4 Max 36GB, temperature 0.0, best of 2 runs:

| Model | Size | Avg tok/s |
|---|---|---|
| 26B-A4B 4-bit | 16.4 GB | 64.4 |
| REAP-21B 4-bit | 13.9 GB | 59.8 |
| REAP-19B 4-bit | 12.6 GB | 61.8 |

Speed is identical (~60-64 tok/s) — only 8 experts active per token regardless of pool size.

## Validation

All models pass trimodal validation: 10/10 vision captions, 3/3 multilingual chat (EN/ZH/JA).

### Vision Samples (REAP-21B 4-bit)

> **Coastline:** "A vast, dramatic expanse of deep blue sea meets a sprawling, sun-drenched coastline..."

> **Winter forest:** "A vast forest of evergreen trees stands draped in heavy, crystalline layers of fresh snow, their branches weighted down by white blankets that catch the soft, golden light of a rising sun..."

> **Woman portrait:** "A lone woman stands in silhouette against a massive, monochromatic yellow wall composed of vertical stripes, creating a striking visual contrast between her dark figure and the vibrant, textured background."

## Convert From Source

```bash
git clone https://github.com/ukint-vs/mlx-gemma4-reap.git
cd mlx-gemma4-reap
pip install -U mlx-vlm

python convert_gemma4.py REAP-21B 4   # 13.9 GB
python convert_gemma4.py REAP-19B 4   # 12.6 GB
python validate_all.py                # trimodal validation
python bench_reap.py --model ./gemma4-reap-21b-mlx-4bit --limit 50  # accuracy bench
```

## How It Works

### PLE-Safe Quantization

Standard quantization breaks Gemma 4 because it quantizes PLE (Per-Layer Embedding) layers with `ScaledLinear` — the scalar multiplication amplifies quantization error catastrophically.

| Quantized (4-bit) | Kept in bf16 |
|---|---|
| Attention projections (q/k/v/o_proj) | ScaledEmbedding (embed_tokens) |
| MLP layers (gate/up/down_proj) | ScaledLinear (PLE pathway) |
| MoE expert layers (SwitchLinear) | Per-layer embeddings |
| | Vision encoder, norms, scalars |

### Bugs Found and Fixed in mlx-vlm

All 5 bugs discovered during this work have been merged upstream into mlx-vlm 0.4.4:

| # | Bug | Impact |
|---|---|---|
| 1 | `ScaledLinear` inherits `nn.Module` not `nn.Linear` | `nn.quantize()` can't find these layers |
| 2 | Standard quantization quantizes PLE layers | Garbage output |
| 3 | `processor.save_pretrained()` strips audio config | Audio silently dropped |
| 4 | `SwitchLinear` (MoE) not included in quantization | 26B-A4B: 49 GB instead of 16 GB |
| 5 | `embed_scale` double-scaling | Vision sees images as text, chat is garbage |

## Credits

- [0xSero](https://huggingface.co/0xSero) — REAP-pruned Gemma 4 models
- [FakeRocket543](https://github.com/FakeRocket543/mlx-gemma4) — PLE-safe quantization strategy and original mlx-vlm bug fixes
- [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — MLX vision-language framework
- [REAP paper](https://arxiv.org/abs/2510.13999) — "REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression"

## License

Model weights: [Google Gemma License](https://ai.google.dev/gemma/terms). Scripts: MIT.
