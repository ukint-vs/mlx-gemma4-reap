# Gemma 4 MLX Fixes

Fixes for running Google Gemma 4 (E4B/E2B) on Apple Silicon via [mlx-vlm](https://github.com/Blaizzy/mlx-vlm).

mlx-vlm added Gemma 4 support in PR #890 (2026-04-02), but has several issues that prevent practical use.

## Issues Found & Fixed

### 1. ScaledLinear quantization crash

**Problem**: `ScaledLinear` inherits `nn.Module` instead of `nn.Linear`, so `nn.quantize()` raises `Unable to quantize model of type ScaledLinear`.

**Fix**: Change inheritance to `nn.Linear` in `language.py`.

```python
# Before (broken)
class ScaledLinear(nn.Module):
    def __init__(self, in_features, out_features, scalar):
        super().__init__()
        self.weight = mx.zeros((out_features, in_features))

# After (fixed)
class ScaledLinear(nn.Linear):
    def __init__(self, in_features, out_features, scalar):
        super().__init__(in_features, out_features, bias=False)
```

### 2. Chat template not applied by generate()

**Problem**: `mlx_vlm.generate()` passes raw prompt text directly to the tokenizer without applying the model's chat template. Gemma 4 requires its specific template (`<bos><|turn>user\n...<turn|>\n<|turn>model\n`), so raw text produces garbage output.

**Workaround**: Manually apply chat template before calling generate:

```python
from mlx_vlm import load, generate

model, processor = load("google/gemma-4-E4B-it")
tokenizer = processor.tokenizer

messages = [{"role": "user", "content": [
    {"type": "image", "url": "photo.jpg"},
    {"type": "text", "text": "Describe this image."},
]}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)
out = generate(model, processor, prompt, ["photo.jpg"],
    max_tokens=200, repetition_penalty=1.2, temperature=0.7)
```

### 3. 4-bit quantization quality collapse

**Problem**: Even with the ScaledLinear fix, 4-bit quantized Gemma 4 produces incoherent output. The PLE (Per-Layer Embeddings) architecture is highly sensitive to quantization — the `scalar` multiplier in ScaledLinear amplifies quantization error.

**Status**: Unresolved. bf16 works correctly (44 tok/s, 16.8GB on M4 Max). 4-bit and 8-bit both produce garbage. Likely needs mixed-precision quantization that keeps PLE-related layers in higher precision.

**Comparison with llama.cpp**: GGUF Q4_K_M works fine (77 tok/s, 6.3GB) because llama.cpp uses a different quantization strategy that handles the scaling internally.

### 4. Third-party quantized models missing multimodal weights

**Problem**: Quantized models from unsloth (`unsloth/gemma-4-E4B-it-UD-MLX-4bit`) are missing 963 parameters — the entire `audio_tower`, `vision_tower`, `embed_audio`, and `embed_vision` weights were stripped during quantization.

**Workaround**: Quantize from the original `google/gemma-4-E4B-it` weights yourself:

```python
from mlx_vlm import convert
convert(
    hf_path="google/gemma-4-E4B-it",
    mlx_path="./gemma4-e4b-mlx-4bit",
    quantize=True, q_bits=4, q_group_size=64,
)
```

This preserves all multimodal weights but still suffers from issue #3.

## Working Configuration (bf16)

```bash
pip install git+https://github.com/Blaizzy/mlx-vlm.git@main
python test_gemma4.py
```

| Metric | Value |
|---|---|
| Model | google/gemma-4-E4B-it (bf16) |
| Peak memory | 16.8 GB |
| Decode speed | 44 tok/s |
| Prefill speed | 630 tok/s |
| Vision | ✅ |
| Audio | ✅ (untested) |
| Chat template | ⚠️ Manual apply required |

## Files

```
mlx_vlm_patches/models/gemma4/
├── language.py          ← ScaledLinear fix (nn.Linear inheritance)
├── gemma4.py            ← Main model (unchanged from PR #890)
├── vision.py            ← Vision encoder (unchanged)
├── audio.py             ← Audio encoder (unchanged)
├── config.py            ← Config (unchanged)
├── processing_gemma4.py ← Processor (unchanged)
└── audio_feature_extractor.py
test_gemma4.py           ← Working test with manual chat template
```

## TODO

- [ ] Fix quantization quality (mixed precision for PLE layers)
- [ ] Auto-apply chat template in generate() for Gemma 4
- [ ] Submit PR to mlx-vlm
- [ ] Test audio input
