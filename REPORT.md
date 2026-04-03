# Gemma 4 MLX PLE-Safe Quantization — Full Validation Report

> Date: 2026-04-03
> Hardware: Apple M4 Max 128GB
> Framework: mlx-vlm (+ 4 bug fixes)
> Repo: https://git.lcn.tw:33333/felix/mlx_gemma4.git

## 1. Why This Exists

mlx-community and unsloth's existing Gemma 4 MLX quantized models are **all broken**:

| Source | Problem | Output |
|---|---|---|
| mlx-community 4bit/8bit | PLE layers quantized, quality destroyed | `ionoxffionoxff...` garbage |
| unsloth 4bit | Vision/audio weights stripped (963 missing params) | Load crash |

Root cause: Gemma 4's **PLE (Per-Layer Embeddings)** architecture uses `ScaledLinear` with scalar multipliers that amplify quantization error. Standard quantization destroys the model.

## 2. PLE-Safe Quantization Strategy

Only quantize large `nn.Linear` and `SwitchLinear` (MoE experts) in decoder layers. Keep everything else in bf16:

| Layer Category | Count | Params | Precision |
|---|---|---|---|
| language/MLP (gate/up/down_proj) | 126-410 | 49-94% | **4-bit or 8-bit** |
| language/attention (q/k/v/o_proj) | 168-410 | | **4-bit or 8-bit** |
| MoE experts (SwitchLinear) | 0-90 | | **4-bit or 8-bit** |
| ScaledEmbedding (embed_tokens) | 1-2 | 1-51% | bf16 |
| ScaledLinear (PLE) | 1 | | bf16 |
| PLE (per_layer_*) | 127+ | | bf16 |
| vision_tower | 209-658 | | bf16 |
| audio_tower (E2B/E4B only) | 270-751 | | bf16 |
| norms, scalars | 253+ | | bf16 |

## 3. All 12 Variants — Validation Results

| Model | Precision | Disk | Vision (10) | Audio (5) | Chat (3) | Total |
|---|---|---|---|---|---|---|
| 26B-A4B | 4bit | 16.4G | 10/10 | N/A | 3/3 | **13/13** ✅ |
| 26B-A4B | 8bit | 28.6G | 10/10 | N/A | 3/3 | **13/13** ✅ |
| 26B-A4B | bf16 | 51.6G | 10/10 | N/A | 3/3 | **13/13** ✅ |
| 31B | 4bit | 20.4G | 10/10 | N/A | 3/3 | **13/13** ✅ |
| 31B | 8bit | 35.1G | 10/10 | N/A | 3/3 | **13/13** ✅ |
| 31B | bf16 | 62.5G | 10/10 | N/A | 3/3 | **13/13** ✅ |
| E2B | 4bit | 7.6G | 10/10 | 4/5 | 3/3 | **17/18** ✅ |
| E2B | 8bit | 8.5G | 10/10 | 4/5 | 3/3 | **17/18** ✅ |
| E2B | bf16 | 10.2G | 10/10 | 4/5 | 3/3 | **17/18** ✅ |
| E4B | 4bit | 10.3G | 10/10 | 4/5 | 3/3 | **17/18** ✅ |
| E4B | 8bit | 12.3G | 10/10 | 4/5 | 3/3 | **17/18** ✅ |
| E4B | bf16 | 16.0G | 10/10 | 4/5 | 3/3 | **17/18** ✅ |

> ⚠️ E2B/E4B show 17/18 due to keyword matching strictness in test harness.
> The model correctly outputs "Hello, my name is John." but the test expects lowercase match.
> bf16 has the same 17/18 score — **not a quantization issue**.

## 4. Audio Validation Detail

### Speech Transcription (E2B/E4B only)

| Audio | Language | Ground Truth | E4B 4bit | E4B bf16 |
|---|---|---|---|---|
| obama_30s.wav | EN | This week I traveled to Chicago... | This week I traveled to Chicago to deliver my final farewell | This week I traveled to Chicago to deliver my final farewell |
| en_greeting.wav | EN | Hello, my name is John... | Hello, my name is John. I work as a software engineer in San | Hello, my name is John. I work as a software engineer in San |
| en_numbers.wav | EN | 72 degrees Fahrenheit, 15234 points | The temperature today is 72 degrees Fahrenheit, the stock ma | The temperature today is 72 degrees Fahrenheit, the stock ma |
| zh_greeting.wav | ZH | 你好，我叫小明... | 你好，我叫小明。我在台北的一家科技公司當軟體工程師,今天天氣非常好，我打算下班後去公園散步。最近我在學習機器學習,覺得非 | 你好，我叫小明。我在台北的一家科技公司當軟體工程師。今天天氣非常好，我打算下班後去公園散步。最近我在學習機器學習,覺得非 |
| ja_greeting.wav | JA | こんにちは、田中です... | こんにちは。私の名前は田中です。東京でエンジニアとして働いています。 | こんにちは。私の名前は田中です。東京でエンジニアとして働いています。 |

### Music Recognition

Tested 10 songs (Rick Astley, Queen, Adele, etc.) — **0/10 correct across all variants including bf16**.
This is a model limitation, not quantization: Google states "Music and non-speech sounds were not part of the training data."
The audio encoder is designed for **speech only**.

## 5. Vision Caption Samples

### gemma4-e4b-mlx-4bit

**0910E859-8380-4F5C-BF0C-12E05A** (1.2s):
> A cozy, high-angle view capturing a domestic scene where a fluffy cat rests on an elevated shelf next to a pet food dispenser, set against the backdrop of modern interior design featuring white walls 

**107F1F60-8472-4BBA-A7A4-A78AF8** (0.9s):
> A regal, silver-grey and white tabby cat with striking green eyes gazes upwards while perched elegantly on a wooden surface, sporting a brown collar adorned with a small mint-colored bell, perfectly c

**236CD9B3-0D9D-4E3F-998A-C00A06** (0.9s):
> A vibrant and fresh salad, brimming with crisp green lettuce leaves mixed with finely chopped ingredients like dark leafy greens, chunks of chicken or tuna, and possibly cucumber or other fresh vegeta

### gemma4-31b-mlx-4bit

**0910E859-8380-4F5C-BF0C-12E05A** (8.8s):
> A high-angle, wide shot captured by a Tapo security camera shows a white and brown tabby cat perched on a light-colored wooden shelf in the upper right corner of a room. Below the shelf, a grey plasti

**107F1F60-8472-4BBA-A7A4-A78AF8** (8.7s):
> A high-angle, medium shot captures a curious tabby and white cat sitting upright on a brown surface against a plain white background. The cat has short, dense fur with grey and black tabby stripes on 

**236CD9B3-0D9D-4E3F-998A-C00A06** (4.4s):
> A high-angle, close-up shot shows a wooden bowl filled with a fresh salad consisting of large pieces of light green lettuce on top, and a mixture of dark leafy greens, sliced cucumbers, pineapple chun

## 6. Bugs Found & Fixed in mlx-vlm

| # | Bug | Impact | Fix |
|---|---|---|---|
| 1 | `ScaledLinear(nn.Module)` | `nn.quantize()` crash | Change to `ScaledLinear(nn.Linear)` |
| 2 | Standard quantize destroys PLE | 4bit/8bit output garbage | PLE-safe `class_predicate` |
| 3 | `processor.save_pretrained()` strips audio config | Audio silently dropped | Copy `processor_config.json` from source |
| 4 | `SwitchLinear` not quantized | MoE 26B: 49G instead of 16G | Add `has_quant = hasattr(m, 'to_quantized')` check |

## 7. Reproduce

```bash
git clone https://git.lcn.tw:33333/felix/mlx_gemma4.git
cd mlx_gemma4
pip install git+https://github.com/Blaizzy/mlx-vlm.git@main

# Apply ScaledLinear fix
cp mlx_vlm_patches/models/gemma4/language.py \
   $(python -c "import mlx_vlm; print(mlx_vlm.__path__[0])")/models/gemma4/

# Convert all
python convert_gemma4.py all

# Validate all
python validate_all.py
```