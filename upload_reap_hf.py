#!/usr/bin/env python3
"""Generate model cards and upload REAP MLX models to HuggingFace."""
import os, json
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN") or None  # None = use cached login
OWNER = "ukint-vs"
BASE = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "REAP-21B": {
        "hf_source": "0xSero/gemma-4-21b-a4b-it-REAP",
        "params": "21B MoE (20% pruned)",
        "experts": 103,
        "pruning": "20%",
        "size_4bit": "13.9 GB",
        "local_dir": "gemma4-reap-21b-mlx-4bit",
        "repo_name": "gemma-4-21b-a4b-it-REAP-MLX-4bit",
    },
    "REAP-19B": {
        "hf_source": "0xSero/gemma-4-19b-a4b-it-REAP",
        "params": "19B MoE (30% pruned)",
        "experts": 90,
        "pruning": "30%",
        "size_4bit": "12.6 GB",
        "local_dir": "gemma4-reap-19b-mlx-4bit",
        "repo_name": "gemma-4-19b-a4b-it-REAP-MLX-4bit",
    },
}

BENCH_21B = {
    "elementary_mathematics": 84,
    "philosophy": 62,
    "world_religions": 52,
    "college_computer_science": 68,
    "high_school_mathematics": 40,
    "abstract_algebra": 60,
    "college_mathematics": 52,
    "gsm8k": 76,
}

BENCH_19B = {
    "elementary_mathematics": 44,
    "philosophy": 54,
    "world_religions": 34,
    "college_computer_science": 34,
    "high_school_mathematics": 22,
    "abstract_algebra": 36,
    "college_mathematics": 16,
    "gsm8k": 62,
}

BENCH_26B_A4B_4BIT = {
    "elementary_mathematics": 84,
    "philosophy": 66,
    "world_religions": 66,
    "college_computer_science": 58,
    "high_school_mathematics": 26,
    "abstract_algebra": 44,
    "college_mathematics": 36,
    "gsm8k": 64,
}


def bench_table(variant):
    bench = BENCH_21B if variant == "REAP-21B" else BENCH_19B
    rows = []
    for task, our_score in bench.items():
        display = task.replace("_", " ").title()
        orig_4bit = BENCH_26B_A4B_4BIT.get(task, "—")
        orig_str = f"{orig_4bit}%" if isinstance(orig_4bit, int) else orig_4bit
        rows.append(f"| {display} | {orig_str} | **{our_score}%** |")
    return "\n".join(rows)


def gen_readme(variant):
    m = MODELS[variant]
    bench = BENCH_21B if variant == "REAP-21B" else BENCH_19B
    other = "REAP-21B" if variant == "REAP-19B" else "REAP-19B"
    other_m = MODELS[other]

    return f"""---
language:
  - en
  - zh
  - ja
  - ko
  - de
  - fr
  - es
  - pt
  - it
  - ar
  - hi
license: gemma
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
library_name: mlx
pipeline_tag: image-text-to-text
base_model: {m["hf_source"]}
tags:
- mlx
- gemma4
- reap
- ple-safe
- quantized
- apple-silicon
- vision
- moe
---

# {m["repo_name"]}

**PLE-safe** MLX 4-bit weights for [{m["hf_source"]}](https://huggingface.co/{m["hf_source"]}) on Apple Silicon.

[REAP](https://arxiv.org/abs/2510.13999) (Router-weighted Expert Activation Pruning) removes {m["pruning"]} of MoE experts while keeping the same active parameters per token (8 of {m["experts"]} experts selected). Combined with PLE-safe 4-bit quantization, this model runs in **{m["size_4bit"]}** — fits on 16GB+ Macs.

| | Original 26B | REAP ({m["params"]}) | **This model** |
|---|---|---|---|
| Experts/layer | 128 | {m["experts"]} | {m["experts"]} |
| Precision | BF16 | BF16 | **4-bit** |
| Disk size | ~52 GB | ~{43 if variant == "REAP-21B" else 36} GB | **{m["size_4bit"]}** |

## Also available

- [{other_m["repo_name"]}](https://huggingface.co/{OWNER}/{other_m["repo_name"]}) — {other_m["params"]}, {other_m["size_4bit"]}

## Accuracy Benchmarks

0-shot generative, thinking enabled, 50 samples per task, identical eval harness. Apple M4 Max 36GB.

| Task | 26B-A4B 4-bit (16.4 GB) | **This model ({m["size_4bit"]})** |
|---|---|---|
{bench_table(variant)}

{"**Wins 5 of 8 tasks vs the original 26B-A4B 4-bit** while being 2.5 GB smaller. REAP pruning removes low-utility experts, improving reasoning quality." if variant == "REAP-21B" else "The 30% expert pruning compounds with 4-bit quantization. Note: high extraction failure rates (up to 60%) on some tasks — the model generates verbose explanations instead of single-letter answers, so true accuracy may be higher than reported. Consider the [21B variant](https://huggingface.co/" + OWNER + "/" + MODELS["REAP-21B"]["repo_name"] + ") for better accuracy."}

Extraction failures (unparseable responses) are counted as incorrect. {"REAP-21B: 38/350 (11%)." if variant == "REAP-21B" else "REAP-19B: 113/400 (28%)."} True accuracy may be higher. Full methodology: [GitHub](https://github.com/ukint-vs/mlx-gemma4-reap).

## Quantization Details

- **Bits:** 4
- **Group size:** 64
- **Strategy:** PLE-safe — only large `nn.Linear` and `SwitchLinear` (MoE expert) layers are quantized. All PLE/ScaledLinear/vision layers stay in bf16.

| Quantized (4-bit) | Kept in bf16 |
|---|---|
| Attention projections (q/k/v/o_proj) | ScaledEmbedding (embed_tokens) |
| MLP layers (gate/up/down_proj) | ScaledLinear (PLE pathway) |
| MoE expert layers (SwitchLinear) | Per-layer embeddings (per_layer_*) |
| | Vision encoder |
| | All norms and scalars |

## Usage

```bash
pip install -U mlx-vlm
```

### Vision

```python
from mlx_vlm import load, generate

model, processor = load("{OWNER}/{m["repo_name"]}")
tokenizer = processor.tokenizer

messages = [{{"role": "user", "content": [
    {{"type": "image", "url": "photo.jpg"}},
    {{"type": "text", "text": "Describe this image in detail."}},
]}}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
out = generate(model, processor, prompt, ["photo.jpg"],
    max_tokens=200, repetition_penalty=1.2, temperature=0.7)
print(out.text)
```

### Text

```python
messages = [{{"role": "user", "content": "What is the capital of France?"}}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
out = generate(model, processor, prompt, max_tokens=100, temperature=0.0)
print(out.text)
```

## Validation

Trimodal validation: 10/10 vision, 3/3 chat (EN/ZH/JA). Full results: [GitHub](https://github.com/ukint-vs/mlx-gemma4-reap).

## Bugs Fixed in mlx-vlm

| # | Bug | Fix |
|---|---|---|
| 1 | `ScaledLinear` inherits `nn.Module` not `nn.Linear` | Change to `ScaledLinear(nn.Linear)` |
| 2 | Standard quantization quantizes PLE layers | PLE-safe `class_predicate` |
| 3 | `processor.save_pretrained()` strips audio config | Copy `processor_config.json` from source |
| 4 | `SwitchLinear` (MoE) not quantized | Check `hasattr(module, 'to_quantized')` |
| 5 | `embed_scale` double-scaling (mlx-vlm 0.4.4+) | Set `Gemma4TextModel.embed_scale = 1.0` |

## License

Model weights: [Google Gemma License](https://ai.google.dev/gemma/terms). Quantization scripts: MIT.
"""


if __name__ == "__main__":
    api = HfApi(token=TOKEN)

    for variant, m in MODELS.items():
        repo_name = m["repo_name"]
        full_repo = f"{OWNER}/{repo_name}"
        local_dir = os.path.join(BASE, m["local_dir"])

        if not os.path.isdir(local_dir):
            print(f"SKIP {local_dir} (not found)")
            continue

        # Generate README
        readme = gen_readme(variant)
        readme_path = os.path.join(local_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme)
        print(f"Generated README for {repo_name}")

        # Create repo
        api.create_repo(full_repo, repo_type="model", exist_ok=True)
        print(f"Repo {full_repo} ready")

        # Upload
        print(f"Uploading {local_dir} ...")
        api.upload_folder(
            folder_path=local_dir,
            repo_id=full_repo,
            commit_message=f"Upload {repo_name} — PLE-safe MLX 4-bit REAP weights",
            ignore_patterns=[".*"],
        )
        print(f"Uploaded {full_repo}\n")
