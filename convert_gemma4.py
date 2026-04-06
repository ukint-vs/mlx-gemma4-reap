#!/usr/bin/env python3
"""Convert all Gemma 4 variants to MLX with PLE-safe quantization.

Usage:
  python convert_gemma4.py E4B 4        # single
  python convert_gemma4.py E4B bf16     # no quantization
  python convert_gemma4.py all          # all variants × all precisions
"""
import sys, os, shutil
import mlx.nn as nn
from mlx_vlm.models.gemma4.language import ScaledLinear, ScaledEmbedding

SKIP = ("vision_tower", "audio_tower", "embed_vision", "embed_audio",
        "embed_tokens", "per_layer", "norm", "layer_scalar",
        "embedding_post_projection")

_orig_quantize = nn.quantize

def ple_safe_quantize(model, group_size=64, bits=4, class_predicate=None, **kwargs):
    def predicate(path, module):
        if class_predicate and not class_predicate(path, module):
            return False
        # Allow nn.Linear and SwitchLinear (MoE experts)
        is_linear = isinstance(module, nn.Linear)
        has_quant = hasattr(module, "to_quantized")
        if not (is_linear or has_quant): return False
        if isinstance(module, (ScaledLinear, ScaledEmbedding, nn.QuantizedLinear)): return False
        for s in SKIP:
            if s in path: return False
        if hasattr(module, "weight") and module.weight.size % 64 != 0: return False
        return True
    count = sum(1 for p, m in model.named_modules() if predicate(p, m))
    print(f"[PLE-safe] {count} layers -> {bits}bit (skip PLE/vision/audio)")
    _orig_quantize(model, group_size=group_size, bits=bits, class_predicate=predicate, **kwargs)

nn.quantize = ple_safe_quantize

from mlx_vlm import convert
from mlx_vlm.utils import get_model_path

MODELS = {
    "E2B":       "google/gemma-4-E2B-it",
    "E4B":       "google/gemma-4-E4B-it",
    "26B-A4B":   "google/gemma-4-26B-A4B-it",
    "REAP-21B":  "0xSero/gemma-4-21b-a4b-it-REAP",
    "REAP-19B":  "0xSero/gemma-4-19b-a4b-it-REAP",
    "31B":       "google/gemma-4-31B-it",
}

def do_convert(variant, bits):
    hf_path = MODELS[variant]
    tag = f"bf16" if bits == 16 else f"{bits}bit"
    dst = f"gemma4-{variant.lower()}-mlx-{tag}"

    if os.path.exists(dst) and os.path.exists(f"{dst}/config.json"):
        print(f"SKIP {dst} (already exists)")
        return dst

    print(f"\n{'='*60}")
    print(f"  Converting {hf_path} -> {dst}")
    print(f"{'='*60}")

    quantize = bits < 16
    convert(hf_path=hf_path, mlx_path=dst, quantize=quantize,
            q_bits=bits if quantize else 4, q_group_size=64)

    # Fix: copy complete processor_config.json (mlx-vlm strips audio config)
    src_path = get_model_path(hf_path)
    src_proc = os.path.join(src_path, "processor_config.json")
    if os.path.exists(src_proc):
        shutil.copy2(src_proc, os.path.join(dst, "processor_config.json"))
        print(f"[Fix] Copied complete processor_config.json")

    sz = sum(os.path.getsize(os.path.join(dst, f))
             for f in os.listdir(dst) if not f.startswith("."))
    print(f"Done: {dst} ({sz/1e9:.1f} GB)")
    return dst

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_gemma4.py <variant> <bits|bf16|all>")
        print("  variant: E2B, E4B, 26B-A4B, REAP-21B, REAP-19B, 31B, all")
        print("  bits: 4, 8, bf16, all")
        sys.exit(1)

    variant = sys.argv[1].upper()
    precision = sys.argv[2] if len(sys.argv) > 2 else "all"

    if variant == "ALL":
        variants = list(MODELS.keys())
    else:
        variants = [variant]

    if precision == "all":
        precisions = [4, 8, 16]
    elif precision == "bf16":
        precisions = [16]
    else:
        precisions = [int(precision)]

    for v in variants:
        for b in precisions:
            do_convert(v, b)
