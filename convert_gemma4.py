#!/usr/bin/env python3
"""Convert Gemma 4 (E2B/E4B) to MLX with PLE-safe quantization.

Fixes two mlx-vlm bugs:
1. ScaledLinear quantization (nn.Module -> nn.Linear)
2. processor_config.json missing audio feature_extractor after convert

Usage:
  python convert_gemma4.py E4B 4    # Gemma 4 E4B 4-bit
  python convert_gemma4.py E4B 8    # Gemma 4 E4B 8-bit
  python convert_gemma4.py E2B 4    # Gemma 4 E2B 4-bit
  python convert_gemma4.py E2B 8    # Gemma 4 E2B 8-bit
"""
import sys, os, shutil, json
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
        if not isinstance(module, nn.Linear): return False
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
    "E4B": "google/gemma-4-E4B-it",
    "E2B": "google/gemma-4-E2B-it",
}

if __name__ == "__main__":
    variant = sys.argv[1].upper() if len(sys.argv) > 1 else "E4B"
    bits = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    hf_path = MODELS[variant]
    dst = f"gemma4-{variant.lower()}-mlx-{bits}bit"

    print(f"Converting {hf_path} -> {dst} ({bits}bit, PLE-safe)")
    convert(hf_path=hf_path, mlx_path=dst, quantize=True, q_bits=bits, q_group_size=64)

    # Fix: copy complete processor_config.json from source (mlx-vlm strips audio config)
    src_path = get_model_path(hf_path)
    src_proc = os.path.join(src_path, "processor_config.json")
    if os.path.exists(src_proc):
        shutil.copy2(src_proc, os.path.join(dst, "processor_config.json"))
        print(f"[Fix] Copied complete processor_config.json (with audio feature_extractor)")

    print(f"Done: {dst}")
