#!/usr/bin/env python3
"""Convert Gemma 4 E4B to MLX with PLE-safe quantization."""
import mlx.nn as nn
from mlx_vlm.models.gemma4.language import ScaledLinear, ScaledEmbedding

SKIP = ("vision_tower", "audio_tower", "embed_vision", "embed_audio",
        "embed_tokens", "per_layer", "norm", "layer_scalar",
        "embedding_post_projection")

_orig_quantize = nn.quantize

def ple_safe_quantize(model, group_size=64, bits=4, class_predicate=None, **kwargs):
    def predicate(path, module):
        # If upstream predicate says no, respect it
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
import sys

bits = int(sys.argv[1]) if len(sys.argv) > 1 else 4
dst = f"gemma4-e4b-mlx-{bits}bit"
convert(hf_path="google/gemma-4-E4B-it", mlx_path=dst, quantize=True, q_bits=bits, q_group_size=64)
print(f"Done: {dst}")
