#!/usr/bin/env python3
"""
Fetch the top trending text-generation models from HuggingFace and build
an offline cache JSON for the Multi-Model GPU Planner.

Usage:
    HF_TOKEN=hf_xxx python config_explorer/scripts/build_model_cache.py

Output: config_explorer/model_cache.json
"""

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from huggingface_hub import HfApi
from src.config_explorer.capacity_planner import (
    get_model_config_from_hf,
    get_text_config,
    model_memory_req,
    model_total_params,
    is_moe as check_is_moe,
    is_quantized,
    get_quant_method,
    precision_to_byte,
)

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "model_cache.json")
TARGET_COUNT = 1000


def detect_quant_label(model_config) -> str:
    if not is_quantized(model_config):
        return "BF16 (no quantization)"
    method = get_quant_method(model_config)
    try:
        bpp = precision_to_byte(method)
    except (ValueError, TypeError):
        return "FP8"
    if bpp <= 0.5:
        return "INT4"
    if bpp <= 1.0:
        return "INT8" if "int" in method.lower() else "FP8"
    return "BF16 (no quantization)"


def main():
    api = HfApi(token=HF_TOKEN)

    print(f"Fetching top {TARGET_COUNT} text-generation models from HuggingFace...")
    models_iter = api.list_models(
        task="text-generation",
        sort="downloads",
        direction=-1,
        limit=TARGET_COUNT,
    )
    model_ids = [m.id for m in models_iter]
    print(f"Found {len(model_ids)} model IDs")

    cache = {}
    errors = []

    for i, model_id in enumerate(model_ids):
        print(f"[{i+1}/{len(model_ids)}] {model_id}...", end=" ", flush=True)
        try:
            model_info = api.model_info(model_id)

            # Skip models without safetensors metadata (can't compute weight memory)
            if not model_info.safetensors:
                print("SKIP (no safetensors)")
                continue

            model_config = get_model_config_from_hf(model_id, HF_TOKEN)
            text_config = get_text_config(model_config)

            total_params = model_total_params(model_info)
            weight_memory_gib = model_memory_req(model_info, model_config)
            moe = check_is_moe(text_config)
            quant = detect_quant_label(model_config)

            cache[model_id] = {
                "params_billion": round(total_params / 1e9, 2),
                "weight_memory_gib": round(weight_memory_gib, 2),
                "is_moe": moe,
                "quantization": quant,
            }
            print(f"OK ({cache[model_id]['params_billion']}B, {quant})")

        except Exception as e:
            short_err = str(e).split("\n")[0][:120]
            print(f"ERROR: {short_err}")
            errors.append({"model": model_id, "error": short_err})

    # Write cache
    output = {
        "_meta": {
            "description": "Offline model cache for Multi-Model GPU Planner",
            "model_count": len(cache),
            "source": "HuggingFace Hub API, sorted by downloads, text-generation task",
        },
        "models": cache,
    }

    with open(CACHE_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. Cached {len(cache)} models to {CACHE_PATH}")
    if errors:
        print(f"Errors: {len(errors)} models failed")
        for e in errors[:10]:
            print(f"  {e['model']}: {e['error']}")


if __name__ == "__main__":
    main()
