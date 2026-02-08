"""
Mocks DB storing info about common accelerators used for LLM serving and inference
"""
import json

gpu_specs = {}

with open("config_explorer/db.json") as f:
    _raw = json.load(f)
    gpu_specs = {k: v for k, v in _raw.items() if not k.startswith("_")}
