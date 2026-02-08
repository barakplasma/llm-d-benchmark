# Config Explorer — CLAUDE.md

## What this is

Capacity planning and GPU recommendation tool for LLM inference on llm-d. Provides a CLI (`config-explorer`), a Streamlit web UI, and a Python library.

## Quick reference

```bash
# Install (from repo root)
pip install -e ./config_explorer

# Install Streamlit extras
pip install -r config_explorer/requirements-streamlit.txt

# Run tests (from config_explorer/)
cd config_explorer
pytest tests/ -v

# Launch web UI
config-explorer start

# CLI examples
config-explorer plan --model Qwen/Qwen3-32B --gpu-memory 80 --max-model-len 16000
config-explorer estimate --model Qwen/Qwen3-32B --input-len 512 --output-len 128 --pretty
```

## Project layout

```
config_explorer/
├── src/config_explorer/          # Installable package (src-layout)
│   ├── cli.py                    # CLI entrypoint (argparse): plan, estimate, start
│   ├── capacity_planner.py       # Core memory/KV-cache math, HF model fetching
│   ├── recommender/
│   │   ├── recommender.py        # GPURecommender — wraps BentoML llm-optimizer
│   │   └── cost_manager.py       # GPU cost lookup (gpu_costs.json)
│   ├── explorer.py               # Benchmark report analysis, Pareto fronts
│   ├── constants.py              # Bound prefixes, column mappings
│   └── plotting.py               # Matplotlib helpers
├── pages/                        # Streamlit pages (numbered for sidebar order)
│   ├── 2_GPU_Recommender.py
│   ├── 3_Sweep_Visualizer.py
│   └── 4_Multi_Model_GPU_Planner.py
├── Capacity_Planner.py           # Streamlit main page
├── util.py                       # Streamlit session-state helpers, Scenario dataclass
├── db.json                       # GPU specs database (memory, TDP, generation)
├── gpu_costs.json                # Reference GPU pricing (for relative comparison only)
├── tests/                        # pytest tests
├── pytest.ini                    # pythonpath = src .
└── pyproject.toml                # Python >=3.11, setuptools, entry_points
```

## Architecture notes

- **src-layout**: the installable package lives in `src/config_explorer/`. The `pytest.ini` adds both `src` and `.` to `pythonpath`.
- **Streamlit pages** are standalone scripts under `pages/`. They import from `src.config_explorer.*` (relative to the `config_explorer/` working directory). They are **not** part of the installed package.
- **GPU database** (`db.json`): keys starting with `_` are metadata and are filtered out on load. Each GPU entry has `memory` (GiB), `tdp_watts`, `generation`, `mem_type`.
- **Memory model**: activation memory constants are empirically validated (see `empirical-vllm-memory-results.md`). Key constants: `ACTIVATION_MEMORY_BASE_DENSE_GIB = 5.5`, `ACTIVATION_MEMORY_BASE_MOE_GIB = 8.0`.

## Key conventions

- **Python 3.11+** — uses `X | None` union syntax, `StrEnum`, etc.
- **Dataclasses** for data models (`ModelEntry`, `KVCacheDetail`, `Scenario`, `InventoryItem`).
- **Type hints** on all function signatures.
- **Streamlit session state** pattern: initialize in an `_init()` function, use `st.session_state[KEY]` throughout, callbacks via `on_change`.
- **No linter/formatter enforced** — no ruff, black, or pre-commit config.

## GPU allocation model (Multi-Model Planner)

The `allocate_servers()` function in `4_Multi_Model_GPU_Planner.py` uses first-fit bin packing:

- **TP GPUs** must be co-located on one server (NVLink requirement).
- **PP stages** can span servers (network fabric).
- **DP replicas** are fully independent.
- If `TP > gpus_per_server`, the config is impossible — an error is returned.
- `total_servers` is derived from the actual packing result, not naive division.

## Testing

```bash
cd config_explorer && pytest tests/ -v
```

- Tests in `tests/` use `pytest` with class-based organization.
- CLI tests (`test_cli.py`) invoke `config-explorer` via `subprocess.run`.
- Planner tests (`test_multi_model_gpu_planner.py`) mock heavy deps (streamlit, transformers, huggingface_hub) so they run without the full dependency tree.
- Some tests (capacity_planner_test.py) hit the HuggingFace API and require network access.

## Common pitfalls

- The Streamlit pages import `from src.config_explorer.capacity_planner import ...` which only works when CWD is `config_explorer/`. The `config-explorer start` command handles this.
- `db.json` is loaded relative to CWD (`config_explorer/db.json`), not relative to the module.
- The `llm-optimizer` dependency is a git install from BentoML — it's in `pyproject.toml` but not in `requirements.txt`.
