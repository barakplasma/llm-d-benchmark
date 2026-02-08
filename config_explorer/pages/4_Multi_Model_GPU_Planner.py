"""
Multi-Model GPU Planner

A holistic tool for planning GPU infrastructure across multiple models simultaneously.
Estimates total GPU requirements, electricity usage, and generates a procurement summary
suitable for presenting to finance departments.

Supports fetching real model metadata from HuggingFace for accurate memory estimates.
"""

import streamlit as st
import pandas as pd
import json
import math
import os
from dataclasses import dataclass

from src.config_explorer.capacity_planner import (
    get_model_info_from_hf,
    get_model_config_from_hf,
    get_text_config,
    model_memory_req,
    model_total_params,
    is_moe,
    is_quantized,
    get_quant_method,
    precision_to_byte,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GPU_DB_PATH = "config_explorer/db.json"
DEFAULT_PUE = 1.4

QUANT_OPTIONS = {
    "BF16 (no quantization)": 2.0,
    "FP8": 1.0,
    "INT8": 1.0,
    "INT4": 0.5,
}

# Activation memory constants (from capacity_planner.py empirical data)
ACTIVATION_MEM_DENSE_GIB = 5.5
ACTIVATION_MEM_MOE_GIB = 8.0
NON_TORCH_MEM_TP1_GIB = 0.15
NON_TORCH_MEM_TPN_GIB = 0.6

DEFAULT_GPU = "NVIDIA-H100-SXM-80GB"


# ---------------------------------------------------------------------------
# GPU database
# ---------------------------------------------------------------------------
@st.cache_data
def load_gpu_db():
    with open(GPU_DB_PATH) as f:
        raw = json.load(f)
    # Filter out metadata keys (start with _)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def gpu_label(name: str, spec: dict) -> str:
    """Human-readable GPU label for dropdowns."""
    return f"{name}  ({spec['memory']} GB, {spec['tdp_watts']}W)"


# ---------------------------------------------------------------------------
# HuggingFace fetch (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_model_from_hf(model_id: str, hf_token: str | None):
    model_info = get_model_info_from_hf(model_id, hf_token)
    model_config = get_model_config_from_hf(model_id, hf_token)
    return model_info, model_config


def detect_quant_option(model_config) -> str:
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


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class ModelEntry:
    name: str = ""
    params_billion: float = 7.0
    quantization: str = "BF16 (no quantization)"
    is_moe: bool = False
    tp: int = 1
    pp: int = 1
    dp: int = 1
    gpu_type: str = DEFAULT_GPU
    hf_weight_memory_gib: float | None = None
    fetched: bool = False

    @property
    def bytes_per_param(self) -> float:
        return QUANT_OPTIONS[self.quantization]

    @property
    def model_memory_gib(self) -> float:
        if self.hf_weight_memory_gib is not None:
            return self.hf_weight_memory_gib
        return self.params_billion * 1e9 * self.bytes_per_param / (1024**3)

    @property
    def per_gpu_model_memory_gib(self) -> float:
        return self.model_memory_gib / (self.tp * self.pp)

    @property
    def activation_memory_gib(self) -> float:
        return ACTIVATION_MEM_MOE_GIB if self.is_moe else ACTIVATION_MEM_DENSE_GIB

    @property
    def non_torch_memory_gib(self) -> float:
        return NON_TORCH_MEM_TP1_GIB if self.tp == 1 else NON_TORCH_MEM_TPN_GIB

    @property
    def total_gpus(self) -> int:
        return self.tp * self.pp * self.dp

    def fits_on_gpu(self, gpu_memory_gib: float, gpu_mem_util: float = 0.9) -> bool:
        available = gpu_memory_gib * gpu_mem_util
        needed = self.per_gpu_model_memory_gib + self.activation_memory_gib + self.non_torch_memory_gib
        return needed <= available

    def per_gpu_free_for_kv(self, gpu_memory_gib: float, gpu_mem_util: float = 0.9) -> float:
        available = gpu_memory_gib * gpu_mem_util
        needed = self.per_gpu_model_memory_gib + self.activation_memory_gib + self.non_torch_memory_gib
        return max(0.0, available - needed)

    @property
    def label(self) -> str:
        q = self.quantization.split(" ")[0]
        return f"{self.name or '(unnamed)'} — {self.params_billion}B {q}"


@dataclass
class InventoryItem:
    gpu_type: str = DEFAULT_GPU
    count: int = 0


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init():
    for key, default in {
        "fleet_models": [ModelEntry()],
        "inventory": [],
        "electricity_budget_kw": 0.0,
        "rack_slots": 0,
        "gpus_per_server": 8,
        "pue": DEFAULT_PUE,
        "fetch_errors": {},
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _get_hf_token() -> str | None:
    token = st.session_state.get("hf_token_input", "").strip()
    if token:
        return token
    env = os.environ.get("HF_TOKEN", "").strip()
    return env or None


# ---------------------------------------------------------------------------
# Fetch helper
# ---------------------------------------------------------------------------
def _do_fetch(m: ModelEntry, idx: int, hf_token: str | None) -> bool:
    """Fetch HF metadata for a ModelEntry. Returns True on success."""
    try:
        model_info, model_config = fetch_model_from_hf(m.name.strip(), hf_token)
        text_config = get_text_config(model_config)
        m.params_billion = round(model_total_params(model_info) / 1e9, 2)
        m.hf_weight_memory_gib = model_memory_req(model_info, model_config)
        m.is_moe = is_moe(text_config)
        m.quantization = detect_quant_option(model_config)
        m.fetched = True
        st.session_state["fetch_errors"].pop(idx, None)
        return True
    except Exception as e:
        st.session_state["fetch_errors"][idx] = str(e)
        return False


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Multi-Model GPU Planner", layout="wide")
    _init()
    gpu_db = load_gpu_db()
    gpu_names = list(gpu_db.keys())
    # Build a reverse map from label -> name for selectboxes
    gpu_labels = [gpu_label(n, gpu_db[n]) for n in gpu_names]
    label_to_name = dict(zip(gpu_labels, gpu_names))

    # ── Header ─────────────────────────────────────────────────────────────
    st.title("Multi-Model GPU Planner")
    st.markdown(
        "Plan GPU infrastructure across multiple models. "
        "Estimate GPUs to buy and electricity to budget."
    )

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")

        st.text_input(
            "HuggingFace token",
            type="password",
            key="hf_token_input",
            help="Paste your HF token or set the HF_TOKEN env var. Needed for gated models.",
        )
        hf_token = _get_hf_token()
        if hf_token:
            st.caption("Token set")

        st.divider()
        st.subheader("Infrastructure")

        st.session_state["gpus_per_server"] = st.number_input(
            "GPUs per server", value=st.session_state["gpus_per_server"],
            min_value=1, step=1,
        )
        st.session_state["pue"] = st.number_input(
            "PUE (cooling overhead)", value=st.session_state["pue"],
            min_value=1.0, step=0.05,
            help="Power Usage Effectiveness. 1.0 = no overhead, 1.4 = typical DC.",
        )
        st.session_state["electricity_budget_kw"] = st.number_input(
            "Electricity budget (kW)", value=st.session_state["electricity_budget_kw"],
            min_value=0.0, step=1.0, help="0 = no limit",
        )
        st.session_state["rack_slots"] = st.number_input(
            "Rack slots available", value=st.session_state["rack_slots"],
            min_value=0, step=1, help="0 = no limit",
        )

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab_models, tab_inventory, tab_results = st.tabs([
        "Models", "GPU Inventory", "Results"
    ])

    models: list[ModelEntry] = st.session_state["fleet_models"]
    inventory: list[InventoryItem] = st.session_state["inventory"]

    # ── TAB 1: Models ──────────────────────────────────────────────────────
    with tab_models:
        st.markdown(
            "Add each model you want to serve. "
            "Enter a HuggingFace ID and press **Fetch** to auto-fill, or enter details manually."
        )

        for i, m in enumerate(models):
            # Determine the current gpu_label for the selectbox
            current_label_idx = gpu_names.index(m.gpu_type) if m.gpu_type in gpu_names else 0
            status = "HF" if m.fetched else ""

            with st.expander(f"**Model {i+1}:** {m.label}  {status}", expanded=(i == len(models) - 1)):
                # Row 1: Model ID + Fetch
                r1a, r1b = st.columns([4, 1])
                m.name = r1a.text_input(
                    "HuggingFace model ID",
                    value=m.name,
                    key=f"mn_{i}",
                    placeholder="org/model-name (e.g. meta-llama/Llama-3.1-70B)",
                )
                if r1b.button("Fetch", key=f"fetch_{i}", use_container_width=True):
                    if m.name.strip() and "/" in m.name:
                        with st.spinner(f"Fetching {m.name}…"):
                            _do_fetch(m, i, hf_token)
                        st.rerun()
                    else:
                        st.session_state["fetch_errors"][i] = "Enter a valid HF model ID (org/model)"

                err = st.session_state["fetch_errors"].get(i)
                if err:
                    st.error(err)
                elif m.fetched:
                    st.caption(
                        f"Fetched from HuggingFace — exact weight memory: "
                        f"**{m.hf_weight_memory_gib:.2f} GiB** from SafeTensors"
                    )

                # Row 2: Params + Quant + MoE
                r2a, r2b, r2c = st.columns(3)
                m.params_billion = r2a.number_input(
                    "Parameters (billions)", value=m.params_billion,
                    min_value=0.1, step=0.1, key=f"mp_{i}",
                )
                m.quantization = r2b.selectbox(
                    "Quantization", list(QUANT_OPTIONS.keys()),
                    index=list(QUANT_OPTIONS.keys()).index(m.quantization),
                    key=f"mq_{i}",
                )
                m.is_moe = r2c.toggle("Mixture of Experts", value=m.is_moe, key=f"mmoe_{i}")

                # Row 3: Parallelism
                r3a, r3b, r3c = st.columns(3)
                m.tp = r3a.number_input("Tensor Parallel (TP)", value=m.tp, min_value=1, step=1, key=f"mtp_{i}")
                m.pp = r3b.number_input("Pipeline Parallel (PP)", value=m.pp, min_value=1, step=1, key=f"mpp_{i}")
                m.dp = r3c.number_input("Data Parallel (DP)", value=m.dp, min_value=1, step=1, key=f"mdp_{i}")

                # Row 4: GPU
                selected_label = st.selectbox(
                    "GPU", gpu_labels, index=current_label_idx, key=f"mg_{i}",
                )
                m.gpu_type = label_to_name[selected_label]

                # Quick feedback
                gpu_mem = gpu_db[m.gpu_type]["memory"]
                fits = m.fits_on_gpu(gpu_mem)
                gpus = m.total_gpus
                mem = m.model_memory_gib
                per = m.per_gpu_model_memory_gib

                if fits:
                    st.success(
                        f"**{gpus} GPU{'s' if gpus > 1 else ''}** — "
                        f"weight {mem:.1f} GiB total, {per:.1f} GiB/GPU, "
                        f"{m.per_gpu_free_for_kv(gpu_mem):.1f} GiB free for KV cache"
                    )
                else:
                    st.error(
                        f"Does not fit — needs {per:.1f} GiB/GPU + "
                        f"{m.activation_memory_gib:.1f} GiB activation, "
                        f"but only {gpu_mem * 0.9:.1f} GiB available. "
                        "Increase TP/PP or use a more aggressive quantization."
                    )

        # Action buttons
        c1, c2, c3 = st.columns(3)
        if c1.button("Add model", use_container_width=True):
            models.append(ModelEntry())
            st.rerun()
        if c2.button("Remove last", use_container_width=True) and len(models) > 1:
            models.pop()
            st.rerun()
        if c3.button("Fetch all", use_container_width=True):
            for idx, m in enumerate(models):
                if m.name.strip() and "/" in m.name and not m.fetched:
                    _do_fetch(m, idx, hf_token)
            st.rerun()

    # ── TAB 2: GPU Inventory ───────────────────────────────────────────────
    with tab_inventory:
        st.markdown("List GPUs you **already own**. The planner will subtract these from the purchase total.")

        if not inventory:
            st.info("No GPUs in inventory. Click below to add.")

        for j, inv in enumerate(inventory):
            current_inv_idx = gpu_names.index(inv.gpu_type) if inv.gpu_type in gpu_names else 0
            ic1, ic2 = st.columns([3, 1])
            selected_inv_label = ic1.selectbox(
                "GPU type", gpu_labels, index=current_inv_idx, key=f"ig_{j}",
            )
            inv.gpu_type = label_to_name[selected_inv_label]
            inv.count = ic2.number_input("Count", value=inv.count, min_value=0, step=1, key=f"ic_{j}")

        ic1, ic2 = st.columns(2)
        if ic1.button("Add inventory row", use_container_width=True):
            inventory.append(InventoryItem())
            st.rerun()
        if ic2.button("Remove last row", use_container_width=True) and inventory:
            inventory.pop()
            st.rerun()

    # ── TAB 3: Results ─────────────────────────────────────────────────────
    with tab_results:
        electricity_budget_kw = st.session_state["electricity_budget_kw"]
        rack_slots = st.session_state["rack_slots"]
        gpus_per_server = st.session_state["gpus_per_server"]
        pue = st.session_state["pue"]

        # Build inventory lookup
        owned: dict[str, int] = {}
        for inv in inventory:
            owned[inv.gpu_type] = owned.get(inv.gpu_type, 0) + inv.count

        # Per-model calculation
        rows = []
        gpu_demand: dict[str, int] = {}

        for m in models:
            spec = gpu_db.get(m.gpu_type, {})
            gpu_mem = spec.get("memory", 80)
            tdp = spec.get("tdp_watts", 700)
            total = m.total_gpus
            fits = m.fits_on_gpu(gpu_mem)
            power_kw = round(total * tdp / 1000, 2)
            gpu_demand[m.gpu_type] = gpu_demand.get(m.gpu_type, 0) + total

            rows.append({
                "Model": m.name or "(unnamed)",
                "Params (B)": m.params_billion,
                "Quant": m.quantization.split(" ")[0],
                "MoE": m.is_moe,
                "TP×PP×DP": f"{m.tp}×{m.pp}×{m.dp}",
                "GPUs": total,
                "GPU type": m.gpu_type,
                "Weight (GiB)": round(m.model_memory_gib, 1),
                "Per-GPU (GiB)": round(m.per_gpu_model_memory_gib, 1),
                "KV free (GiB)": round(m.per_gpu_free_for_kv(gpu_mem), 1),
                "Fits": "Yes" if fits else "NO",
                "Power (kW)": power_kw,
                "Source": "HF" if m.fetched else "manual",
            })

        df = pd.DataFrame(rows)

        # Procurement
        proc_rows = []
        total_to_buy = 0
        total_power_kw = 0.0

        for gpu_type, needed in gpu_demand.items():
            have = owned.get(gpu_type, 0)
            to_buy = max(0, needed - have)
            tdp = gpu_db.get(gpu_type, {}).get("tdp_watts", 700)
            power = needed * tdp / 1000
            proc_rows.append({
                "GPU type": gpu_type,
                "Needed": needed,
                "Owned": have,
                "To buy": to_buy,
                "Power (kW)": round(power, 2),
            })
            total_to_buy += to_buy
            total_power_kw += power

        proc_df = pd.DataFrame(proc_rows)

        total_gpus = sum(gpu_demand.values())
        total_servers = math.ceil(total_gpus / gpus_per_server)
        facility_kw = round(total_power_kw * pue, 2)
        annual_kwh = round(facility_kw * 8760, 0)

        # ── Headline metrics ──────────────────────────────────────────────
        st.subheader("What to buy")
        m1, m2, m3 = st.columns(3)
        m1.metric("GPUs to purchase", f"{total_to_buy}")
        m2.metric("Servers needed", f"{total_servers}")
        m3.metric("Models served", f"{len(models)}")

        st.subheader("Power & electricity")
        p1, p2, p3 = st.columns(3)
        p1.metric("GPU power", f"{round(total_power_kw, 1)} kW")
        p2.metric(f"Facility (PUE {pue})", f"{facility_kw} kW")
        p3.metric("Annual energy", f"{annual_kwh:,.0f} kWh")

        # ── Warnings ──────────────────────────────────────────────────────
        not_fitting = [r for r in rows if r["Fits"] == "NO"]
        if not_fitting:
            st.error(
                f"{len(not_fitting)} model(s) do NOT fit on the selected GPU. "
                "Go back to the Models tab and increase TP/PP or use more aggressive quantization."
            )

        if electricity_budget_kw > 0:
            if facility_kw > electricity_budget_kw:
                st.warning(
                    f"Over electricity budget by {round(facility_kw - electricity_budget_kw, 1)} kW "
                    f"({facility_kw} kW needed vs {electricity_budget_kw} kW budget)."
                )
            else:
                st.success(
                    f"Within electricity budget. Headroom: "
                    f"{round(electricity_budget_kw - facility_kw, 1)} kW."
                )

        if rack_slots > 0:
            if total_servers > rack_slots:
                st.warning(
                    f"Need {total_servers} servers but only {rack_slots} rack slots available."
                )
            else:
                st.success(f"Fits in rack. Headroom: {rack_slots - total_servers} slots.")

        # ── Tables ────────────────────────────────────────────────────────
        st.subheader("GPU procurement by type")
        st.dataframe(proc_df, use_container_width=True, hide_index=True)

        st.subheader("Per-model breakdown")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Export ────────────────────────────────────────────────────────
        st.subheader("Export for finance")

        summary = f"""GPU Infrastructure Procurement Summary
======================================
Models to serve: {len(models)}
Total GPUs needed: {total_gpus}
GPUs to purchase: {total_to_buy}
Servers needed: {total_servers} (at {gpus_per_server} GPUs/server)

Power & Electricity
  GPU power draw: {round(total_power_kw, 1)} kW
  Facility power (PUE {pue}): {facility_kw} kW
  Est. annual energy: {annual_kwh:,.0f} kWh

Procurement by GPU type
"""
        for _, row in proc_df.iterrows():
            summary += (
                f"  {row['GPU type']}: need {row['Needed']}, "
                f"own {row['Owned']}, buy {row['To buy']} "
                f"({row['Power (kW)']} kW)\n"
            )
        summary += "\nModels\n"
        for _, row in df.iterrows():
            summary += (
                f"  {row['Model']}: {row['Params (B)']}B {row['Quant']}, "
                f"{row['TP×PP×DP']}, {row['GPUs']}x {row['GPU type']}, "
                f"fits={row['Fits']} ({row['Source']})\n"
            )

        st.code(summary, language="text")

        d1, d2 = st.columns(2)
        d1.download_button(
            "Download models CSV",
            df.to_csv(index=False),
            file_name="gpu_planner_models.csv",
            mime="text/csv",
            use_container_width=True,
        )
        d2.download_button(
            "Download procurement CSV",
            proc_df.to_csv(index=False),
            file_name="gpu_planner_procurement.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
