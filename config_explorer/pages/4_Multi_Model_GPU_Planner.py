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
# Load GPU database
# ---------------------------------------------------------------------------
GPU_DB_PATH = "config_explorer/db.json"

@st.cache_data
def load_gpu_db():
    with open(GPU_DB_PATH) as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# HuggingFace fetch (cached to avoid repeated API calls)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_model_from_hf(model_id: str, hf_token: str | None):
    """Fetch model info and config from HuggingFace. Returns (model_info, model_config) or raises."""
    model_info = get_model_info_from_hf(model_id, hf_token)
    model_config = get_model_config_from_hf(model_id, hf_token)
    return model_info, model_config


def detect_quant_option(model_config) -> str:
    """Map a model's quantization config to one of our QUANT_OPTIONS keys."""
    if not is_quantized(model_config):
        return "BF16 (no quantization)"
    method = get_quant_method(model_config)
    try:
        bpp = precision_to_byte(method)
    except (ValueError, TypeError):
        return "FP8"  # safe fallback for exotic quant methods
    if bpp <= 0.5:
        return "INT4"
    if bpp <= 1.0:
        # Distinguish FP8 vs INT8 by name
        if "int" in method.lower():
            return "INT8"
        return "FP8"
    return "BF16 (no quantization)"


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------
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

# Typical PUE (Power Usage Effectiveness) for data centers
DEFAULT_PUE = 1.4

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class ModelEntry:
    """A single model the user wants to serve."""
    name: str = ""
    params_billion: float = 7.0
    quantization: str = "BF16 (no quantization)"
    is_moe: bool = False
    tp: int = 1
    pp: int = 1
    dp: int = 1
    gpu_type: str = "NVIDIA-H100-80GB-HBM3"
    # When populated from HF, this holds the exact weight memory from SafeTensors
    hf_weight_memory_gib: float | None = None
    fetched: bool = False

    @property
    def bytes_per_param(self) -> float:
        return QUANT_OPTIONS[self.quantization]

    @property
    def model_memory_gib(self) -> float:
        """Model weight memory in GiB. Uses HF exact value if fetched, else estimates."""
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


@dataclass
class InventoryItem:
    """GPUs the user already owns."""
    gpu_type: str = "NVIDIA-H100-80GB-HBM3"
    count: int = 0


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _init():
    if "fleet_models" not in st.session_state:
        st.session_state["fleet_models"] = [ModelEntry()]
    if "inventory" not in st.session_state:
        st.session_state["inventory"] = []
    if "electricity_budget_kw" not in st.session_state:
        st.session_state["electricity_budget_kw"] = 0.0
    if "rack_slots" not in st.session_state:
        st.session_state["rack_slots"] = 0
    if "gpus_per_server" not in st.session_state:
        st.session_state["gpus_per_server"] = 8
    if "pue" not in st.session_state:
        st.session_state["pue"] = DEFAULT_PUE
    if "fetch_errors" not in st.session_state:
        st.session_state["fetch_errors"] = {}


def _get_hf_token() -> str | None:
    """Return the HF token from sidebar input or env var."""
    token = st.session_state.get("hf_token_input", "").strip()
    if token:
        return token
    env = os.environ.get("HF_TOKEN", "").strip()
    return env or None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Multi-Model GPU Planner", layout="wide")
    st.title("Multi-Model GPU Planner")
    st.caption(
        "Plan GPU infrastructure across multiple models. "
        "Estimate how many GPUs to buy and how much electricity you'll use. "
        "Numbers are estimates — use them as a starting point for procurement discussions."
    )

    _init()
    gpu_db = load_gpu_db()
    gpu_names = list(gpu_db.keys())

    # ── Sidebar: HuggingFace token ─────────────────────────────────────────
    with st.sidebar:
        st.header("HuggingFace Token")
        st.text_input(
            "HF Token (for gated models)",
            type="password",
            key="hf_token_input",
            help="Paste your HuggingFace token here, or set HF_TOKEN env var. "
                 "Required for gated models; optional for public models.",
        )
        hf_token = _get_hf_token()
        if hf_token:
            st.success("Token set")
        else:
            st.info("No token — public models only")

    # ── 1. Models to serve ─────────────────────────────────────────────────
    st.header("1 — Models to Serve")
    st.caption(
        "Enter a HuggingFace model ID (e.g. `meta-llama/Llama-3.1-70B`) and click **Fetch** "
        "to auto-populate parameters, quantization, and MoE status. "
        "Or fill in the fields manually."
    )

    models: list[ModelEntry] = st.session_state["fleet_models"]

    cols_header = st.columns([2.2, 1.0, 1.2, 0.5, 0.5, 0.5, 0.5, 1.5, 0.7])
    headers = ["Model name / HF ID", "Params (B)", "Quantization", "MoE?", "TP", "PP", "DP", "GPU type", ""]
    for c, h in zip(cols_header, headers):
        c.markdown(f"**{h}**")

    for i, m in enumerate(models):
        cols = st.columns([2.2, 1.0, 1.2, 0.5, 0.5, 0.5, 0.5, 1.5, 0.7])
        m.name = cols[0].text_input("model", value=m.name, key=f"mn_{i}", label_visibility="collapsed",
                                     placeholder="e.g. meta-llama/Llama-3.1-70B")
        m.params_billion = cols[1].number_input("params", value=m.params_billion, min_value=0.1,
                                                 step=0.1, key=f"mp_{i}", label_visibility="collapsed")
        m.quantization = cols[2].selectbox("quant", list(QUANT_OPTIONS.keys()),
                                            index=list(QUANT_OPTIONS.keys()).index(m.quantization),
                                            key=f"mq_{i}", label_visibility="collapsed")
        m.is_moe = cols[3].checkbox("moe", value=m.is_moe, key=f"mmoe_{i}", label_visibility="collapsed")
        m.tp = cols[4].number_input("tp", value=m.tp, min_value=1, step=1, key=f"mtp_{i}", label_visibility="collapsed")
        m.pp = cols[5].number_input("pp", value=m.pp, min_value=1, step=1, key=f"mpp_{i}", label_visibility="collapsed")
        m.dp = cols[6].number_input("dp", value=m.dp, min_value=1, step=1, key=f"mdp_{i}", label_visibility="collapsed")
        m.gpu_type = cols[7].selectbox("gpu", gpu_names,
                                        index=gpu_names.index(m.gpu_type) if m.gpu_type in gpu_names else 0,
                                        key=f"mg_{i}", label_visibility="collapsed")

        # Fetch button
        if cols[8].button("Fetch", key=f"fetch_{i}", help="Fetch model details from HuggingFace"):
            if m.name.strip() and "/" in m.name:
                with st.spinner(f"Fetching {m.name}..."):
                    try:
                        model_info, model_config = fetch_model_from_hf(m.name.strip(), hf_token)
                        text_config = get_text_config(model_config)

                        # Params
                        total_params = model_total_params(model_info)
                        m.params_billion = round(total_params / 1e9, 2)

                        # Exact weight memory from SafeTensors
                        m.hf_weight_memory_gib = model_memory_req(model_info, model_config)

                        # MoE
                        m.is_moe = is_moe(text_config)

                        # Quantization
                        m.quantization = detect_quant_option(model_config)

                        m.fetched = True
                        st.session_state["fetch_errors"].pop(i, None)
                        st.rerun()
                    except Exception as e:
                        st.session_state["fetch_errors"][i] = str(e)
            else:
                st.session_state["fetch_errors"][i] = "Enter a valid HF model ID (org/model)"

        # Show fetch status or error
        err = st.session_state["fetch_errors"].get(i)
        if err:
            st.error(f"Row {i+1}: {err}")
        elif m.fetched:
            st.caption(f"Row {i+1}: Fetched from HF — weight memory: {m.hf_weight_memory_gib:.2f} GiB (exact from SafeTensors)")

    bcol1, bcol2, bcol3 = st.columns(3)
    if bcol1.button("+ Add model"):
        models.append(ModelEntry())
        st.rerun()
    if bcol2.button("- Remove last model") and len(models) > 1:
        models.pop()
        st.rerun()
    if bcol3.button("Fetch all models"):
        for idx, m in enumerate(models):
            if m.name.strip() and "/" in m.name and not m.fetched:
                try:
                    model_info, model_config = fetch_model_from_hf(m.name.strip(), hf_token)
                    text_config = get_text_config(model_config)
                    total_params = model_total_params(model_info)
                    m.params_billion = round(total_params / 1e9, 2)
                    m.hf_weight_memory_gib = model_memory_req(model_info, model_config)
                    m.is_moe = is_moe(text_config)
                    m.quantization = detect_quant_option(model_config)
                    m.fetched = True
                    st.session_state["fetch_errors"].pop(idx, None)
                except Exception as e:
                    st.session_state["fetch_errors"][idx] = str(e)
        st.rerun()

    # ── 2. Current GPU inventory ───────────────────────────────────────────
    st.header("2 — Current GPU Inventory")
    st.caption("List the GPUs you already own. Leave empty if starting from scratch.")

    inventory: list[InventoryItem] = st.session_state["inventory"]

    if inventory:
        icols_header = st.columns([2, 1])
        icols_header[0].markdown("**GPU type**")
        icols_header[1].markdown("**Count owned**")

        for j, inv in enumerate(inventory):
            icols = st.columns([2, 1])
            inv.gpu_type = icols[0].selectbox("igpu", gpu_names,
                                               index=gpu_names.index(inv.gpu_type) if inv.gpu_type in gpu_names else 0,
                                               key=f"ig_{j}", label_visibility="collapsed")
            inv.count = icols[1].number_input("icnt", value=inv.count, min_value=0, step=1,
                                               key=f"ic_{j}", label_visibility="collapsed")

    ibcol1, ibcol2 = st.columns(2)
    if ibcol1.button("+ Add inventory row"):
        inventory.append(InventoryItem())
        st.rerun()
    if ibcol2.button("- Remove last inventory row") and len(inventory) > 0:
        inventory.pop()
        st.rerun()

    # ── 3. Constraints ─────────────────────────────────────────────────────
    st.header("3 — Constraints")
    ccol1, ccol2, ccol3, ccol4 = st.columns(4)
    electricity_budget_kw = ccol1.number_input(
        "Electricity budget (kW)", value=st.session_state["electricity_budget_kw"],
        min_value=0.0, step=1.0, help="0 = no limit"
    )
    st.session_state["electricity_budget_kw"] = electricity_budget_kw

    rack_slots = ccol2.number_input(
        "Available server/rack slots", value=st.session_state["rack_slots"],
        min_value=0, step=1, help="0 = no limit"
    )
    st.session_state["rack_slots"] = rack_slots

    gpus_per_server = ccol3.number_input(
        "GPUs per server", value=st.session_state["gpus_per_server"],
        min_value=1, step=1
    )
    st.session_state["gpus_per_server"] = gpus_per_server

    pue = ccol4.number_input(
        "PUE (power usage effectiveness)", value=st.session_state["pue"],
        min_value=1.0, step=0.05, help="Typical data center PUE is 1.2-1.6. Includes cooling overhead."
    )
    st.session_state["pue"] = pue

    # ── 4. Compute results ─────────────────────────────────────────────────
    st.header("4 — Procurement & Electricity Plan")

    # Build inventory lookup: gpu_type -> count owned
    owned: dict[str, int] = {}
    for inv in inventory:
        owned[inv.gpu_type] = owned.get(inv.gpu_type, 0) + inv.count

    # Per-model results
    rows = []
    gpu_demand: dict[str, int] = {}  # total GPUs needed per type

    for m in models:
        gpu_spec = gpu_db.get(m.gpu_type, {})
        gpu_mem = gpu_spec.get("memory", 80)
        tdp = gpu_spec.get("tdp_watts", 700)

        total = m.total_gpus
        model_mem = round(m.model_memory_gib, 1)
        per_gpu = round(m.per_gpu_model_memory_gib, 1)
        fits = m.fits_on_gpu(gpu_mem)
        free_kv = round(m.per_gpu_free_for_kv(gpu_mem), 1)
        power_kw = round(total * tdp / 1000, 2)
        source = "HF" if m.fetched else "manual"

        gpu_demand[m.gpu_type] = gpu_demand.get(m.gpu_type, 0) + total

        rows.append({
            "Model": m.name or "(unnamed)",
            "Params (B)": m.params_billion,
            "Quant": m.quantization.split(" ")[0],
            "MoE": m.is_moe,
            "TP×PP×DP": f"{m.tp}×{m.pp}×{m.dp}",
            "GPUs needed": total,
            "GPU type": m.gpu_type,
            "Weight mem (GiB)": model_mem,
            "Per-GPU mem (GiB)": per_gpu,
            "Free for KV (GiB/GPU)": free_kv,
            "Fits?": "Yes" if fits else "NO",
            "Power (kW)": power_kw,
            "Source": source,
        })

    df = pd.DataFrame(rows)
    st.subheader("Per-model breakdown")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Highlight models that don't fit
    not_fitting = df[df["Fits?"] == "NO"]
    if len(not_fitting) > 0:
        st.error(
            f"{len(not_fitting)} model(s) do NOT fit on the selected GPU with current parallelism/quantization. "
            "Increase TP/PP, change GPU, or use a more aggressive quantization."
        )

    # ── Procurement summary ────────────────────────────────────────────────
    st.subheader("GPU procurement summary")

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
            "Total needed": needed,
            "Already owned": have,
            "To purchase": to_buy,
            "Power draw (kW)": round(power, 2),
        })
        total_to_buy += to_buy
        total_power_kw += power

    proc_df = pd.DataFrame(proc_rows)
    st.dataframe(proc_df, use_container_width=True, hide_index=True)

    # ── Totals ─────────────────────────────────────────────────────────────
    st.subheader("Totals")

    total_gpus_needed = sum(gpu_demand.values())
    total_servers = math.ceil(total_gpus_needed / gpus_per_server)
    facility_power_kw = round(total_power_kw * pue, 2)
    annual_kwh = round(facility_power_kw * 8760, 0)

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Total GPUs needed", total_gpus_needed)
    mcol2.metric("GPUs to purchase", total_to_buy)
    mcol3.metric("Servers needed", total_servers)
    mcol4.metric("Models served", len(models))

    pcol1, pcol2, pcol3 = st.columns(3)
    pcol1.metric("GPU power draw", f"{round(total_power_kw, 2)} kW")
    pcol2.metric(f"Facility power (PUE={pue})", f"{facility_power_kw} kW")
    pcol3.metric("Est. annual energy", f"{annual_kwh:,.0f} kWh")

    # ── Constraint warnings ────────────────────────────────────────────────
    if electricity_budget_kw > 0 and facility_power_kw > electricity_budget_kw:
        st.warning(
            f"Facility power ({facility_power_kw} kW) exceeds your electricity budget "
            f"({electricity_budget_kw} kW) by {round(facility_power_kw - electricity_budget_kw, 2)} kW. "
            "Consider more aggressive quantization, fewer DP replicas, or a lower-TDP GPU."
        )
    elif electricity_budget_kw > 0:
        st.success(
            f"Facility power ({facility_power_kw} kW) is within your electricity budget ({electricity_budget_kw} kW). "
            f"Headroom: {round(electricity_budget_kw - facility_power_kw, 2)} kW."
        )

    if rack_slots > 0 and total_servers > rack_slots:
        st.warning(
            f"You need {total_servers} servers but only have {rack_slots} rack slots. "
            f"Shortfall: {total_servers - rack_slots} slots."
        )
    elif rack_slots > 0:
        st.success(
            f"Server requirement ({total_servers}) fits within your rack capacity ({rack_slots} slots). "
            f"Headroom: {rack_slots - total_servers} slots."
        )

    # ── Export section ─────────────────────────────────────────────────────
    st.subheader("Export for finance")
    st.caption("Copy the summary below or download as CSV for your procurement proposal.")

    summary_text = f"""GPU Infrastructure Procurement Summary
======================================
Models to serve: {len(models)}
Total GPUs needed: {total_gpus_needed}
GPUs to purchase: {total_to_buy}
Servers needed: {total_servers} (at {gpus_per_server} GPUs/server)

Power & Electricity:
  GPU power draw: {round(total_power_kw, 2)} kW
  Facility power (PUE {pue}): {facility_power_kw} kW
  Estimated annual energy: {annual_kwh:,.0f} kWh

Procurement by GPU type:
"""
    for _, row in proc_df.iterrows():
        summary_text += f"  {row['GPU type']}: need {row['Total needed']}, own {row['Already owned']}, buy {row['To purchase']} (power: {row['Power draw (kW)']} kW)\n"

    summary_text += "\nModels:\n"
    for _, row in df.iterrows():
        summary_text += f"  {row['Model']}: {row['Params (B)']}B {row['Quant']}, {row['TP×PP×DP']}, {row['GPUs needed']}x {row['GPU type']}, fits={row['Fits?']} ({row['Source']})\n"

    st.code(summary_text, language="text")

    # CSV download buttons
    dcol1, dcol2 = st.columns(2)
    dcol1.download_button(
        "Download model breakdown (CSV)",
        df.to_csv(index=False),
        file_name="gpu_planner_models.csv",
        mime="text/csv",
    )
    dcol2.download_button(
        "Download procurement summary (CSV)",
        proc_df.to_csv(index=False),
        file_name="gpu_planner_procurement.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
