"""Tests for Multi-Model GPU Planner server allocation logic.

We mock out heavy dependencies (streamlit, transformers, huggingface_hub)
so these tests run with just pytest + stdlib.
"""

import sys
import types
import os
from unittest import mock
from dataclasses import dataclass

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy optional imports so we can load the planner module without
# installing streamlit / transformers / huggingface_hub in CI.
# ---------------------------------------------------------------------------
_STUBS = {}
for mod_name in (
    "streamlit",
    "transformers",
    "huggingface_hub",
    "src",
    "src.config_explorer",
    "src.config_explorer.capacity_planner",
):
    stub = types.ModuleType(mod_name)
    # streamlit needs a few attributes the module references at import time
    if mod_name == "streamlit":
        stub.cache_data = lambda *a, **kw: (lambda fn: fn)  # noqa: E731
        stub.set_page_config = lambda **kw: None
        stub.session_state = {}
    _STUBS[mod_name] = stub

# Provide all names imported from capacity_planner as no-op stubs
_cp = _STUBS["src.config_explorer.capacity_planner"]
for _name in (
    "get_model_info_from_hf",
    "get_model_config_from_hf",
    "get_text_config",
    "model_memory_req",
    "model_total_params",
    "is_moe",
    "is_quantized",
    "get_quant_method",
    "precision_to_byte",
):
    setattr(_cp, _name, lambda *a, **kw: None)

with mock.patch.dict(sys.modules, _STUBS):
    # Now we can safely import the planner
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), os.pardir, "pages")
    )
    from importlib import import_module

    _planner = import_module("4_Multi_Model_GPU_Planner")

ModelEntry = _planner.ModelEntry
allocate_servers = _planner.allocate_servers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
GPU = "NVIDIA-H100-SXM-80GB"
# Minimal gpu_db entry – only "memory" is used by fits_on_gpu()
GPU_DB = {GPU: {"memory": 80, "tdp_watts": 700}}


def _model(name="m", tp=1, pp=1, dp=1, params=7.0, gpu=GPU, **kw):
    return ModelEntry(
        name=name, tp=tp, pp=pp, dp=dp,
        params_billion=params, gpu_type=gpu, **kw,
    )


def _total_slots(servers):
    return sum(len(s["slots"]) for s in servers)


# ---------------------------------------------------------------------------
# Basic packing
# ---------------------------------------------------------------------------
class TestBasicPacking:
    def test_single_model_single_gpu(self):
        servers, errors = allocate_servers([_model(tp=1, pp=1, dp=1)], 8, GPU_DB)
        assert not errors
        assert len(servers) == 1
        assert len(servers[0]["slots"]) == 1

    def test_single_model_fills_one_server(self):
        # TP=8 on 8-GPU server → exactly 1 server
        servers, errors = allocate_servers([_model(tp=8, pp=1, dp=1)], 8, GPU_DB)
        assert not errors
        assert len(servers) == 1
        assert len(servers[0]["slots"]) == 8

    def test_dp_replicas_pack_same_server(self):
        # TP=2, DP=4 → 8 GPUs, fits in 1 server of 8
        servers, errors = allocate_servers([_model(tp=2, pp=1, dp=4)], 8, GPU_DB)
        assert not errors
        assert len(servers) == 1
        assert len(servers[0]["slots"]) == 8

    def test_dp_replicas_overflow_to_second_server(self):
        # TP=2, DP=4 → 8 GPUs, but only 4 per server → 2 servers
        servers, errors = allocate_servers([_model(tp=2, pp=1, dp=4)], 4, GPU_DB)
        assert not errors
        assert len(servers) == 2
        assert _total_slots(servers) == 8


# ---------------------------------------------------------------------------
# PP stages spanning servers  (the main bug fix)
# ---------------------------------------------------------------------------
class TestPipelineParallelSpansServers:
    def test_tp8_pp2_needs_two_servers_on_8gpu_nodes(self):
        """TP=8, PP=2 → 16 GPUs.  Each PP stage is 8 GPUs (one full server).
        The old code tried to cram all 16 onto one 8-GPU server."""
        servers, errors = allocate_servers(
            [_model(tp=8, pp=2, dp=1)], 8, GPU_DB,
        )
        assert not errors
        assert len(servers) == 2
        assert _total_slots(servers) == 16
        for s in servers:
            assert len(s["slots"]) <= s["capacity"]

    def test_tp4_pp4_spans_four_servers(self):
        # TP=4, PP=4 → 16 GPUs, 4 per stage → on 4-GPU servers → 4 servers
        servers, errors = allocate_servers(
            [_model(tp=4, pp=4, dp=1)], 4, GPU_DB,
        )
        assert not errors
        assert len(servers) == 4
        assert _total_slots(servers) == 16

    def test_tp4_pp2_fits_one_server_of_8(self):
        # TP=4, PP=2 → 8 GPUs.  Each stage is 4 GPUs.
        # Both stages fit on one 8-GPU server.
        servers, errors = allocate_servers(
            [_model(tp=4, pp=2, dp=1)], 8, GPU_DB,
        )
        assert not errors
        assert len(servers) == 1
        assert _total_slots(servers) == 8

    def test_tp8_pp2_dp2_four_servers(self):
        # 2 replicas × (TP=8 × PP=2) = 32 GPUs → 4 × 8-GPU servers
        servers, errors = allocate_servers(
            [_model(tp=8, pp=2, dp=2)], 8, GPU_DB,
        )
        assert not errors
        assert len(servers) == 4
        assert _total_slots(servers) == 32
        for s in servers:
            assert len(s["slots"]) <= s["capacity"]


# ---------------------------------------------------------------------------
# TP > gpus_per_server  (impossible configuration)
# ---------------------------------------------------------------------------
class TestTPExceedsServerCapacity:
    def test_tp_exceeds_gpus_per_server_gives_error(self):
        servers, errors = allocate_servers(
            [_model(tp=16, pp=1, dp=1)], 8, GPU_DB,
        )
        assert len(errors) == 1
        assert "TP=16" in errors[0]
        assert _total_slots(servers) == 0

    def test_partial_allocation_with_one_bad_model(self):
        """One model fits, another has TP too large. Only the good one placed."""
        good = _model(name="good", tp=4, pp=1, dp=1)
        bad = _model(name="bad", tp=16, pp=1, dp=1)
        servers, errors = allocate_servers([good, bad], 8, GPU_DB)
        assert len(errors) == 1
        assert "bad" in errors[0]
        assert _total_slots(servers) == 4


# ---------------------------------------------------------------------------
# No server overflows its capacity
# ---------------------------------------------------------------------------
class TestNoOverflow:
    def test_no_server_exceeds_capacity(self):
        """Regression: no server should ever have more slots than capacity."""
        models = [
            _model(name="a", tp=8, pp=2, dp=2),
            _model(name="b", tp=4, pp=1, dp=3),
            _model(name="c", tp=2, pp=4, dp=1),
        ]
        servers, errors = allocate_servers(models, 8, GPU_DB)
        assert not errors
        for s in servers:
            assert len(s["slots"]) <= s["capacity"], (
                f"Server overflowed: {len(s['slots'])} slots "
                f"> {s['capacity']} capacity"
            )

    def test_large_pp_does_not_overflow(self):
        """PP=8, TP=4 on 4-GPU servers → 8 servers, none overflowed."""
        servers, errors = allocate_servers(
            [_model(tp=4, pp=8, dp=1)], 4, GPU_DB,
        )
        assert not errors
        assert len(servers) == 8
        for s in servers:
            assert len(s["slots"]) <= s["capacity"]


# ---------------------------------------------------------------------------
# Multi-model co-location
# ---------------------------------------------------------------------------
class TestMultiModelPacking:
    def test_two_small_models_share_server(self):
        a = _model(name="a", tp=2, pp=1, dp=1)
        b = _model(name="b", tp=2, pp=1, dp=1)
        servers, errors = allocate_servers([a, b], 8, GPU_DB)
        assert not errors
        assert len(servers) == 1
        assert len(servers[0]["slots"]) == 4

    def test_different_gpu_types_separate_servers(self):
        gpu_b = "NVIDIA-A100-SXM-80GB"
        db = {**GPU_DB, gpu_b: {"memory": 80, "tdp_watts": 400}}
        a = _model(name="a", tp=2, pp=1, dp=1, gpu=GPU)
        b = _model(name="b", tp=2, pp=1, dp=1, gpu=gpu_b)
        servers, errors = allocate_servers([a, b], 8, db)
        assert not errors
        assert len(servers) == 2


# ---------------------------------------------------------------------------
# Model too large for GPU memory is silently skipped
# ---------------------------------------------------------------------------
class TestMemoryFit:
    def test_model_too_large_for_gpu_is_skipped(self):
        # 1000B params at BF16 ≈ 1863 GiB – won't fit on 80 GB GPU
        big = _model(name="huge", tp=1, pp=1, dp=1, params=1000.0)
        servers, errors = allocate_servers([big], 8, GPU_DB)
        assert not errors
        assert len(servers) == 0
