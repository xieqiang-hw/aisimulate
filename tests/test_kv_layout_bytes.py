"""Persistent KV geometry must match the native model used for HBM sizing."""
import pytest
from aisimulate.capacity import estimate_kv_bytes_per_token
from aisimulate.compiler import _resolve_kv_bytes_per_token, prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig

MODEL = "zai-org/GLM-5.2-FP8"

@pytest.mark.parametrize("quant,expected", [(None, 60216), ("fp8", 60216), ("bfloat16", 100152)])
def test_glm_dsa_native_layout_includes_rope_and_indexer(quant, expected):
    assert estimate_kv_bytes_per_token(
        MODEL, tp_size=8, pp_size=1, moe_tp_size=8, moe_ep_size=1,
        kvcache_quant_mode=quant,
    ) == expected

@pytest.mark.parametrize("name", [
    "experiments/disk_swe/configs/swe_p_local_8ssd_p128gb_1x_hbm2048.yaml",
    "experiments/disk_swe/configs/swe_enclosure_sample.yaml",
])
def test_local_and_enclosure_lowering_share_native_geometry(name):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    config = CorePredictionConfig.from_yaml(str(root / name))
    deployment = prediction_to_replay_spec(config).backend_deployment
    for role in ("prefill", "decode"):
        rank = getattr(deployment, role + "_engine_args")
        assert rank["kv_cache_bytes_per_token"] == 60216
        assert rank["block_size"] == 64
        assert rank["native_host_offload"]["num_host_blocks"] == (4152 if role == "prefill" else 8303)
    assert 60216 * 64 * 8 == 30830592
    if config.engine.kv_transfer.enclosure is None:
        assert deployment.prefill_engine_args["kv_transfer_bytes_per_token"] == 60216
    worker = config.engine.workers.prefill.model_copy(deep=True)
    worker.kv_cache.bytes_per_token = 12345
    assert _resolve_kv_bytes_per_token(config.engine, worker, 12345, role="prefill") == 12345


def test_supported_layout_errors_do_not_fall_back(monkeypatch):
    import aisimulate_core.sdk.models as models
    def invalid(*args, **kwargs):
        raise ValueError("invalid native layout")
    monkeypatch.setattr(models, "get_model", invalid)
    with pytest.raises(ValueError, match="invalid native layout"):
        estimate_kv_bytes_per_token(MODEL, tp_size=8, pp_size=1)


def test_fpm_declared_geometry_bypasses_native_model(monkeypatch):
    from types import SimpleNamespace
    import aisimulate_core.sdk.fpm_profile as profiles
    import aisimulate_core.sdk.models as models
    class Deployment:
        resources = SimpleNamespace(cache_layout="linear", kv_bytes_per_token=1234)
        def validate_overrides(self, **kwargs):
            assert kwargs == {"kvcache_quant_mode": "fp8"}
    class Profile:
        def select(self, **identity):
            assert identity["worker_type"] == "decode"
            assert identity["tp_size"] == 8
            return Deployment()
    monkeypatch.setattr(profiles, "load_fpm_profile", lambda _: Profile())
    monkeypatch.setattr(models, "get_model", lambda *a, **k: pytest.fail("FPM must not build a model"))
    assert estimate_kv_bytes_per_token(
        "profile-only", tp_size=8, pp_size=1, fpm_profile={"path": "unused"},
        worker_type="decode", kvcache_quant_mode="fp8",
    ) == 1234
