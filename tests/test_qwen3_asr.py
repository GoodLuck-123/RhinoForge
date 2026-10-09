"""Host-only admission, preprocessing and ownership checks for the ASR profile."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rpu_backend.api import Qwen3ASRModel
from rpu_backend.api import qwen3_asr as asr
from rpu_backend.api.errors import RPUSingleHandleError, RPUUnsupportedDtypeError, UnsupportedModelError


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, "fp16"])
def test_dtype_rejected_before_checkpoint_access(tmp_path, dtype):
    with pytest.raises(RPUUnsupportedDtypeError):
        Qwen3ASRModel.from_pretrained(tmp_path / "missing", dtype=dtype)


def test_unreviewed_config_rejected_before_loading(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_asr"}')
    monkeypatch.setattr(asr, "_load_components", lambda *a: pytest.fail("weights loaded"))
    with pytest.raises(UnsupportedModelError, match="config.json"):
        Qwen3ASRModel.from_pretrained(tmp_path)


@pytest.mark.parametrize("extra", ["tokenizer.json", "added_tokens.json", "special_tokens_map.json"])
def test_tokenizer_overrides_rejected_before_loading(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(asr, "_IDENTITIES", {})
    monkeypatch.setattr(asr, "_WEIGHT_SIZE", 0)
    (tmp_path / "model.safetensors").touch()
    (tmp_path / extra).write_text("{}")
    with pytest.raises(UnsupportedModelError, match="original tokenizer files"):
        asr._preflight(tmp_path, torch.float16)


@pytest.mark.parametrize("audio,rate,tokens", [
    (np.zeros(7999, dtype=np.float32), 16000, 64),
    (np.zeros(128001, dtype=np.float32), 16000, 64),
    (np.zeros((1, 16000), dtype=np.float32), 16000, 64),
    (np.zeros(16000, dtype=np.int16), 16000, 64),
    (np.full(16000, np.nan), 16000, 64),
    (np.full(16000, np.inf), 16000, 64),
    (np.zeros(16000, dtype=np.float32), 24000, 64),
    (np.zeros(16000, dtype=np.float32), 16000, 65),
    (np.zeros(16000, dtype=np.float32), 16000, True),
    (np.zeros(16000, dtype=np.float32), 16000, 1.5),
])
def test_invalid_audio_rejected_without_inference(audio, rate, tokens):
    with pytest.raises(ValueError):
        asr._validate_audio(audio, rate, tokens)


@pytest.mark.parametrize("samples", [8000, 67263, 128000])
def test_valid_audio_boundaries(samples):
    value = asr._validate_audio(torch.zeros(samples), 16000, 64)
    assert value.shape == (samples,)
    assert value.dtype == np.float32


@pytest.mark.parametrize("frames,expected", [(50, 7), (99, 13), (100, 13), (101, 14), (420, 55), (800, 104)])
def test_audio_tokens_respect_chunk_boundary(frames, expected):
    assert asr._audio_token_length(frames) == expected


def test_preprocessor_checks_feature_placeholder_agreement():
    def extractor(*args, **kwargs):
        return {"attention_mask": torch.ones(1, 420, dtype=torch.long),
                "input_features": torch.zeros(1, 128, 420)}

    def tokenizer(prompts, **kwargs):
        assert prompts == ["prefix" + "<|audio_pad|>" * 55]
        return {"input_ids": torch.tensor([[151676] * 54]), "attention_mask": torch.ones(1, 54)}

    with pytest.raises(ValueError, match="placeholder count"):
        asr._prepare_inputs(tokenizer, extractor, "prefix<|audio_pad|>", np.zeros(67263, dtype=np.float32))


def test_failed_load_releases_owner_and_does_not_install(tmp_path, monkeypatch):
    from rpu_backend.api import causal_lm

    events = []
    monkeypatch.setattr(asr, "_preflight", lambda *args: {})
    monkeypatch.setattr(asr, "_load_processor", lambda *args: (None, None, ""))
    monkeypatch.setattr(causal_lm, "_claim_live_instance", lambda owner: events.append("claim"))
    monkeypatch.setattr(causal_lm, "_release_live_instance", lambda owner: events.append("release"))
    monkeypatch.setattr(Qwen3ASRModel, "_install_rpu", lambda self: pytest.fail("RPU accessed"))

    def fail(*args):
        events.append("load")
        raise ValueError("invalid weights")

    monkeypatch.setattr(asr, "_load_components", fail)
    with pytest.raises(ValueError, match="invalid weights"):
        Qwen3ASRModel.from_pretrained(tmp_path)
    assert events == ["claim", "load", "release"]


def test_occupied_owner_rejects_before_processor_or_weights(tmp_path, monkeypatch):
    from rpu_backend.api import causal_lm

    monkeypatch.setattr(asr, "_preflight", lambda *args: {})
    monkeypatch.setattr(asr, "_load_processor", lambda *args: pytest.fail("processor loaded"))
    monkeypatch.setattr(asr, "_load_components", lambda *args: pytest.fail("weights loaded"))

    def occupied(owner):
        raise RPUSingleHandleError("occupied")

    monkeypatch.setattr(causal_lm, "_claim_live_instance", occupied)
    with pytest.raises(RPUSingleHandleError, match="occupied"):
        Qwen3ASRModel.from_pretrained(tmp_path)


def test_close_retires_before_releasing_and_is_idempotent(monkeypatch):
    from rpu_backend.api import causal_lm

    events = []
    monkeypatch.setattr(causal_lm, "_release_live_instance", lambda owner: events.append("release"))
    model = Qwen3ASRModel()
    model._closed = False
    model._text = SimpleNamespace(_rpu_decoder_retirement_state=SimpleNamespace(retire=lambda: events.append("retire")))
    model.close()
    model.close()
    assert events == ["retire", "release"]
    assert model._text is None
    with pytest.raises(RuntimeError, match="closed"):
        model.transcribe(np.zeros(8000, dtype=np.float32))
    with pytest.raises(RuntimeError, match="closed"):
        model.execution_stats()


def test_close_cannot_release_an_active_request():
    model = Qwen3ASRModel()
    model._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="serial"):
            model.close()
        with pytest.raises(RuntimeError, match="serial"):
            model.execution_stats()
    finally:
        model._lock.release()


def test_failed_retirement_does_not_release_owner(monkeypatch):
    from rpu_backend.api import causal_lm

    events = []
    monkeypatch.setattr(causal_lm, "_release_live_instance", lambda owner: events.append("release"))
    model = Qwen3ASRModel()
    model._closed = False

    def fail():
        raise RuntimeError("retirement failed")

    model._text = SimpleNamespace(_rpu_decoder_retirement_state=SimpleNamespace(retire=fail))
    with pytest.raises(RuntimeError, match="retirement failed"):
        model.close()
    assert events == [] and not model._closed and model._text is not None
    # No real native resource exists in this host test.
    model._closed = True


def test_decoder_install_bounds_graphs_to_leave_audio_launch_capacity(monkeypatch):
    from rpu_backend.runtime import decoder, topology, weights

    captured = {}
    model = Qwen3ASRModel()
    table = torch.zeros(1, 512, 128, dtype=torch.float16)
    model._text = SimpleNamespace(
        rotary_emb=lambda *args: (table, table),
        embed_tokens=SimpleNamespace(weight=torch.zeros(1, 1024, dtype=torch.float16)),
        to=lambda device: None,
    )
    monkeypatch.setattr(topology, "resolve_decoder_topology", lambda **kwargs: object())
    monkeypatch.setattr(weights, "convert_linear_weights_inplace", lambda model: None)

    def install(model, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop before audio installation")

    monkeypatch.setattr(decoder, "_install_causal_decoder_forward", install)
    with pytest.raises(RuntimeError, match="stop before audio"):
        model._install_rpu()
    assert captured["_graph_cache_max_entries"] == 2


def test_shape_switch_releases_graphs_before_audio_and_same_shape_replays(monkeypatch):
    events = []
    graphs = set()

    def clear():
        events.append("clear")
        graphs.clear()

    model = Qwen3ASRModel()
    model._closed = False
    model._tokenizer = model._feature_extractor = model._prompt = None
    model._text = SimpleNamespace(_rpu_decoder_graph_cache=SimpleNamespace(clear=clear))
    monkeypatch.setattr(asr, "_prepare_inputs", lambda *args: {
        "input_ids": torch.zeros(1, args[-1].size // 100, dtype=torch.long)})

    def generate(inputs, max_new_tokens):
        events.append("audio")
        graphs.add(("prefill", inputs["input_ids"].shape[1]))
        graphs.add(("decode", 1))
        assert len(graphs) <= 2
        return asr.ASRTranscription("", "", ())

    model._generate = generate
    try:
        for samples in (8000, 8000, 16000, 8000):
            model.transcribe(np.zeros(samples, dtype=np.float32))
        assert events == ["audio", "audio", "clear", "audio", "clear", "audio"]
    finally:
        model.close()
