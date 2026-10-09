"""Board-free rejection and lifetime checks for the bounded speech API."""
from types import SimpleNamespace
import wave

import numpy as np
import pytest
import torch

from rpu_backend.api import Qwen3TTSModel
from rpu_backend.api import qwen3_tts as tts
from rpu_backend.api.errors import RPUUnsupportedDtypeError


@pytest.mark.parametrize("text,language,limit", [
    ("", "English", 32), ("  ", "Chinese", 32), ("x" * 129, "English", 32),
    ("hello", "French", 32), ("hello", "English", 1), ("hello", "English", 65),
    ("hello", "English", True), ("hello", "English", 1.5),
])
def test_request_rejection(text, language, limit):
    with pytest.raises(ValueError):
        tts._validate_request(text, language, limit)


def test_supported_request_boundaries():
    assert tts._validate_request("hello", "English", 2) == 2
    assert tts._validate_request("你好。", "Chinese", 64) == 64


@pytest.mark.parametrize("rate,channels,frames", [(16000, 1, 16000), (24000, 2, 24000), (24000, 1, 11999), (24000, 1, 480001)])
def test_reference_rejected_before_decoding(monkeypatch, rate, channels, frames):
    import soundfile

    monkeypatch.setattr(soundfile, "info", lambda path: SimpleNamespace(samplerate=rate, channels=channels, frames=frames))
    monkeypatch.setattr(soundfile, "read", lambda *a, **kw: pytest.fail("unsupported audio decoded"))
    with pytest.raises(ValueError, match="reference"):
        tts._read_reference("local.wav")


def test_reference_is_real_float32_audio(tmp_path):
    path = tmp_path / "ref.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(np.zeros(12000, dtype="<i2").tobytes())
    samples, rate = tts._read_reference(path)
    assert rate == 24000 and samples.shape == (12000,) and samples.dtype == np.float32


def test_dtype_rejected_before_loading(monkeypatch):
    from rpu_backend.adapters import qwen3_tts

    monkeypatch.setattr(qwen3_tts, "load_cpu_model", lambda *a: pytest.fail("weights loaded"))
    with pytest.raises(RPUUnsupportedDtypeError):
        Qwen3TTSModel.from_pretrained("missing", source="missing", dtype=torch.bfloat16)


def test_failed_load_releases_owner(monkeypatch):
    from rpu_backend.adapters import qwen3_tts
    from rpu_backend.api import causal_lm

    calls = []
    monkeypatch.setattr(causal_lm, "_claim_live_instance", lambda model: calls.append("claim"))
    monkeypatch.setattr(causal_lm, "_release_live_instance", lambda model: calls.append("release"))
    def fail(*args):
        calls.append("load")
        raise ValueError("wrong identity")
    monkeypatch.setattr(qwen3_tts, "load_cpu_model", fail)
    with pytest.raises(ValueError, match="identity"):
        Qwen3TTSModel.from_pretrained("missing", source="missing")
    assert calls == ["claim", "load", "release"]


def test_close_and_stats_are_serial():
    model = Qwen3TTSModel()
    model._lock.acquire()
    try:
        for method in (model.close, model.execution_stats):
            with pytest.raises(RuntimeError, match="serial"):
                method()
    finally:
        model._lock.release()
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.synthesize("你好", reference_audio="missing")


def test_decoder_rejects_bad_shapes_and_nonfinite_without_device_work():
    for hidden in (torch.zeros(2, 1, 1024), torch.zeros(1, 1, 1000), torch.full((1, 1, 1024), float("nan"))):
        with pytest.raises(ValueError):
            tts._decoder_forward({}, hidden, None, None, None, None, False)


def test_graph_creation_failure_does_not_create_native_handle(monkeypatch):
    from rpu_backend import graph

    model = Qwen3TTSModel()
    core = SimpleNamespace(layers=[])
    model._wrapper = SimpleNamespace(model=SimpleNamespace(
        talker=SimpleNamespace(model=core, code_predictor=SimpleNamespace(model=core))))

    def fail_graph():
        raise RuntimeError("graph allocation failed")

    monkeypatch.setattr(graph, "GraphCache", fail_graph)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_create", lambda: pytest.fail("untracked handle created"), raising=False)
    with pytest.raises(RuntimeError, match="graph allocation failed"):
        model._install_rpu()
    assert model._components == {}


@pytest.mark.parametrize("failure", ["graph", "destroy"])
def test_retirement_failure_retains_parent_and_blocks_new_owner_after_gc(monkeypatch, failure):
    import gc
    import weakref
    from rpu_backend.api import causal_lm, _execution
    from rpu_backend.runtime import _native_retirement

    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setattr(_native_retirement, "_FAILED_RETIREMENTS", [])
    events = []
    model = Qwen3TTSModel()
    model._closed = False
    causal_lm._claim_live_instance(model)
    for name, handle in (("talker", 1), ("predictor", 2)):
        def clear(name=name):
            events.append(("clear", name))
            if failure == "graph":
                raise RuntimeError("graph retirement failed")

        def destroy(value):
            events.append(("destroy", value))
            if failure == "destroy":
                raise RuntimeError("native retirement failed")

        cache = SimpleNamespace(clear=clear, cache_invariant_ok=lambda: True)
        part = {"handle": handle, "graph": cache}
        handle_name = "_native_" + name + "_handle"
        setattr(model, handle_name, handle)
        resource = _native_retirement._InstalledNativeResource(
            model, handle, destroy, graphs=(cache,), keepalive=part,
            label="test TTS", handle_name=handle_name)
        part["resource"] = resource
        model._components[name] = part
        resource.take_ownership(model)
        assert not resource.finalizer.alive
    with pytest.raises(RuntimeError, match="retirement failed"):
        model.close()
    previous = list(events)
    with pytest.raises(RuntimeError, match="restart"):
        model.close()
    with pytest.raises(RuntimeError, match="restart"):
        model.synthesize("hello", language="English", reference_audio="unused")
    assert events == previous  # No retry after uncertain Graph/native cleanup.
    if failure == "destroy":
        assert events[:2] == [("clear", "talker"), ("clear", "predictor")]
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is not None  # Quarantine retains both components and heads.
    with pytest.raises(RuntimeError, match="unsafe"):
        causal_lm._claim_live_instance(Qwen3TTSModel())
    assert len(_native_retirement._FAILED_RETIREMENTS) == 2


def test_install_failure_preserves_original_error_when_cleanup_also_fails(monkeypatch):
    from rpu_backend.adapters import qwen3_tts
    from rpu_backend.api import causal_lm

    monkeypatch.setattr(causal_lm, "_claim_live_instance", lambda owner: None)
    monkeypatch.setattr(qwen3_tts, "load_cpu_model", lambda *args: object())
    def install(self):
        raise ValueError("original install failure")
    def close(self):
        raise RuntimeError("cleanup failure")
    monkeypatch.setattr(Qwen3TTSModel, "_install_rpu", install)
    monkeypatch.setattr(Qwen3TTSModel, "close", close)
    with pytest.raises(ValueError, match="original install failure") as captured:
        Qwen3TTSModel.from_pretrained("unused", source="unused")
    assert "cleanup failure" in captured.value.__notes__[0]


@pytest.mark.parametrize("token_count", [32, 33])
def test_text_token_limit_precedes_reference_work_and_returns_owned_output(monkeypatch, token_count):
    source_wave = np.zeros(24, dtype=np.float32)
    calls = []
    def reference(path):
        calls.append("reference")
        return np.zeros(12000, dtype=np.float32), 24000
    def generate(**kwargs):
        calls.append("generate")
        return [source_wave], 24000
    model = Qwen3TTSModel()
    model._closed = False
    model._components = {"talker": {}}
    model._wrapper = SimpleNamespace(
        processor=SimpleNamespace(tokenizer=lambda text: {"input_ids": list(range(token_count))}),
        generate_voice_clone=generate)
    monkeypatch.setattr(tts, "_read_reference", reference)
    try:
        if token_count == 33:
            with pytest.raises(ValueError, match="32 tokenizer tokens"):
                model.synthesize("hello", reference_audio="unused")
            assert calls == []
        else:
            result, rate = model.synthesize("hello", reference_audio="unused")
            source_wave[:] = 1
            assert calls == ["reference", "generate"]
            assert rate == 24000 and np.array_equal(result, np.zeros(24))
    finally:
        # This host fixture has no claimed owner or native resources.
        model._closed = True
