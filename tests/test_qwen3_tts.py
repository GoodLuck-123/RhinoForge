"""Board-free rejection and lifetime checks for the bounded speech API."""
from types import SimpleNamespace
import wave

import numpy as np
import pytest
import torch

from rpu_backend.api import Qwen3TTSModel
from rpu_backend.api import qwen3_tts as tts
from rpu_backend.api.errors import RPUUnsupportedDtypeError


def test_fixed_predictor_greedy_steps_and_cache():
    class Predictor:
        generation_config = SimpleNamespace(eos_token_id=None, repetition_penalty=None)

        def __init__(self):
            self.calls = 0

        def __call__(self, **kwargs):
            index = self.calls
            assert kwargs["generation_steps"] == index
            assert kwargs["past_key_values"] == (None if index == 0 else index)
            if index == 0:
                assert kwargs["inputs_embeds"].shape == (1, 2, 1024)
                assert kwargs["input_ids"] is None
            else:
                assert kwargs["inputs_embeds"] is None
                assert kwargs["input_ids"].item() == index - 1
            self.calls += 1
            logits = torch.zeros(1, 1, 16)
            logits[0, 0, index] = 1
            return SimpleNamespace(logits=logits, past_key_values=self.calls)

    owner = Predictor()
    result = tts._greedy_predictor_generate(owner, torch.zeros(1, 2, 1024),
        max_new_tokens=15, do_sample=False)
    assert result.sequences.tolist() == [list(range(15))]
    for count, sampling in ((14, False), (15, True)):
        with pytest.raises(ValueError, match="15-step"):
            tts._greedy_predictor_generate(owner, torch.zeros(1, 2, 1024),
                max_new_tokens=count, do_sample=sampling)
    assert owner.calls == 15


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


def test_close_restores_predictor_generation(monkeypatch):
    from rpu_backend.api import causal_lm

    monkeypatch.setattr(causal_lm, "_release_live_instance", lambda owner: None)
    original = object()
    predictor = SimpleNamespace(generate=object())
    model = Qwen3TTSModel()
    model._closed = False
    model._restore_generate = (predictor, original)
    model.close()
    assert predictor.generate is original and model._restore_generate is None


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
    def speaker(**kwargs):
        calls.append("speaker")
        return torch.zeros(1024)
    model = Qwen3TTSModel()
    model._closed = False
    model._components = {"talker": {}}
    model._wrapper = SimpleNamespace(
        model=SimpleNamespace(speaker_encoder_sample_rate=24000, extract_speaker_embedding=speaker),
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
            assert calls == ["reference", "speaker", "generate"]
            assert rate == 24000 and np.array_equal(result, np.zeros(24))
    finally:
        # This host fixture has no claimed owner or native resources.
        model._closed = True


def test_speaker_cache_tracks_audio_content_and_never_runs_reference_codec(monkeypatch):
    wave = np.zeros(12000, dtype=np.float32)
    embeddings = []
    prompts = []
    def speaker(**kwargs):
        value = torch.tensor([float(kwargs["audio"].sum())])
        embeddings.append(value)
        return value
    def generate(**kwargs):
        assert "ref_audio" not in kwargs
        prompt = kwargs["voice_clone_prompt"]
        assert prompt["ref_code"] == [None] and prompt["icl_mode"] == [False]
        assert prompt["x_vector_only_mode"] == [True]
        prompts.append(prompt)
        return [np.zeros(24, dtype=np.float32)], 24000
    model = Qwen3TTSModel()
    model._closed = False
    model._components = {"talker": {}}
    model._wrapper = SimpleNamespace(
        model=SimpleNamespace(speaker_encoder_sample_rate=24000, extract_speaker_embedding=speaker),
        processor=SimpleNamespace(tokenizer=lambda text: {"input_ids": [1]}),
        generate_voice_clone=generate)
    monkeypatch.setattr(tts, "_read_reference", lambda path: (wave, 24000))
    try:
        model.synthesize("first", reference_audio="same.wav")
        model.synthesize("second", reference_audio="same.wav")
        wave[0] = 1
        model.synthesize("third", reference_audio="same.wav")
        assert len(embeddings) == 2
        assert prompts[0] is prompts[1] and prompts[2] is not prompts[1]
        assert prompts[2]["ref_spk_embedding"][0].item() == 1
    finally:
        model._closed = True


def test_predictor_reuses_kv_with_fresh_positions_inputs_and_owned_outputs(monkeypatch):
    from contextlib import nullcontext
    from rpu_backend.api import cache as cache_module

    allocations, positions = [], []
    class Cache:
        def __init__(self, *args):
            self.position = 0
            self.k_caches = self.v_caches = []
            allocations.append(self)
        def reset_to_position(self, pos):
            self.position = pos
        def update_position(self, count):
            self.position += count
    monkeypatch.setattr(cache_module, "RPUCache", Cache)
    original_to = torch.Tensor.to
    def to(tensor, *args, **kwargs):
        if args == ("rpu",):
            return tensor
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", to)
    output = torch.empty(1, 1, 1024, dtype=torch.float16)
    def forward(handle, hidden, k, v, mask, position, causal, **kwargs):
        positions.append(position)
        return output.copy_(hidden)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_forward", forward, raising=False)
    part = dict(layers=5, capacity=32, name="predictor", handle=1, descriptor=[],
                signature=None, graph=SimpleNamespace(capture=lambda sig: nullcontext()), calls=0, tokens=0)
    first = tts._decoder_forward(part, torch.ones(1, 2, 1024), None, None, None, None, False)
    second = tts._decoder_forward(part, torch.zeros(1, 1, 1024), None, None, None, None, False)
    assert len(allocations) == 1 and positions == [0, 1, 0]
    assert first.last_hidden_state.eq(1).all() and second.last_hidden_state.eq(0).all()


def test_head_reuses_only_current_single_token_device_output():
    hidden = torch.zeros(1, 1, 1024)
    device = object()
    part = {"head_hidden": hidden, "head_device": device}
    assert tts._take_head_input(part, hidden.clone()) is None
    assert part == {}
    part.update(head_hidden=hidden, head_device=device)
    assert tts._take_head_input(part, hidden) is device
    assert tts._take_head_input(part, hidden) is None
    prompt = torch.zeros(1, 2, 1024)
    part.update(head_hidden=prompt, head_device=device)
    assert tts._take_head_input(part, prompt) is None
    assert part == {}


def test_talker_prefill_padding_positions_shape_switch_and_owned_outputs(monkeypatch):
    from contextlib import nullcontext
    from rpu_backend.api import cache as cache_module
    from rpu_backend.runtime import decoder

    class Cache:
        def __init__(self, *args):
            self.position = 0
            self.k_caches = self.v_caches = []
        def reset_to_position(self, position):
            self.position = position
        def update_position(self, count):
            self.position += count
    monkeypatch.setattr(cache_module, "RPUCache", Cache)
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, *args, **kwargs:
        tensor if args == ("rpu",) else original_to(tensor, *args, **kwargs))
    captures, evicted, calls = [], [], []
    def capture(signature):
        captures.append(signature)
        return nullcontext()
    graph = SimpleNamespace(capture=capture, evict=evicted.append,
                            clear=lambda: pytest.fail("decode Graph must be retained"))
    plan = SimpleNamespace(selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=(7,))),
                           graph_key_words=lambda: (7,))
    monkeypatch.setattr(decoder, "_text_prefill_execution_plan", lambda *args: (16, 16, plan))
    buffer = torch.empty(1, 16, 1024, dtype=torch.float16)
    def forward(handle, hidden, k, v, mask, position, causal, **kwargs):
        calls.append((position, hidden.clone(), kwargs["planned_stage_descriptor"]))
        return buffer[:, :hidden.shape[1]].copy_(hidden)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_forward", forward, raising=False)
    part = dict(layers=28, capacity=128, generation_limit=64, name="talker", handle=1,
                descriptor=(1,), signature=None, graph=graph, planner=object(), calls=0, tokens=0)
    first = tts._decoder_forward(part, torch.ones(1, 3, 1024), None, None, None, None, True)
    cache = first.past_key_values
    assert cache.position == 3 and first.last_hidden_state.shape == (1, 3, 1024)
    assert calls[0][1][:, 3:].eq(0).all() and calls[0][2] == (7,)
    second = tts._decoder_forward(part, torch.full((1, 1, 1024), 9.), cache, None, None, None, False)
    assert cache.position == 4 and calls[1][0] == 3 and calls[1][2] == (1,)
    assert first.last_hidden_state.eq(1).all() and second.last_hidden_state.eq(9).all()
    for _ in range(2):
        third = tts._decoder_forward(part, torch.zeros(1, 5, 1024), None, None, None, None, False)
        assert third.past_key_values is cache and cache.position == 5
    assert evicted == [captures[0]] and calls[2][0] == calls[3][0] == 0
    assert captures[0] != captures[2] and captures[2] == captures[3]


@pytest.mark.parametrize("kernel,stride", [(2, 2), (16, 8)])
def test_codec_convolution_paths_match_fp32(kernel, stride):
    from rpu_backend.adapters.qwen3_tts.codec import install_codec_convolutions

    torch.manual_seed(12)
    module = torch.nn.ConvTranspose1d(8, 8, kernel, stride=stride).eval()
    inputs = [torch.randn(1, 8, n) for n in (70, 91)]
    with torch.inference_mode():
        expected = [module(x) for x in inputs]
        restore = []
        original = module.forward
        install_codec_convolutions(module, restore)
        for x, reference in zip(inputs, expected):
            result = module(x)
            assert result.dtype == torch.float32 and torch.isfinite(result).all()
            torch.testing.assert_close(result, reference, rtol=1e-5, atol=2e-6)
        assert restore == [(module, original)]
        module.forward = restore[0][1]
        assert torch.equal(module(inputs[0]), expected[0])


def test_codec_geometry_rejected_before_any_patch():
    from rpu_backend.adapters.qwen3_tts.codec import install_codec_convolutions

    modules = torch.nn.Sequential(torch.nn.ConvTranspose1d(4, 4, 2, stride=2), torch.nn.ConvTranspose1d(4, 4, 3, padding=1))
    originals = [m.forward for m in modules]
    restore = []
    with pytest.raises(ValueError, match="geometry"):
        install_codec_convolutions(modules, restore)
    assert restore == [] and [m.forward for m in modules] == originals
