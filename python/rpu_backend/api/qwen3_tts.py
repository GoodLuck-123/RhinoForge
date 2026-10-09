"""Bounded Qwen3-TTS Base speech inference with RPU autoregressive components."""
from __future__ import annotations

import operator
from pathlib import Path
import threading
import types

import numpy as np
import torch

from .errors import RPUUnsupportedDtypeError


def _validate_request(text, language, max_new_tokens):
    if not isinstance(text, str) or not text.strip() or len(text) > 128:
        raise ValueError("TTS text must contain 1–128 characters")
    if language not in ("Chinese", "English"):
        raise ValueError("TTS profile admits Chinese or English")
    if isinstance(max_new_tokens, bool):
        raise ValueError("max_new_tokens must be an integer in [2, 64]")
    try:
        count = operator.index(max_new_tokens)
    except TypeError as exc:
        raise ValueError("max_new_tokens must be an integer in [2, 64]") from exc
    if not 2 <= count <= 64:
        raise ValueError("max_new_tokens must be an integer in [2, 64]")
    return count


def _read_reference(path):
    import soundfile as sf

    path = Path(path)
    info = sf.info(path)
    if info.samplerate != 24000 or info.channels != 1 or not 12000 <= info.frames <= 480000:
        raise ValueError("TTS reference must be mono 24 kHz audio, 0.5–20 seconds")
    samples, rate = sf.read(path, dtype="float32")
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("TTS reference must contain finite mono samples")
    return samples, rate


class Qwen3TTSModel:
    """Frozen 0.6B Base, short non-streaming voice cloning, batch one.

    Talker, Code Predictor and their output heads execute FP16 on RPU.
    Embedding, speaker encoder, speech codec and composition use CPU FP32.
    Only speech output is exposed; internal per-layer traces are not an API.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._closed = True
        self._wrapper = None
        self._components = {}
        self._restore = []
        self._head_weights = []
        self._head_calls = [0] * 16
        self._retirement_error = None

    @classmethod
    def from_pretrained(cls, checkpoint, *, source, dtype=torch.float16):
        """Load local frozen weights and the explicitly prepared upstream source."""
        from rpu_backend.adapters.qwen3_tts import load_cpu_model
        from .causal_lm import _claim_live_instance

        if dtype is not torch.float16:
            raise RPUUnsupportedDtypeError("TTS RPU components require torch.float16; CPU components use float32")
        model = cls()
        _claim_live_instance(model)
        model._closed = False
        try:
            model._wrapper = load_cpu_model(checkpoint, source)
            model._install_rpu()
        except BaseException as error:
            try:
                model.close()
            except BaseException as cleanup_error:
                error.add_note(f"TTS cleanup failed: {cleanup_error!r}")
            raise
        return model

    def _install_rpu(self):
        from rpu_backend.graph import GraphCache, GraphSignature
        from rpu_backend.runtime.weights import transform_linear_weight
        from rpu_backend.runtime._native_retirement import _InstalledNativeResource

        talker = self._wrapper.model.talker
        for name, core in (("talker", talker.model), ("predictor", talker.code_predictor.model)):
            graph = GraphCache()
            handle = torch.ops.rpu.causal_decoder_create()
            part = {"handle": handle, "graph": graph, "calls": 0, "tokens": 0,
                    "capacity": 128 if name == "talker" else 32, "layers": len(core.layers),
                    "name": name, "generation_limit": 64}
            self._components[name] = part
            handle_name = "_native_" + name + "_handle"
            setattr(self, handle_name, handle)
            part["resource"] = _InstalledNativeResource(
                self, handle, torch.ops.rpu.causal_decoder_destroy, graphs=(graph,),
                keepalive=part, label="TTS " + name, handle_name=handle_name,
            )
            part["resource"].take_ownership(self)
            groups = []
            for path, partition in (("self_attn.q_proj", 1), ("self_attn.k_proj", 1),
                                    ("self_attn.v_proj", 1), ("self_attn.o_proj", 0)):
                groups.append([transform_linear_weight(layer.get_submodule(path).weight.detach().half(), partition, 8).to("rpu")
                               for layer in core.layers])
            for path in ("self_attn.q_norm", "self_attn.k_norm", "input_layernorm", "post_attention_layernorm"):
                groups.append([layer.get_submodule(path).weight.detach().half().to("rpu") for layer in core.layers])
            for path, partition in (("mlp.gate_proj", 1), ("mlp.up_proj", 1), ("mlp.down_proj", 0)):
                groups.append([transform_linear_weight(layer.get_submodule(path).weight.detach().half(), partition, 8).to("rpu")
                               for layer in core.layers])
            positions = torch.arange(part["capacity"]).view(1, -1)
            if name == "talker":
                positions = positions.unsqueeze(0).expand(3, -1, -1)
            with torch.inference_mode():
                cos, sin = core.rotary_emb(torch.zeros(1, 1, 1024, dtype=torch.float16), positions)
                if name == "talker":
                    if not all(torch.equal(value[0], value[index]) for value in (cos, sin) for index in (1, 2)):
                        raise ValueError("TTS requires coincident position axes")
                    cos, sin = cos[0], sin[0]
                cos = cos.squeeze(0)[:, :64].contiguous().to("rpu")
                sin = sin.squeeze(0)[:, :64].contiguous().to("rpu")
                norm = core.norm.weight.detach().half().to("rpu")
            part["storage"] = (groups, cos, sin, norm)
            torch.ops.rpu.causal_decoder_set_weights(handle, *groups, cos, sin, norm, 16, 8, 128, 1024, 3072, 1e-6, True)
            descriptor = list(torch.ops.rpu.causal_decoder_resolve_decode_stage_descriptor(handle))
            part["descriptor"] = descriptor
            part["signature"] = GraphSignature(op_id="qwen3_tts_" + name, shapes=[1, 1, 1024],
                dyn_dims=[handle, *descriptor], dtypes=[torch.float16])
            self._restore.append((core, core.forward))

            def forward(owner, inputs_embeds=None, past_key_values=None, attention_mask=None,
                        position_ids=None, cache_position=None, input_ids=None,
                        output_attentions=False, output_hidden_states=False, part=part, **kwargs):
                if input_ids is not None or output_attentions:
                    raise ValueError("TTS speech components require embeddings and no attention trace")
                return _decoder_forward(part, inputs_embeds, past_key_values, attention_mask,
                                        position_ids, cache_position, output_hidden_states)
            core.forward = types.MethodType(forward, core)

        calls = self._head_calls
        for index, head in enumerate([talker.codec_head, *talker.code_predictor.lm_head]):
            weight = transform_linear_weight(head.weight.detach().half(), 1, 8).to("rpu")
            self._head_weights.append(weight)
            self._restore.append((head, head.forward))

            def linear(owner, hidden, weight=weight, index=index):
                result = torch.ops.rpu.linear_with_accumulation(
                    hidden.reshape(-1, 1024).half().contiguous().to("rpu"), weight, None, 1, 8, False).cpu().float()
                if not torch.isfinite(result).all():
                    raise FloatingPointError("non-finite TTS RPU logits")
                calls[index] += 1
                return result.reshape(*hidden.shape[:-1], owner.out_features)
            head.forward = types.MethodType(linear, head)

    def synthesize(self, text, *, reference_audio, language="Chinese", max_new_tokens=64):
        """Return an independent float32 mono waveform and sample rate (24 kHz).

        The fixed token cap can truncate speech. Reference audio is a local
        mono 24 kHz file; this profile uses speaker embedding without a transcript.
        """
        limit = _validate_request(text, language, max_new_tokens)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("TTS requests, stats and close must be serial")
        try:
            if self._closed:
                raise RuntimeError("TTS model is closed")
            if self._retirement_error is not None:
                raise RuntimeError("TTS retirement failed; restart the process") from self._retirement_error
            if len(self._wrapper.processor.tokenizer(text)["input_ids"]) > 32:
                raise ValueError("TTS text exceeds 32 tokenizer tokens")
            reference = _read_reference(reference_audio)
            self._components["talker"]["generation_limit"] = limit
            with torch.inference_mode():
                waves, rate = self._wrapper.generate_voice_clone(text=text, language=language,
                    ref_audio=reference, x_vector_only_mode=True, non_streaming_mode=True,
                    do_sample=False, subtalker_dosample=False, max_new_tokens=limit)
            if len(waves) != 1 or rate != 24000:
                raise RuntimeError("TTS output shape or sample rate differs from the profile")
            waveform = np.asarray(waves[0], dtype=np.float32)
            if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
                raise FloatingPointError("TTS produced empty or non-finite audio")
            return waveform.copy(), rate
        finally:
            self._lock.release()

    def execution_stats(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("TTS requests, stats and close must be serial")
        try:
            if self._closed:
                raise RuntimeError("TTS model is closed")
            if self._retirement_error is not None:
                raise RuntimeError("TTS retirement failed; restart the process") from self._retirement_error
            result = {"head_calls": list(self._head_calls), "components": {}}
            for name, part in self._components.items():
                graph = part["graph"]
                result["components"][name] = {"calls": part["calls"], "tokens": part["tokens"],
                    "graph_size": graph.size(), "graph_replays": sum(e.replay_count for e in graph.snapshot()),
                    "graph_invariant_ok": graph.cache_invariant_ok()}
            return result
        finally:
            self._lock.release()

    def close(self):
        from .causal_lm import _release_live_instance

        if not self._lock.acquire(blocking=False):
            raise RuntimeError("TTS requests, stats and close must be serial")
        try:
            if self._closed:
                return
            if self._retirement_error is not None:
                raise RuntimeError("TTS retirement failed; restart the process") from self._retirement_error
            try:
                # Retire every captured queue before either component's weights.
                for part in self._components.values():
                    part["graph"].clear()
                for part in reversed(list(self._components.values())):
                    part["resource"].retire()
                    part["handle"] = None
            except BaseException as error:
                self._retirement_error = error
                # Retain the whole parent, including the other component and all
                # heads. Native cleanup is uncertain and must never be retried.
                for part in self._components.values():
                    part["resource"].retain_failure(error, self)
                raise
            for module, original in reversed(self._restore):
                module.forward = original
            self._restore.clear()
            self._components.clear()
            self._head_weights.clear()
            self._wrapper = None
            # Drop the last loop's storage before admitting a new resident model.
            part = None
            self._closed = True
            _release_live_instance(self)
        finally:
            self._lock.release()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("TTS model is closed")
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _decoder_forward(part, hidden, cache, attention_mask, position_ids, cache_position, output_hidden_states):
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from .cache import RPUCache

    if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != 1024:
        raise ValueError("TTS decoder requires [1, tokens, 1024] inputs")
    if hidden.device.type != "cpu" or hidden.dtype != torch.float32 or not torch.isfinite(hidden).all():
        raise ValueError("TTS decoder requires finite CPU FP32 inputs")
    if attention_mask is not None and not bool(attention_mask.all()):
        raise ValueError("TTS padding is outside this profile")
    if not isinstance(cache, RPUCache):
        if cache is not None and cache.get_seq_length() != 0:
            raise ValueError("cannot replace a populated CPU TTS cache")
        cache = RPUCache(part["layers"], 1, part["capacity"], 8, 128)
    count = hidden.shape[1]
    if count < 1 or cache.position + count > part["capacity"]:
        raise ValueError("TTS cache capacity exceeded")
    if part["name"] == "talker" and cache.position == 0 and count + part["generation_limit"] > part["capacity"]:
        raise ValueError("TTS prompt plus requested generation exceeds 128 positions")
    expected = torch.arange(cache.position, cache.position + count)
    if position_ids is not None:
        if position_ids.ndim == 3:
            if position_ids.shape[0] != 3 or not torch.equal(position_ids[0], position_ids[1]) or not torch.equal(position_ids[0], position_ids[2]):
                raise ValueError("TTS requires coincident position axes")
            position_ids = position_ids[0]
        if not torch.equal(position_ids.cpu().reshape(-1), expected.to(position_ids.dtype)):
            raise ValueError("TTS positions must match the contiguous cache range")
    if cache_position is not None and not torch.equal(cache_position.cpu().reshape(-1), expected):
        raise ValueError("TTS cache_position differs from the contiguous cache range")
    chunks = []
    # shortcut: short prompts use the proved decode plan one token at a time; profile longer prompts separately.
    for index in range(count):
        half_input = hidden[:, index:index + 1].half().contiguous()
        if not torch.isfinite(half_input).all():
            raise FloatingPointError("TTS input exceeds FP16 range")
        device_input = half_input.to("rpu")
        with part["graph"].capture(part["signature"]):
            raw = torch.ops.rpu.causal_decoder_forward(part["handle"], device_input,
                cache.k_caches, cache.v_caches, None, cache.position, True,
                planned_stage_descriptor=part["descriptor"])
        value = raw.cpu().float()
        if value.shape != (1, 1, 1024) or not torch.isfinite(value).all():
            raise FloatingPointError("non-finite or invalid TTS RPU hidden state")
        # CPU copies own their storage across the next component and Graph replay.
        chunks.append(value)
        cache.update_position(1)
    part["calls"] += 1
    part["tokens"] += count
    result = torch.cat(chunks, dim=1)
    # The upstream speech wrapper consumes only the final normalized hidden state.
    return BaseModelOutputWithPast(last_hidden_state=result, past_key_values=cache,
        hidden_states=(result,) if output_hidden_states else None)
