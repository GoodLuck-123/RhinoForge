"""Frozen Qwen3-ASR-0.6B: resident device audio blocks and text decoder, CPU glue.

Audio length and placeholder semantics follow QwenLM/Qwen3-ASR commit
7c6daf77a2421100f5fb066495372c00129d39ff (Apache-2.0).
"""
from __future__ import annotations

import hashlib
import json
import operator
import threading
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .errors import RPUUnsupportedDtypeError, UnsupportedModelError


_IDENTITIES = {
    "config.json": "76d3ae4601ce939830b2517f4a6cadb86cc51316c3900af6b020b051c21a478c",
    "preprocessor_config.json": "45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d",
    "tokenizer_config.json": "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c",
    "chat_template.json": "75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
}
_WEIGHT_SHA256 = "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea"
_WEIGHT_SIZE = 1876091704
_AUDIO_TOKEN = "<|audio_pad|>"
_AUDIO_TOKEN_ID = 151676
_CACHE_LENGTH = 512


@dataclass(frozen=True)
class ASRTranscription:
    """Independent host output; raw_text retains the model's language metadata."""

    text: str
    raw_text: str
    token_ids: tuple[int, ...]


def _audio_token_length(mel_frames: int) -> int:
    # Upstream chunks every 100 mel frames before three stride-2 convolutions.
    remainder = mel_frames % 100
    return (remainder + 7) // 8 + (mel_frames // 100) * 13


def _validate_audio(audio, sample_rate: int, max_new_tokens: int) -> np.ndarray:
    if isinstance(sample_rate, bool) or sample_rate != 16000:
        raise ValueError("Qwen3-ASR requires mono 16 kHz audio")
    if isinstance(max_new_tokens, bool):
        raise ValueError("max_new_tokens must be an integer in [1, 64]")
    try:
        count = operator.index(max_new_tokens)
    except TypeError as exc:
        raise ValueError("max_new_tokens must be an integer in [1, 64]") from exc
    if not 1 <= count <= 64:
        raise ValueError("max_new_tokens must be an integer in [1, 64]")
    if torch.is_tensor(audio):
        if audio.device.type != "cpu" or not audio.is_floating_point():
            raise ValueError("audio tensor must contain CPU floating-point samples")
        audio = audio.detach().float().numpy()
    waveform = np.asarray(audio)
    if waveform.ndim != 1 or waveform.dtype.kind != "f":
        raise ValueError("audio must be one mono floating-point waveform")
    if not 8000 <= waveform.size <= 128000 or not np.isfinite(waveform).all():
        raise ValueError("audio must be finite and 0.5–8 seconds long")
    waveform = np.asarray(waveform, dtype=np.float32)
    if not np.isfinite(waveform).all():
        raise ValueError("audio must remain finite in float32")
    return waveform


def _preflight(checkpoint: Path, dtype) -> dict:
    if dtype is not torch.float16:
        raise RPUUnsupportedDtypeError("Qwen3-ASR supports only torch.float16")
    from importlib.metadata import version

    if version("transformers") != "5.3.0":
        raise UnsupportedModelError("Qwen3-ASR requires the pinned transformers==5.3.0 runtime")
    if not checkpoint.is_dir():
        raise UnsupportedModelError("Qwen3-ASR requires a local frozen checkpoint directory")
    for name, expected in _IDENTITIES.items():
        path = checkpoint / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise UnsupportedModelError(f"Qwen3-ASR frozen checkpoint mismatch: {name}")
    weight = checkpoint / "model.safetensors"
    if not weight.is_file() or weight.stat().st_size != _WEIGHT_SIZE:
        raise UnsupportedModelError("Qwen3-ASR requires its original model.safetensors")
    if any((checkpoint / name).exists() for name in ("tokenizer.json", "special_tokens_map.json", "added_tokens.json")):
        raise UnsupportedModelError("Qwen3-ASR frozen profile uses the original tokenizer files")
    if any(not hasattr(torch.ops.rpu, name) for name in (
            "siglip_set_audio_mode", "siglip_set_audio_projection", "assemble_inputs_embeds")):
        raise UnsupportedModelError("Qwen3-ASR resident audio requires the rebuilt native backend")
    return json.loads((checkpoint / "config.json").read_text())["thinker_config"]


def _load_processor(checkpoint: Path):
    from transformers import Qwen2TokenizerFast, WhisperFeatureExtractor

    tokenizer = Qwen2TokenizerFast.from_pretrained(checkpoint, local_files_only=True)
    # Transformers 5.3 passes this argument twice if supplied to from_pretrained.
    tokenizer._patch_mistral_regex(
        tokenizer, str(checkpoint), fix_mistral_regex=True, local_files_only=True, is_local=True,
    )
    template = json.loads((checkpoint / "chat_template.json").read_text())["chat_template"]
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": ""},
         {"role": "user", "content": [{"type": "audio", "audio": ""}]}],
        chat_template=template, add_generation_prompt=True, tokenize=False,
    )
    if prompt.count(_AUDIO_TOKEN) != 1 or tokenizer.convert_tokens_to_ids(_AUDIO_TOKEN) != _AUDIO_TOKEN_ID:
        raise UnsupportedModelError("Qwen3-ASR audio placeholder contract differs")
    return tokenizer, WhisperFeatureExtractor.from_pretrained(checkpoint, local_files_only=True), prompt


def _prepare_inputs(tokenizer, feature_extractor, prompt: str, waveform: np.ndarray) -> dict:
    features = feature_extractor(
        [waveform], sampling_rate=16000, padding=True, truncation=False,
        return_attention_mask=True, return_tensors="pt",
    )
    lengths = features["attention_mask"].sum(-1)
    count = _audio_token_length(int(lengths.item()))
    tokens = tokenizer(
        [prompt.replace(_AUDIO_TOKEN, _AUDIO_TOKEN * count)],
        padding=True, padding_side="left", return_tensors="pt",
    )
    if tokens["input_ids"].shape[0] != 1 or not bool(tokens["attention_mask"].all()):
        raise ValueError("Qwen3-ASR requires one unpadded prompt")
    if int((tokens["input_ids"] == _AUDIO_TOKEN_ID).sum()) != count:
        raise ValueError("Qwen3-ASR audio placeholder count differs from encoder length")
    if tokens["input_ids"].shape[1] + 64 > _CACHE_LENGTH:
        raise ValueError("Qwen3-ASR prompt exceeds the cache envelope")
    return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"],
            "input_features": features["input_features"].to(torch.float16), "feature_lens": lengths}


def _load_components(checkpoint: Path, config: dict):
    from safetensors import safe_open
    from transformers import Qwen3Config, Qwen3Model
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

    audio_config = Qwen3OmniMoeAudioEncoderConfig(**config["audio_config"])
    audio_config._attn_implementation = "eager"
    values = dict(config["text_config"])
    values.pop("rope_scaling")
    values["rope_parameters"] = {"rope_type": "default", "rope_theta": values.pop("rope_theta")}
    text_config = Qwen3Config(**values)
    text_config._attn_implementation = "eager"
    with torch.device("meta"):
        audio = Qwen3OmniMoeAudioEncoder(audio_config)
        text = Qwen3Model(text_config)
    expected = {"thinker.audio_tower." + k: tuple(v.shape) for k, v in audio.state_dict().items()}
    expected.update({"thinker.model." + k: tuple(v.shape) for k, v in text.state_dict().items()})
    expected["thinker.lm_head.weight"] = (151936, 1024)
    weight_path = checkpoint / "model.safetensors"
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        # safetensors may omit either alias of the tied embedding/head.
        required = set(expected) - {"thinker.lm_head.weight", "thinker.model.embed_tokens.weight"}
        if not required <= keys or keys - set(expected) or not keys & (set(expected) - required):
            raise UnsupportedModelError("Qwen3-ASR weight keys differ from the frozen profile")
        for name in keys:
            view = handle.get_slice(name)
            if tuple(view.get_shape()) != expected[name] or view.get_dtype() not in ("BF16", "F32", "F16"):
                raise UnsupportedModelError(f"Qwen3-ASR weight metadata mismatch: {name}")
    with weight_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != _WEIGHT_SHA256:
            raise UnsupportedModelError("Qwen3-ASR weight identity differs from the frozen revision")
    # Construct CPU buffers normally: meta .to_empty would lose rotary/sinusoid values.
    audio = Qwen3OmniMoeAudioEncoder(audio_config).half().eval()
    text = Qwen3Model(text_config).eval()
    # from_pretrained(dtype=FP16) preserves the explicitly FP32 RoPE frequencies.
    # A blanket .half() would round these before cos/sin generation.
    inv_freq = text.rotary_emb.inv_freq.clone()
    text.half()
    text.rotary_emb.inv_freq = inv_freq
    text.rotary_emb.original_inv_freq = inv_freq.clone()
    states = {"audio": {}, "text": {}}
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        for name in keys:
            if name.startswith("thinker.audio_tower."):
                states["audio"][name.removeprefix("thinker.audio_tower.")] = handle.get_tensor(name)
            elif name.startswith("thinker.model."):
                states["text"][name.removeprefix("thinker.model.")] = handle.get_tensor(name)
        if "embed_tokens.weight" not in states["text"]:
            states["text"]["embed_tokens.weight"] = handle.get_tensor("thinker.lm_head.weight")
        audio.load_state_dict(states["audio"], strict=True)
        text.load_state_dict(states["text"], strict=True)
    return audio, text


class Qwen3ASRModel:
    """One resident device ASR model; requests are serial and outputs live on CPU.

    Use ``with Qwen3ASRModel.from_pretrained(local_checkpoint) as model:``.
    Only the frozen Qwen/Qwen3-ASR-0.6B FP16 profile is admitted. Audio must be
    mono float samples at 16 kHz, 0.5–8 seconds. No timestamps or streaming.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._closed = True
        self._audio = self._text = self._embedding_weight = None
        self._head_weight = None
        self._language_model = None
        self._audio_linear_calls = {}
        self._audio_encoder = None
        self._decoder_calls = 0
        self._restore_forwards = []
        self._graph_prompt_length = None
        self._cache = None
        self._retirement_error = None

    @classmethod
    def from_pretrained(cls, checkpoint, *, dtype=torch.float16) -> "Qwen3ASRModel":
        from .causal_lm import _claim_live_instance

        path = Path(checkpoint)
        config = _preflight(path, dtype)
        model = cls()
        _claim_live_instance(model)
        model._closed = False
        try:
            model._tokenizer, model._feature_extractor, model._prompt = _load_processor(path)
            model._audio, model._text = _load_components(path, config)
            torch.rpu.set_caching_allocator(True)
            model._install_rpu()
        except BaseException as error:
            try:
                model.close()
            except BaseException as cleanup_error:
                error.add_note(f"ASR cleanup failed: {cleanup_error!r}")
            raise
        return model

    def _install_rpu(self):
        from rpu_backend.adapters.qwen3 import lookup_causal_decoder, _apply_fused_lm_head_for_rpu
        from rpu_backend.runtime.decoder import _install_causal_decoder_forward
        from rpu_backend.runtime.topology import resolve_decoder_topology
        from rpu_backend.runtime.weights import convert_linear_weights_inplace, transform_linear_weight

        topology = resolve_decoder_topology(num_cores=8, hidden_size=1024, intermediate_size=3072,
            num_q_heads=16, num_kv_heads=8, head_dim=128, vocab_size=151936)
        with torch.inference_mode():
            cos, sin = self._text.rotary_emb(torch.zeros(1, 1, 1024, dtype=torch.float16),
                                            torch.arange(_CACHE_LENGTH).view(1, -1))
        tables = (cos[0, :, :64].contiguous(), sin[0, :, :64].contiguous())
        self._embedding_weight = self._text.embed_tokens.weight.detach()
        convert_linear_weights_inplace(self._text)
        self._text.to("rpu")
        if self._embedding_weight.device.type != "cpu" or self._embedding_weight.dtype != torch.float16:
            raise RuntimeError("ASR host embedding/head storage moved during device preparation")
        _install_causal_decoder_forward(self._text, arch="qwen3", topology=topology,
            cos_sin=tables, chunk_envelope_for=lookup_causal_decoder,
            execution_config={"model": {"num_cores": 8}, "prefill": {"linear_acc32": True}},
            _graph_cache_max_entries=2)
        self._head_weight = transform_linear_weight(self._embedding_weight, 1, 8).to("rpu")
        # Reuse the causal-LM output contract; the base decoder must still return hidden states.
        self._language_model = torch.nn.Module()
        self._language_model.model = self._text
        self._language_model.config = self._text.config
        head = torch.nn.Module()
        head.weight = torch.nn.Parameter(self._head_weight, requires_grad=False)
        head._rpu_linear_num_cores, head._rpu_linear_partition = 8, 1
        head.forward = self._project_logits
        self._language_model.lm_head = head
        _apply_fused_lm_head_for_rpu(self._language_model)
        counters = self._audio_linear_calls
        for name, module in self._audio.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self._restore_forwards.append((module, module.forward))
                weight = module.weight.detach().float().contiguous(memory_format=torch.channels_last)
                bias = module.bias.detach().float() if module.bias is not None else None

                def convolution(owner, hidden, weight=weight, bias=bias):
                    # Use CPU FP32 convolution with FP16 input/output boundaries.
                    hidden = hidden.float().contiguous(memory_format=torch.channels_last)
                    return torch.nn.functional.conv2d(hidden, weight, bias,
                        owner.stride, owner.padding, owner.dilation, owner.groups).half()

                module.forward = types.MethodType(convolution, module)
                continue
            if not isinstance(module, torch.nn.Linear):
                continue
            counters[name] = 0
            if name.startswith("layers.") or name in ("proj1", "proj2"):
                continue
            if module.in_features % 16 or module.out_features % 128:
                raise UnsupportedModelError(f"unaligned ASR audio Linear: {name}")
            weight = transform_linear_weight(module.weight.detach().contiguous(), 1, 8).to("rpu")
            bias = module.bias.detach().contiguous().to("rpu") if module.bias is not None else None
            counters[name] = 0
            self._restore_forwards.append((module, module.forward))

            def forward(owner, hidden, weight=weight, bias=bias, name=name):
                if hidden.device.type not in ("cpu", "rpu") or hidden.dtype != torch.float16:
                    raise ValueError("ASR audio Linear requires CPU or device FP16 input")
                result = torch.ops.rpu.linear_with_core_count(hidden.contiguous().to("rpu"), weight, bias, 1, 8)
                counters[name] += 1
                return result.cpu() if hidden.device.type == "cpu" else result

            module.forward = types.MethodType(forward, module)
        if len(counters) != 111:
            raise UnsupportedModelError("ASR audio projection count differs")
        from ._qwen3_asr_audio import ResidentAudioEncoder

        self._audio_encoder = ResidentAudioEncoder(self._audio, counters)
        self._audio_encoder.install(self)
        self._restore_forwards.append((self._audio, self._audio.forward))
        self._audio.forward = self._audio_encoder.forward

    def transcribe(self, audio, *, sample_rate=16000, max_new_tokens=64) -> ASRTranscription:
        """Transcribe one waveform; accepts CPU float Tensor or numpy samples."""
        waveform = _validate_audio(audio, sample_rate, max_new_tokens)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Qwen3-ASR requests and close must be serial")
        try:
            if self._closed:
                raise RuntimeError("Qwen3-ASR model is closed")
            if self._retirement_error is not None:
                raise RuntimeError("ASR retirement failed; restart the process") from self._retirement_error
            with torch.inference_mode():
                inputs = _prepare_inputs(self._tokenizer, self._feature_extractor, self._prompt, waveform)
                prompt_length = inputs["input_ids"].shape[1]
                if self._graph_prompt_length is not None and prompt_length != self._graph_prompt_length:
                    # Keep only the current shape's prefill/decode pair, leaving
                    # launch arena capacity for the next audio eager Linear.
                    self._text._rpu_decoder_graph_cache.clear()
                self._graph_prompt_length = prompt_length
                return self._generate(inputs, max_new_tokens)
        finally:
            self._lock.release()

    def _generate(self, inputs, max_new_tokens):
        from .cache import RPUCache

        if self._cache is None:
            self._cache = RPUCache(28, 1, _CACHE_LENGTH, 8, 128, dtype=torch.float16, attn_tp=8)
        cache = self._cache
        cache.reset_to_position(0)
        length = inputs["feature_lens"]
        features = self._audio(inputs["input_features"][0, :, :int(length.item())],
                               feature_lens=length).last_hidden_state
        ids = inputs["input_ids"]
        mask = ids == _AUDIO_TOKEN_ID
        if features.shape != (int(mask.sum()), 1024) or not torch.isfinite(features.cpu()).all():
            raise ValueError("ASR audio features violate the placeholder/finite contract")
        hidden = self._embed_tokens(ids, features)
        attention_mask = inputs["attention_mask"]
        generated = []
        for _ in range(max_new_tokens):
            start = cache.get_seq_length()
            positions = torch.arange(start, start + hidden.shape[1])
            output = self._language_model(inputs_embeds=hidden.contiguous().to("rpu"),
                position_ids=positions.unsqueeze(0), cache_position=positions,
                attention_mask=attention_mask, past_key_values=cache, use_cache=True,
                logits_to_keep=1, return_dict=True)
            self._decoder_calls += 1
            logits = output.logits.cpu()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite ASR logits")
            token = logits[:, -1].argmax(-1, keepdim=True)
            generated.append(int(token.item()))
            if generated[-1] in (151645, 151643):
                break
            hidden = self._embed_tokens(token)
            attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)), dim=1)
        raw = self._tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # Preserve raw output for diagnostics; expose the text after ASR metadata.
        text = raw.split("<asr_text>", 1)[-1].strip()
        return ASRTranscription(text=text, raw_text=raw, token_ids=tuple(generated))

    def _embed_tokens(self, ids, features=None):
        weight = self._text.embed_tokens.weight
        if (ids.device.type != "cpu" or ids.dtype != torch.long or ids.ndim != 2 or
                ids.shape[0] != 1 or ids.numel() == 0 or
                bool((ids < 0).any()) or bool((ids >= weight.shape[0]).any())):
            raise ValueError("ASR embedding requires one row of valid CPU token IDs")
        # Use CPU embedding lookup for single-token requests.
        if features is None:
            return torch.nn.functional.embedding(ids, self._embedding_weight).to("rpu")
        positions = (ids[0] == _AUDIO_TOKEN_ID).nonzero().flatten()
        if (positions.numel() == 0 or features.shape != (positions.numel(), 1024) or
                not torch.equal(positions, torch.arange(int(positions[0]), int(positions[0]) + positions.numel()))):
            raise ValueError("ASR requires one contiguous audio placeholder block")
        device_ids = ids.flatten().to(torch.int32).to("rpu")
        hidden = torch.ops.rpu.assemble_inputs_embeds(
            weight, device_ids, features.contiguous().to("rpu"), int(positions[0]))
        return hidden.unsqueeze(0)

    def _project_logits(self, hidden):
        return torch.ops.rpu.linear_with_accumulation(
            hidden.contiguous(), self._head_weight, None, 1, 8, True).cpu()

    def execution_stats(self) -> dict:
        """Return execution-route counters."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Qwen3-ASR requests, statistics and close must be serial")
        try:
            if self._closed:
                raise RuntimeError("Qwen3-ASR model is closed")
            if self._retirement_error is not None:
                raise RuntimeError("ASR retirement failed; restart the process") from self._retirement_error
            graph = self._text._rpu_decoder_graph_cache
            audio_graph = self._audio_encoder.graph
            return {"audio_linear_calls": dict(self._audio_linear_calls), "decoder_calls": self._decoder_calls,
                    "audio_resident_calls": self._audio_encoder.calls,
                    "audio_graph_size": audio_graph.size(),
                    "audio_graph_replays": sum(e.replay_count for e in audio_graph.snapshot()),
                    "audio_graph_invariant_ok": audio_graph.cache_invariant_ok(),
                    "graph_size": graph.size(), "graph_replays": sum(e.replay_count for e in graph.snapshot()),
                    "graph_invariant_ok": graph.cache_invariant_ok()}
        finally:
            self._lock.release()

    def close(self):
        """Retire the decoder before releasing device weights and process ownership."""
        from .causal_lm import _release_live_instance

        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Qwen3-ASR requests and close must be serial")
        try:
            if self._closed:
                return
            if self._retirement_error is not None:
                raise RuntimeError("ASR retirement failed; restart the process") from self._retirement_error
            resource = getattr(self._text, "_rpu_decoder_retirement_state", None)
            audio_resource = getattr(self._audio_encoder, "_resource", None)
            try:
                if resource is not None:
                    resource.retire()
                if self._audio_encoder is not None:
                    self._audio_encoder.close()
            except BaseException as error:
                self._retirement_error = error
                for installed in (resource, audio_resource):
                    if installed is not None:
                        installed.retain_failure(error, self)
                raise
            self._audio_encoder = None
            for module, forward in reversed(self._restore_forwards):
                module.forward = forward
            self._restore_forwards.clear()
            if self._text is not None:
                vars(self._text).pop("forward", None)
            self._audio = self._text = self._embedding_weight = None
            self._head_weight = None
            self._language_model = None
            self._cache = None
            self._closed = True
            _release_live_instance(self)
        finally:
            self._lock.release()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Qwen3-ASR model is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
