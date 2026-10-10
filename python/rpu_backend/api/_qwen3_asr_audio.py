"""Bounded ASR audio blocks on the shared fused bidirectional encoder."""
import torch
import torch.nn.functional as F

from .cache import RPUCache
from .errors import UnsupportedModelError
from rpu_backend.runtime.weights import transform_linear_weight


def _pad_attention(weight, bias, *, output=False):
    # Pad heads, not the residual width: LayerNorm must still see 896 channels.
    if output:
        return F.pad(weight, (0, 128)), bias
    return F.pad(weight, (0, 0, 0, 128)), F.pad(bias, (0, 128))


class ResidentAudioEncoder:
    def __init__(self, audio, counters):
        self.audio = audio
        self.counters = counters
        self.handle = None
        self.graph = None
        self.cache = None
        self.last_length = None
        self.calls = 0
        self.positions = None

    def install(self):
        a = self.audio
        c = a.config
        if (c.d_model, c.encoder_layers, c.encoder_attention_heads, c.encoder_ffn_dim,
                c.activation_function, c.n_window, c.n_window_infer, c.output_dim) != (
                896, 18, 14, 3584, "gelu", 50, 800, 1024):
            raise UnsupportedModelError("resident ASR audio geometry differs")
        for layer in a.layers:
            if layer.self_attn_layer_norm.eps != 1e-5 or layer.final_layer_norm.eps != 1e-5:
                raise UnsupportedModelError("resident ASR LayerNorm epsilon differs")
        if a.ln_post.eps != 1e-5:
            raise UnsupportedModelError("resident ASR final LayerNorm epsilon differs")
        self.handle = torch.ops.rpu.siglip_create(True)
        columns = [[] for _ in range(16)]
        for layer in a.layers:
            attn = layer.self_attn
            modules = [attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj, layer.fc1, layer.fc2]
            biases = []
            for i, module in enumerate(modules):
                weight, bias = module.weight.detach(), module.bias.detach()
                if i < 4:
                    weight, bias = _pad_attention(weight, bias, output=i == 3)
                partition = 0 if i in (3, 5) else 1
                columns[i].append(transform_linear_weight(weight.contiguous(), partition, 8).to("rpu"))
                biases.append(bias.contiguous().to("rpu"))
            for i, tensor in enumerate([layer.self_attn_layer_norm.weight, layer.self_attn_layer_norm.bias,
                                       layer.final_layer_norm.weight, layer.final_layer_norm.bias]):
                columns[6 + i].append(tensor.detach().to("rpu"))
            for i, bias in enumerate(biases):
                columns[10 + i].append(bias)
        torch.ops.rpu.siglip_set_weights(self.handle, *columns,
            a.ln_post.weight.detach().to("rpu"), a.ln_post.bias.detach().to("rpu"),
            transform_linear_weight(a.proj1.weight.detach(), 1, 8).to("rpu"),
            a.proj1.bias.detach().to("rpu"), 16, 64, 896, 3584, 896, 1e-5)
        torch.ops.rpu.siglip_set_audio_mode(self.handle)
        torch.ops.rpu.siglip_set_audio_projection(self.handle,
            transform_linear_weight(a.proj2.weight.detach(), 1, 8).to("rpu"),
            a.proj2.bias.detach().to("rpu"))
        self.cache = RPUCache(18, 1, 128, 16, 64, dtype=torch.float16, attn_tp=8)
        self.graph = torch.rpu.GraphCache(max_entries=1)
        self.positions = a.positional_embedding.positional_embedding[:13].unsqueeze(0).expand(8, -1, -1).contiguous().to("rpu")

    def encode(self, hidden):
        from rpu_backend.adapters.siglip import _siglip_component_execution_plan

        length = hidden.shape[0]
        if hidden.shape != (length, 896) or not 7 <= length <= 104:
            raise ValueError("resident ASR requires 7–104 audio tokens of width 896")
        if self.last_length != length:
            self.graph.clear()
        self.last_length = length
        plan = _siglip_component_execution_plan(self.audio, self.handle, length, 1,
            external_patch_prologue=False, graph_cache=self.graph)
        descriptor = list(plan.selected.stage_tuple.physical_descriptor)
        signature = torch.rpu.GraphSignature(op_id="qwen3_asr_audio", shapes=[length, 896],
            dyn_dims=[self.handle, *plan.graph_key_words()], dtypes=[torch.float16])
        inputs = hidden.unsqueeze(0).contiguous().to("rpu")
        try:
            with self.graph.capture(signature):
                result = torch.ops.rpu.siglip_forward(self.handle, inputs,
                    list(self.cache.k_caches), list(self.cache.v_caches), descriptor)
            result = result[0]
        finally:
            torch.ops.rpu.spm_alloc_reset_temporary()
        for name in self.counters:
            if name.startswith("layers.") or name in ("proj1", "proj2"):
                self.counters[name] += 1
        self.calls += 1
        return result

    def forward(self, input_features, feature_lens=None, aftercnn_lens=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        a = self.audio
        if (input_features.device.type != "cpu" or input_features.dtype != torch.float16 or
                input_features.ndim != 2 or input_features.shape[0] != 128 or
                not 50 <= input_features.shape[1] <= 800 or feature_lens is None or
                feature_lens.numel() != 1 or int(feature_lens.item()) != input_features.shape[1]):
            raise ValueError("resident ASR requires one CPU FP16 mel input with 50–800 frames")
        # shortcut: one <=8-second utterance fits a single 800-frame attention window.
        chunks = list(input_features.T.split(100))
        padded = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True).transpose(1, 2).unsqueeze(1)
        hidden = F.gelu(a.conv2d1(padded))
        hidden = F.gelu(a.conv2d2(hidden))
        hidden = F.gelu(a.conv2d3(hidden))
        b, c, f, t = hidden.shape
        hidden = a.conv_out(hidden.permute(0, 3, 1, 2).contiguous().view(b, t, c * f).to("rpu"))
        hidden = hidden + self.positions[:b, :t]
        # Only the final 100-frame chunk has padding, so valid rows form one prefix.
        tokens = sum((chunk.shape[0] + 7) // 8 for chunk in chunks)
        hidden = hidden.reshape(-1, 896)[:tokens]
        hidden = self.encode(hidden)
        return BaseModelOutputWithPooling(last_hidden_state=hidden)

    def close(self):
        if self.graph is not None:
            self.graph.clear()
        if self.handle is not None:
            torch.ops.rpu.siglip_destroy(self.handle)
            self.handle = None
        self.cache = self.graph = None
        self.positions = None
