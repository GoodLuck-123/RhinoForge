# API 参考

简体中文 | [English](api_reference.md)

发行包名为 `rhinoforge`，Python package 仍是 `rpu_backend`。导入后把 PyTorch
`PrivateUse1` 设备注册为 `rpu`，并创建 `torch.rpu` namespace。

支持范围由 checkpoint、配置和执行入口共同决定，见[示例](model_support.zh.md)。

## 顶层 package

`rpu_backend.__all__` 只包含四个名称：

| 名称 | 契约 |
|---|---|
| `__version__` | 已安装 RhinoForge distribution 版本 |
| `RPUCache` | fused causal attention 的 FP16 KV cache |
| `RPUModelForCausalLM` | 已注册 decoder-only CausalLM 配置 loader |
| `reset_graph_cache()` | 清理进程 Graph cache；无 native backend 时为 no-op |

API compatibility level 位于 `rpu_backend.api.API_VERSION`，当前为 `5.0.0`，与
distribution version 独立。

## Causal language model

```python
import torch
from rpu_backend import RPUCache, RPUModelForCausalLM

model = RPUModelForCausalLM.from_pretrained(
    checkpoint,
    dtype=torch.float16,
    device="rpu",
    rpu_execution={
        "prefill": {
            "chunk_size": "auto",
            "padding_rows": "auto",
            "padding_budget": 64,
        }
    },
)
cache = RPUCache.from_model(model, input_ids=input_ids, max_new_tokens=32)
```

### `RPUModelForCausalLM.from_pretrained`

```python
RPUModelForCausalLM.from_pretrained(
    hf_repo_or_path,
    *,
    dtype=torch.float16,
    device=None,
    rpu_execution=None,
    **hf_kwargs,
)
```

Loader 在完整权重加载前读取 Hugging Face 配置，按 `config.architectures[0]` 选择
adapter，并提前拒绝 unsupported profile。当前入口接收 built-in Qwen3、Llama CausalLM
architecture，只接受 `torch.float16`；量化 checkpoint 自带受检查 metadata。

`rpu_execution` 是 read-only per-handle mapping。CausalLM 的 `prefill` 接受：

- `chunk_size`：`"auto"` 或正的 16 倍数；
- `padding_rows`：`"auto"` 或非负准确行数；
- `padding_budget`：非负整数，与精确 `padding_rows` 互斥。

Adapter 可以按配置进一步缩窄。Execution setting 是冷配置：设备安装后改变它，需要
创建新模型。

通用 CausalLM、image-text loader 和 `Pi05Policy.from_pretrained` 只支持 built-in model
code；`trust_remote_code` 必须为 `False`，custom Hugging Face code 在配置/权重加载前拒绝。

### `RPUCache`

优先使用 factory：

```python
RPUCache.from_model(
    model,
    *,
    input_ids=None,
    max_new_tokens=...,
    max_seq_len=None,
    batch_size=1,
    device="rpu",
)
```

必须且只能指定一种 sizing：`input_ids + max_new_tokens`，或正的 `max_seq_len`。
Cache 提供 `reset()`、`reset_to_position(pos)`、`get_seq_length()`、
`get_max_length()`、`get_cache(layer_idx)` 和 `to_dynamic_cache(device="cpu")`。
Batch>1 只限明确允许的 Qwen3 路径，并要求各行真实 token 长度相等。

`rpu_backend.api.Qwen3_5Cache.from_config(text_config, max_seq_len, ...)` 增加
Qwen3.5 text 的 recurrent/convolution state。它可完整 reset 到 position 0，但无法通过
rewind 重建非零 recurrent state。

## Image-text model

`rpu_backend.api.RPUModelForConditionalGeneration` 是 built-in Qwen3-VL 路径的
Hugging Face 风格 loader：

```python
from rpu_backend.api import RPUModelForConditionalGeneration

model = RPUModelForConditionalGeneration.from_pretrained(
    checkpoint,
    dtype=torch.float16,
    device="rpu",
    rpu_execution={
        "prefill": {"chunk_size": "auto"},
        "vision": {"chunk_size": "auto"},
    },
)
```

它与 CausalLM loader 一样 CPU-first、profile fail-fast。`vision` 接受 `chunk_size`，
`prefill` 接受前述字段。Class 存在不自动扩展视频或
[示例](model_support.zh.md)之外的配置。Qwen3.5 text/vision 与 Gemma4 使用各自
adapter。

### Qwen3.5 27B

官方 27B 提供 4/6/8 核 W8 Text/VL 和 8 核 W4 group32。root6 的 MLP 使用 6 核，attention/GDN/head/Vision 使用 4 核。启用既有量化 opt-in；图像输入另需 Vision opt-in。安装使用缓存分配器、四个冷 Graph arena、CPU 权重最终布局上传及物理 KV24；prefill 为有界单次 Graph，decode 保留独立 Graph。暂不提供 FP16 运行入口。详见[配置与权重](../examples/configs/qwen3_5/README.md)。

## Policy API

`rpu_backend.api` 导出：

| Class | 主要构造与推理方法 |
|---|---|
| `Pi05Policy` | `from_pretrained(...)` / `from_lerobot_policy(...)`；`to("rpu")`；`prepare_graphs(...)`；`predict_action_chunk(...)`；`select_action(...)` |
| `RhinoVLAPolicy` | `from_runtime(...)` / `from_factory(...)` / `from_pretrained(..., runtime_factory=...)`；`prepare_graphs(...)`；`predict(...)`；`predict_action_chunk(...)` |
| `WallOssPolicy` | `from_checkpoint(...)` / `from_pretrained(...)`；`to("rpu")`；`prepare_graphs(...)`；`infer(...)`；`predict_action_chunk(...)` |
| `Lingbot2Policy` | `from_checkpoint(...)`；`to("rpu")`；`prepare_graphs(...)`；`infer(...)`；`predict_action_chunk(...)`；`close()` |
| `HyEmbodiedPolicy` | `from_checkpoint(...)`；`to("rpu")`；`infer(...)`；`predict_action_chunk(...)`；`close()` |

公开 LIBERO 优化配置通过 `Pi05Policy.from_pretrained(..., optimized_profile=...)`
指定：`precision` 为 `fp16`、`w8a16`、`w8_action_nvfp4` 或
`w8_prefill_a8_action_nvfp4`；`num_cameras` 为 2 或 3；`text_tokens` 为
32、64、96 或 128。精度必须匹配 checkpoint 元数据，所需算子必须存在。
`prepare_graphs(...)` 对此配置默认预计算 AdaRMS；示例见
[`examples/configs/pi05/`](../examples/configs/pi05/)。

`Pi05Policy.prepare_graphs(batch, noise=...)` 和
`Pi05Policy.predict_action_chunk(batch, noise=...)` 接受可选的初始噪声张量。
张量必须位于 CPU、类型为 `float32`、所有值有限，形状为已加载 policy 配置定义的
`[batch_size, chunk_size, max_action_dim]`。比较固定输入时，Graph 准备和推理应使用
同一张量；不传此参数时保留原有的种子噪声生成行为。Pi0.5 示例 TOML 通过
`input.noise` 指定 `torch.save` 保存的张量文件，优化示例要求形状为 `[1, 50, 32]`。

Pi0.5 `prepare_graphs()` 返回 Graph 就绪状态与重放诊断。shape、dtype、
finite 和 Graph 不变量须通过检查。
`replay_parity.numerical_status` 为 `DIAGNOSTIC`，`task_quality` 为
`NOT_EVALUATED`。

对于发布绑定的 Pi0.5 组件 profile，传入 `profile="pi0.5-base"` 或
`profile="pi0.5-libero-v044"`，并同时传入首个预处理后的 `admission_batch`。
Loader 会在加载权重前校验精确公开 payload 和请求范围，后续 Graph 与推理请求继续按
同一 profile 校验。

`WallOssActionOutput`、`Lingbot2ActionOutput`、`HyEmbodiedActionOutput` 是对应
结构化结果。带物理单位的 action 字段应用配置的 normalization。

`WallOssPolicy.from_checkpoint(...)` 绑定 checkpoint、相机顺序、归一化与精度，
`.to("rpu")` 安装 runtime。调用 `infer(...)` 时，图像顺序必须与 `camera_names`
一致；policy 按已装载模型检查实际请求的图像几何和状态。可用 `prepare_graphs(...)`
为代表输入绑定明确的有限前缀范围，仅在请求满足该 policy 的准入条件时复用。

`RhinoVLAPolicy` 把 checkpoint 组合与 preprocessing 委托给显式 model-repository
runtime factory。Runtime 必须声明 execution capability 和实际消费的 `rpu_execution`；
facade 拒绝静默 mismatch。

字符串 `runtime_factory="module:callable"` 会 import 并执行该 module 中的 Python code。
只使用已安装、已审查的模型集成，不从不可信 TOML/request 复制该值；已导入 callable
同样需要信任。

`HyEmbodiedPolicy.from_checkpoint` 默认不读取 `norm_stats.pkl`。解码物理 action 需要
`norm_stats_path`、显式 `trust_norm_stats_pickle=True` 和准确文件的
`norm_stats_sha256`；同一字节先验证 hash，再 deserialize。
固定三相机 facade 的 `prefix_len` 仅接受 `{192, 208, 224, 240}`。请求中的图像、
noise、state embedding/state 和末端执行器 pose 必须为有限值；非法值会在 policy
执行前被拒绝。

按各 profile 的说明选择 constructor、input schema 和 precision。

### LingBot v2 runtime Graph

当前 FP16/W8A16 入口使用 `rpu_backend.adapters.lingbot_vla_v2.runtime.build_lingbot_vla_v2(..., graph_runtime_policy=...)` 和 `get_action`。[输入指南](../examples/configs/lingbot/v2/README.md)说明两/三相机、训练 YAML、语言存储和归一化 `[1,50,55]` 输出。旧 `Lingbot2Policy.from_checkpoint` facade 接受 `rpu_execution`，保留独立 exact-Z2 配置。

## Error hierarchy

用户可见错误在 `rpu_backend.api`：

```text
RPUBackendError
├── RPUConfigError
├── UnsupportedModelError
├── RPUUnsupportedDtypeError
├── RPUSingleHandleError
├── SPMExhaustionError
└── WeightShapeMismatchError
```

通用捕获 `RPUBackendError`；调用方有特定恢复动作时捕获 subclass。非法 Python 参数仍
使用标准 `TypeError`、`ValueError`。

## Graph API

`GraphRuntimePolicy` 接受冷配置 `batch_snapshot_mode="auto"` 或 `"off"`。`GraphCache` 及 Pi0.5、Wall-OSS、LingBot v2 构建器在安装权重前接收策略。AUTO 按 SDK/资产能力启用快照或回退普通重放；OFF 保留 Graph 执行。见[快照策略](runtime_config.zh.md#graph-快照策略)。

公开 namespace 为 `rpu_backend.graph`：

```python
from rpu_backend.graph import GraphCache, GraphSignature

cache = GraphCache(max_entries=4)
sig = GraphSignature(
    op_id="my_model_forward",
    shapes=(seq_len, hidden_size),
    dyn_dims=(num_layers,),
    dtypes=(torch.float16,),
)

with cache.capture(sig):
    output = torch.ops.rpu.my_model_forward(...)
```

`GraphCache` 提供 `capture`、`begin_warmup`、`freeze`、`is_frozen`、`lookup`、
`evict`、`clear`、`size`、`max_entries`、`snapshot`、`cache_invariant_ok`。Frozen cache
只允许 lookup，拒绝 online BUILD。应用通常让 adapter 管理 Graph cache；手工 Graph
主要用于 porting。

公开诊断包括 `debug_bucket_counts()`、`debug_branch_counts()`、
`dump_signature_tree()`、`explain_miss(signature)`；`Graph.debug_stats()` 返回最近提交的
聚合 counter。它们解释生命周期，不是数值或支持证据。`Graph.dump_replay_plan()` 返回
pointer-free node/segment 摘要，`Graph.dump_tree()` 按 segment 分组；均不包含 launch
argument word、设备程序或原始 address。

`Graph.runtime_policy` 和 `GraphCache.runtime_policy` 是构造时配置的只读视图，
抽取的子图继承父图配置。capture 失败后，scope 会释放借用的 graph 引用，
因此保留异常 traceback 不会额外阻止 `evict()` 或 `clear()`；调用方显式
持有的 graph 引用仍会阻止回收。

`Graph` 是低层 capture object；`get_default_graph_cache()` 返回 thread-local cache。
新 adapter 通常使用显式 per-model `GraphCache`，便于 ownership 与 teardown。

## Adapter registry 与 plugin

Registry 位于 `rpu_backend.runtime.registry`：

```python
from rpu_backend.runtime.registry import get_adapter, list_adapters, register_adapter
```

Built-in binding 在 `rpu_backend.adapters._manifest.BUILTIN_ADAPTERS`，当前包含
`Qwen3ForCausalLM`、`LlamaForCausalLM`、`DINOv3ViTModel`、
`Gemma4ForConditionalGeneration`、`Qwen3_5ForConditionalGeneration`、
`Qwen3VLForConditionalGeneration`。

外部分发包可声明 `rpu_backend.plugins` entry point：

```toml
[project.entry-points."rpu_backend.plugins"]
my_adapter = "my_package.rpu_plugin:register"
```

```python
def register(api):
    api.register_adapter("MyModelForCausalLM", MyModelAdapter)
```

Plugin 在 unknown-architecture lookup 后延迟加载，或由 `discover_plugins()` 显式发现；
不能覆盖 built-in/已注册 binding。Entry point 会执行安装包代码，只启用受信任 plugin。
Registry 是进程全局且无公开 unregister；改变 plugin set 后启动新进程。

## 模型路径 registry

```python
from rpu_backend.model_registry import MODELS, cache_root, model_path
path = model_path("qwen3-0.6b")
```

`cache_root()` 优先使用 `RPU_MODEL_CACHE`，否则为 `~/.cache/rhinoforge/models`。
`model_path(name)` 拼接受检 relative path，unknown alias 抛 `KeyError`。Alias 只是路径
映射，不是支持或资产可用声明。

## Adapter 作者的 weight-layout API

标准 helper 位于 `rpu_backend.runtime.weights`：

- `get_linear_partition(in_features, out_features, dwidth=2, num_cores=8)`；
- `swizzle_linear(weight, partition, num_cores=8)`；
- `swizzle_model_inplace(model, ..., skip_names=...)`。

Swizzle 原地改变 parameter storage，必须只运行一次。Fused subsystem 自有专用 Linear
应进入 `skip_names`，由该 subsystem 的一个权威转换路径处理。

## `torch.rpu` 设备控制

常用函数：

- `is_available()`、`device_count()`、`current_device()`；
- `set_caching_allocator(bool)`、`empty_cache()`、`memory_stats()`；
- `set_ddr_flush(bool)`、`set_ddr_flush_force(bool)`；
- `set_debug_level(0..5)`、`get_debug_level()`；
- `set_cross_layer_batch_prefill(bool)`、`set_cross_layer_batch_size(int)`；
- `shutdown()`。

开发诊断：

- `set_debug` / `get_debug` / `set_profile` / `get_profile` /
  `reset_profile_accumulators`；
- `set_spm_debug` / `get_spm_debug` / `spm_alloc_dump`。

这些诊断只暴露日志和聚合计数，不暴露模型输入、中间 activation、raw address
或 Graph register/resource payload。

`set_caching_allocator(bool)` 是进程级冷态 tensor storage 选择。第一次显式调用，
或尚未选择时第一次分配非空 设备 tensor，都会把策略冻结到进程结束。同值重复调用
是幂等的；请求相反值会报错，必须使用新进程。应在任何 设备 tensor 物化前调用。
`empty_cache()` 只释放未使用的 cached block，不能释放 live tensor 或 Graph-owned
storage。`memory_stats()` 通过 `caching_allocator_enabled` 返回所选模式，并通过
`caching_allocator_policy_frozen` 返回策略是否已经冻结。

CPU/设备 boundary flush 默认打开，正常推理不要关闭。Chunk size 属于 per-handle
`rpu_execution`，不是进程全局 `torch.rpu` setter。内部 `torch.ops.rpu.*` 是 adapter
使用的 host wrapper interface；未在本页或 model example 中记录的函数视为实现细节。

<a id="bounded-speech-apis"></a>
## 语音 API

```python
from rpu_backend.api import Qwen3ASRModel, Qwen3TTSModel
```

| API | 输入与返回值 |
|---|---|
| `Qwen3ASRModel.from_pretrained(checkpoint, dtype=torch.float16)` | 本地固定 Qwen3-ASR-0.6B checkpoint |
| `model.transcribe(audio, sample_rate=16000, max_new_tokens=64)` | 一维 CPU 有限浮点波形，单声道 16 kHz、0.5–8 秒；生成上限 1–64。返回 `ASRTranscription(text, raw_text, token_ids)` |
| `Qwen3TTSModel.from_pretrained(checkpoint, source=prepared_package, dtype=torch.float16)` | 本地固定 Qwen3-TTS-12Hz-0.6B-Base checkpoint 和准备后的 `qwen_tts/` 源码目录 |
| `model.synthesize(text, reference_audio=local_file, language="Chinese", max_new_tokens=64)` | 文本 1–128 字符且最多 32 tokenizer token；语言 `Chinese` / `English`；参考音频为有限单声道 24 kHz、0.5–20 秒；生成上限 2–64。返回独立 float32 单声道波形及采样率 24000 |

ASR 使用 greedy，不提供 context、时间戳或流式接口。TTS 使用非流式 greedy，
只依赖参考 speaker embedding。Talker prompt 加生成上限须在 128 个位置内；
达到上限可能截断句子。下载、源码准备、CPU/设备精度边界及命令见
[语音指南](qwen3_audio.zh.md)。

两类 API 均支持 `with`、幂等 `close()` 和 `execution_stats()`。请求、统计及关闭
串行调用，一个进程只保留一个存活设备模型/策略。加载器在转换前检查资产，并在
设备分配前选择 caching allocator；TTS 还声明 owner 专属五个 arena。
进程已冻结为不兼容 allocator 或 arena 计划时，启动新进程。

ASR 需要重编译的 native 提供 `siglip_set_audio_mode`、`siglip_set_audio_projection`
和 `assemble_inputs_embeds`。TTS codec 显式调用 `causal_decoder_set_tts_codec_mode`；
缺少 setter 时前段保留 CPU 路径，其他执行错误直接报告。codec 输入限于 1–63 帧。

`execution_stats()` 返回调用及缓存状态：

- ASR 为当前音频 token 长度保留一张音频图，为当前 prompt 长度保留最多一对
  prefill/decode 图。`audio_graph_size`、`audio_graph_replays`、
  `audio_graph_invariant_ok` 描述音频缓存；`graph_size`、`graph_replays`、
  `graph_invariant_ok` 描述文本缓存。替换缓存签名后重放计数重新累计；
  `audio_linear_calls` 统计逻辑投影调用。
- TTS 返回 `codec_frontend.device`、`fallback_reason`、输出头调用计数以及各组件
  的 calls、tokens 和 Graph 状态。Talker 在 prompt 长度变化时替换 prefill 图并保留
  decode；Predictor 保留一张 decode 图；codec 在帧数变化时替换自己的图。

每个新序列重置逻辑 KV 位置并保留私有存储。TTS 按解码后的音频内容只缓存最新
speaker embedding，每次请求仍读取和校验参考文件。输出存储独立于重放；释放时
先清 Graph，再释放 native handle 和权重。CPU 线程数由调用方控制；两种 CLI 均提供
`--threads`，默认 4。
