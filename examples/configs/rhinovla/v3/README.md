<a id="rhinovla-v3-factory-配置"></a>

# RhinoVLA v3 配置

v3 使用公开的 96D 模型，通过 factory 创建 Vision、语言模型和动作专家组成的流水线。五份 non-RTC 配置从 FP16 开始，依次扩大量化范围，最后一档再调整非线性精度和 Q/K 归一化。另有独立的 [RTC FP16 配置](#rtc)，覆盖 `delay=0..6` 的请求契约。

[模型入口](../README.md) · [准备与运行](#quantization) · [RTC](#rtc) · <a id="weight-recipes"></a>

## 配置说明

五份配置的 `input.checkpoint` 均填写同一份公开 v3 原始浮点权重。W8 在模型加载时完成，不需要预先生成五份量化目录。W8A16 表示逐输出通道 INT8 权重、FP16 scale 和 FP16 激活。

| 配置 | 相对前一档的主要变化 | 权重准备 |
|---|---|---|
| [FP16](fp16.toml) | 浮点基准，已启用缓存、融合和 Graph replay | [同一套 v3 原始权重](#quantization)，加载为 FP16 |
| [专家 W8](w8a16_action.toml) | 动作专家每层 Q/K/V/O、Gate/Up/Down 七类投影改为 W8 | [复用原始权重](#quantization)，加载时量化 |
| [完整专家 W8](w8a16_action_adarms.toml) | 增加 18 组 AdaRMS 条件投影 W8 | [复用原始权重](#quantization)，加载时转换并生成表 |
| [全流水线 W8](w8a16_pipeline.toml) | 增加 Vision、Text、动作 IO、最终 AdaRMS 的 W8；同时调整缓存、padding 和 CPU 设置 | [复用原始权重](#quantization)，加载时量化 |
| [全流水线 W8 + HIGH/vector QK](w8a16_pipeline_high_fused_vector_qk.toml) | 权重范围相同，开启五项算术、融合和预计算选项 | [复用原始权重](#quantization)，同一量化配方 |

两份全流水线配置为实验配方。各配置的模块范围和执行差异如下；表中的递进关系用于解释，每份 TOML 均为完整配置。

<a id="baseline"></a>

### FP16 基准

流水线先编码图像，再生成语言模型 prefix，最后执行十步动作去噪。动作专家为 18 层、隐藏维度 1024，动作 horizon 为 30；加入 state token 后，专家处理 31 个有效 suffix token。

FP16 模板已经开启下列优化，后续四档沿用：

| 部分 | 已开启的行为 | 主要字段 |
|---|---|---|
| Vision | 多视角批处理、准备数据缓存；设备 patch/merger、融合 merger、将 merger 纳入图重放 | `VISION_BATCH_VIEWS`、`VISION_PREP_CACHE`；`rpu_patch_embed`、`rpu_mergers`、`fused_merger`、`bake_merger` |
| Language | 缓存 prefix 准备数据，在设备组装 prefix；复用存储、partial MRoPE、merger scatter | `prefix_prep_cache`、`prefix_prep_rpu`、`prefix_no_clone`、`partial_mrope`、`merger_scatter` |
| Action | 展开十步去噪、共享 prefix cache、复用 prefix 存储；state/mask 投影在每次请求的首步计算后复用 | `denoise_unroll`、`shared_cache`、`prefix_alias`、`DENOISE_STATIC_CONTEXT_CACHE` |
| 专家计算 | AdaRMS 表预计算并驻留、gated residual 简化、SiLU×Mul 和专家算子融合、KV 对齐 | `PRECOMPUTE_ADARMS`、`ADARMS_RESIDENT`、`GATED_NO_SUB`、`FUSED_SILU_MUL`、`EXPERT_FUSIONS`、`ALIGNED_KV` |
| 图执行 | 三个组件开启快速重放，Vision/Language 重放时跳过重复的预加载处理 | `fast_replay`、`preload_replay_skip` |

本页表格中的大写开关均带 `RPU_RHINOVLA_` 前缀，写在 `[example.opt_in]`；小写字段位于对应的 `rpu_execution.components` 表。全部模板关闭 `PACKED_QKV` 和 `SKIP_ADARMS_GEMV`。

FP16 描述运行时的主要权重与激活精度。AdaRMS 条件投影在加载时用浮点计算生成表，表以 FP16 存储。

### 专家 W8：量化七类主干投影

相对 FP16，选择专家 W8 的 `example.profile_id`，将 `example.dtype` 改为 `w8a16`，并增加量化参数：

```toml
[example]
profile_id = "rhinovla.v3.w8a16.expert.trusted-factory"
dtype = "w8a16"

[input]
expert_w8a16 = true
full_w8a16 = false
```

18 层专家中，Attention 的 Q/K/V/O 和 MLP 的 Gate/Up/Down 使用 W8A16。Vision、Text、AdaRMS 条件投影、最终 norm 和动作 IO 保持浮点路径。七类投影在去噪过程中反复执行，量化减少了这些权重的存储和读取量。

该模板通过上述 factory 参数选择专家精度，两个环境开关 `EXPERT_W8A16` 和 `FULL_W8A16` 仍为 `"0"`。runner 在首次推理前核对实际安装的 INT8 权重和 FP16 scale。

### 完整专家 W8：再量化 AdaRMS 条件投影

相对专家 W8，将 `input.full_w8a16` 改为 `true`，并选择 `example.profile_id="rhinovla.v3.w8a16.full-expert.trusted-factory"`。AdaRMS 条件投影根据 timestep 生成归一化使用的 scale、shift 和 gate。

每层把 Attention 前和 MLP 前的两个条件投影拼成一组，形成 18 组物理权重，对应 36 个逻辑投影。加载时运行这些 W8 投影，生成十步去噪的 FP16 参数表；最终 norm 的条件投影仍用浮点预计算，动作 IO、Vision 和 Text 也保持浮点。

基准已经预计算 AdaRMS，因此这一档主要改变生成参数表的权重和算术，量化范围扩大不意味着每次推理的计算量进一步减少。这里的 `input.full_w8a16` 只描述专家范围。

<a id="full-pipeline"></a>

### 全流水线 W8：扩大范围并调整运行设置

选用 `example.profile_id="rhinovla.v3.w8a16.full-pipeline.trusted-factory"`，在完整专家 W8 之上增加以下量化：

| 部分 | 新增 W8 的投影 |
|---|---|
| Text prefix | 每层 Q/K/V/O、Gate/Up/Down |
| Vision | block 投影、折叠后的 patch projection、最终和 DeepStack merger 的 MLP |
| 动作 IO | action 输入、state 输入、state mask、action mask、action 输出 |
| 最终 AdaRMS | 最终 norm 的条件投影；预计算从 18 组扩至 19 组 |

norm、bias、embedding 和位置张量保留浮点。除量化外，该模板还显式设置：

| 字段 | 相对完整专家 W8 的变化 | 作用 |
|---|---|---|
| `EXPERT_W8A16`、`FULL_W8A16` | `"0"` → `"1"` | 选择全流水线安装路径 |
| `VISION_INPUT_CACHE` | 新增 `"1"` | 复用 CPU FP16 转换及设备图像输入缓冲，每次请求刷新像素 |
| `language_model.prefill.padding_rows` | 新增 `10` | 指定物理 prefix padding，逻辑 token 数保持原值 |
| Text/Vision `linear_acc32` | 显式设为 `false` | 选择 ACC16 累加 |
| `cpu.threads` | 从继承环境改为 `12` | 指定 Torch CPU 线程数 |
| `run.inference_mode` | 从默认 `true` 改为 `false` | 使用 `no_grad` 执行 |
| HIGH、HIGH 融合、gate tanh 预计算、vector Q/K norm | 显式设为 `"0"` | 使用基础非线性和归一化路径 |
| `PARTIAL_ROPE` | 显式设为 `"0"` | 保持专家原 RoPE 路径；与语言模型的 `partial_mrope` 分别控制 |

全流水线配置同时改变上述量化、缓存和运行参数。

<a id="全流水线-w8--highvector-qk"></a>

### HIGH/vector QK：保持权重配方，改变计算路径

本档的 `example.profile_id` 为 `rhinovla.v3.w8a16.full-pipeline-high.trusted-factory`。相对普通全流水线 W8，以下五项从 `"0"` 改为 `"1"`。线程、padding 和 Text/Vision 的 `linear_acc32=false` 保持相同。

| 开关 | 具体变化 |
|---|---|
| `HIGH_PRECISION` | Text/Action 使用 Newton RMSNorm、高精度 SiLU 和 gate tanh；Vision merger 使用 GELU-ERF，Vision encoder 的 GELU-tanh 使用 HIGH 实现 |
| `HIGH_PRECISION_FUSIONS` | Action 使用高精度 norm+shift 融合，Text/Action 使用高精度 SiLU×up 融合，保留中间 FP16 舍入 |
| `PRECOMPUTE_GATE_TANH` | 加载时把固定 gate 的 tanh 写入 AdaRMS 表，推理重放直接读取结果 |
| `VECTOR_K_NORM` | 提前清零 K/V 尾部，用已有 `32×128` 对齐缓冲执行 Newton V32 K norm |
| `VECTOR_Q_NORM` | Q norm 的物理输入从 `62×128` 补齐至 `64×128`，每次清零新增两行，使用 Newton V32 |

vector Q/K 调整的是动作专家的归一化执行方式，后续 RoPE、attention 和投影仍使用逻辑尺寸。HIGH 调整非线性和归一化算术；Linear 的 ACC16/ACC32 由独立字段选择。

<a id="weight-sources"></a>
<a id="weights"></a>

<a id="quantization"></a>

## 准备与运行

### 固定源码与 checkpoint

使用 [HUIXI-AI/RhinoVLA-v3](https://github.com/HUIXI-AI/RhinoVLA-v3) 提交 `e60650377dca38a78bcd4a2e2d58f9e6a631a275`，checkpoint 为同仓库 Git LFS 文件 `models/rhinovla_v3/steps_150000/model.pt`，SHA-256 为 `2303f9e44cc5847a9990a617cf56c30dfb0ca05f38b6f08b152bf95e67e2f70f`。

```bash
RHINOVLA_V3_SRC=/path/to/RhinoVLA-v3
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/HUIXI-AI/RhinoVLA-v3.git "$RHINOVLA_V3_SRC"
git -C "$RHINOVLA_V3_SRC" checkout e60650377dca38a78bcd4a2e2d58f9e6a631a275
git -C "$RHINOVLA_V3_SRC" lfs pull
RHINOVLA_V3_CKPT="$RHINOVLA_V3_SRC/models/rhinovla_v3/steps_150000/model.pt"
python scripts/validate_rhinovla_v3.py prepare \
  --source "$RHINOVLA_V3_SRC" --checkpoint "$RHINOVLA_V3_CKPT" \
  --fixture /tmp/rhinovla_v3_inputs.pt --example-dir /tmp/rhinovla_v3_example
```

准备环境需要 `qwen-vl-utils==0.0.14`（包含在 `vla` 安装选项中）。准备命令在 CPU 上使用同版本 processor 和 `datasets/example_lerobot_v3/videos/` 的三个相机视频生成输入，state/noise 使用固定种子。输出 fixture 供[独立运行配置](controlled/README.md)使用，`--example-dir` 另生成统一入口所需的 `qin.pt`、`request.pt` 和 `factory.json`。

<a id="fp16-loading"></a>
<a id="使用"></a>

### 统一入口

模板中的 `your_package.rhinovla_v3:create_runtime` 由调用者填写。使用上述公开源码与准备工具时，显式选择内置 `rpu_backend.adapters.rhinovla.factory:v3_runtime_factory`；它按所选 TOML 安装运行配置，W8 在加载时转换。

```bash
cp examples/configs/rhinovla/v3/fp16.toml /tmp/rhinovla_v3.toml
# input.checkpoint = RHINOVLA_V3_CKPT 的实际路径。
# input.runtime_factory = "rpu_backend.adapters.rhinovla.factory:v3_runtime_factory"
# input.factory_config = /tmp/rhinovla_v3_example/factory.json
# input.request = /tmp/rhinovla_v3_example/request.pt
# run.output_dir = 本次结果目录。
vi /tmp/rhinovla_v3.toml
python examples/run.py --config /tmp/rhinovla_v3.toml --check-config
python examples/run.py --config /tmp/rhinovla_v3.toml
```

其他档位替换复制的 TOML，继续使用同一套原始权重与输入。`factory.json` 的 `model_root` 指向源码，`qin` 指向保存的 CPU 输入；统一入口读取后交给内置 factory。`request.pt` 包含状态、噪声和掩码。`trust_model_code=true` 允许执行固定模型源码。已有自定义集成仍可填写自己的 `module:callable` 和配对文件，其 factory 配置原样传入。

| 输入字段 | 内容与绑定 |
|---|---|
| `input.checkpoint` | 同一套 v3 / 96D 浮点 checkpoint，五份配置共用 |
| `input.runtime_factory` | 调用者提供的 `module:callable`；公开资产可选上述内置 factory |
| `input.factory_config` | 准备工具生成的 JSON，含源码与输入位置；同名量化布尔值必须与 TOML 一致 |
| `input.request` | 配对的状态、噪声和掩码，传给 `policy.predict(**request)` |
| `run.output_dir` | 本次产物目录 |

自定义 factory 返回与所选配置匹配的 `RhinoVLAOnRPU`。

factory 接收 checkpoint、steps、量化选择和组件配置。修改安装参数后使用新进程；缓存复用时仍需刷新图像、状态及噪声。

<a id="rtc"></a>

### RTC FP16：delay 0–6

[RTC 模板](rtc_fp16.toml)绑定上述固定 v3 源码与 checkpoint、FP16、batch 1、30×96 动作和 18 层专家，使用十步展开去噪。该精确输入范围已完成本地 R1 硬件验证；配置登记仍为 `example_only` 模板，使用内置 v3 factory，不扩展其他 profile 的支持状态。W8、v1、其他 horizon/步数及 `delay > 6` 不在此契约内。

继续使用[统一安装指南](../../../../docs/getting_started.md)创建的 Miniconda3 `RHINOFORGE_ENV`。完整环境包可提供 `vla` 依赖；运行 RTC 还需要包含本节接口的新版本 RhinoForge 源码及同版本编译的 native 扩展。旧环境包需按统一指南更新或重新构建，不能仅复制 RTC Python 文件或 TOML 后沿用旧 native 库。按照前文生成普通 `factory.json`、`qin.pt` 和 `request.pt`，再创建 RTC 输入：

```bash
export RHINOVLA_V3_EXAMPLE=/tmp/rhinovla_v3_example
python - <<'PY_RTC'
import json
import os
from pathlib import Path
import torch

root = Path(os.environ["RHINOVLA_V3_EXAMPLE"])
factory = json.loads((root / "factory.json").read_text())
factory["rtc"] = True
(root / "rtc_factory.json").write_text(json.dumps(factory, indent=2))
request = torch.load(root / "request.pt", map_location="cpu", weights_only=True)
request["delay"] = 6
# A deterministic test prefix. In deployment, use the already committed actions
# in the same normalized 96D action space as this checkpoint.
request["action_prefix"] = torch.linspace(-0.25, 0.25, 6 * 96).reshape(1, 6, 96)
torch.save(request, root / "rtc_request.pt")
PY_RTC
cp examples/configs/rhinovla/v3/rtc_fp16.toml /tmp/rhinovla_v3_rtc.toml
# Fill input.checkpoint, input.factory_config, input.request and run.output_dir.
vi /tmp/rhinovla_v3_rtc.toml
python -m examples.run --config /tmp/rhinovla_v3_rtc.toml --check-config
python -m examples.run --config /tmp/rhinovla_v3_rtc.toml
```

`--check-config`只检查 TOML；实际运行时，runner 会在加载 qin 和模型权重前检查 RTC 前缀与 delay。内置 factory 固定为 `rpu_backend.adapters.rhinovla.factory:v3_runtime_factory`。`factory_config.rtc=true` 是构造时设置；更改它需要重建 policy。

| 字段 | 契约 |
|---|---|
| `factory.json` 的 `rtc` | RTC 模板要求 JSON 布尔值 `true`；non-RTC 模板保持省略或 `false` |
| `request["delay"]` | Python `int`，允许 `0,1,2,3,4,5,6`；不接受 `bool`、浮点数或 Tensor |
| `request["action_prefix"]` | 有限的 CPU 浮点 Tensor，形状 `[1,D,96]` 或 `[D,96]`；`delay <= D <= 30` |
| `x0/state/state_mask/action_mask` | 有限的 CPU 浮点输入；动作掩码在 30 行间保持相同；输出为 `[1,30,96]` |

RTC 请求必须同时提供 `action_prefix` 和 `delay`。`delay=0` 时仍需提供 prefix，可用 `torch.empty(1,0,96)`；前缀中第 `delay` 行之后的值不参与固定区间。掩码处理后的前缀还须能表示为有限 FP16。普通 non-RTC policy 不接收这两个参数。

RTC 每一步先复写已确定的前 `delay` 个动作，再执行去噪；这些 token 的时间为 0，其余动作使用本步时间。state token 使用全部动作时间嵌入的均值，最终输出再次固定前缀并应用动作掩码。只在输出阶段覆盖历史动作不能实现这一计算路径。

RTC 模板保持 `PRECOMPUTE_ADARMS=1` 和 `denoise_unroll=true`，将 `EXPERT_FUSIONS`、`ADARMS_RESIDENT`、`PACKED_QKV` 及 W8 开关设为 `"0"`。使用模板中的整组设置。每个 delay 对应独立的动作 Graph 签名；同一 delay 更新 prefix 内容需要刷新输入。runtime 在构造时自动预留 10 个 Graph arena，覆盖七个 delay 的动作 Graph、Vision/Language 保留 Graph 和一个准备阶段的临时 Graph；用户无需额外设置环境变量。固定 Graph 集合前先预热需要使用的所有 delay。验证应逐个覆盖 0–6、同 delay 前缀变化、0→6→0 切换和输出存储独立性，不能用 non-RTC 的既有结果替代。

本地 R1 验证使用三幅等尺寸视图、230 个逻辑 prefix token（P230）：`delay=0..6` 的 BUILD/REPLAY 逐位一致，前缀固定、有限值、零掩码、输出独立性、缓存稳定性及 0→6→0、同 delay 前缀 A→B→A 检查均通过。与 CPU FP32 参考的余弦相似度为 0.9999742–0.9999917，仅作数值诊断，不设通用验收阈值。

模板会在构造时固定 qin 中的 token、attention mask 和 grid；这些内容或像素几何变化时需新建 runtime，同几何的像素值可更新。异尺寸视图不在本轮验证范围内，强制 oneshot 模式会被提前拒绝。

<a id="parameters"></a>

### 需要填写和常用调整项

| 字段 | 模板设置与调整方法 |
|---|---|
| `input.num_steps` | 固定 `10`，AdaRMS 表与十步 schedule 配套 |
| `input.expert_w8a16/full_w8a16` | 按所选配方保留；同时匹配 `example.profile_id`、`dtype` 和 factory JSON |
| `rpu_execution.components.language_model.prefill.linear_acc32` | 全流水线模板显式为 `false`；`true` 选择 Text Linear ACC32 |
| `rpu_execution.components.vision_encoder.vision.linear_acc32` | 全流水线模板显式为 `false`；`true` 选择 Vision block 与融合 merger 的 Linear ACC32 |
| `rpu_execution.components.language_model.prefill.padding_rows` | 全流水线为 `10`；随实际 prefix 的物理执行范围调整 |
| `cpu.threads` | 全流水线为 `12`，其余继承环境 |
| `run.warmup/runs` | 默认 `2/3` |

公共字段见[参数说明](../../PARAMETERS.md)。

Text/Vision 的 ACC 字段不控制 action、patch embedding 和冷 AdaRMS 投影，也不写入 action 表。现有优化要求完整 18 层专家、96D 动作和上述十步输入结构。

### 实现参考

- [runner](../../../runners/rhinovla/runner.py)：绑定 factory 参数并核对实际量化范围。
- [convert.py](../../../../python/rpu_backend/adapters/rhinovla/convert.py)：专家投影量化、AdaRMS 预计算及动作 IO 安装。
- [precision.py](../../../../python/rpu_backend/adapters/rhinovla/precision.py)：Text 量化、Linear 累加与全流水线权重检查。
- [pipeline.py](../../../../python/rpu_backend/adapters/rhinovla/pipeline.py)：组件构造、缓存和执行。
- [运行参数](../../../../docs/runtime_config.md)与[两份运行库](../../../../docs/runtime_assets.md)。
