# LingBot v2 运行时 Graph 配置

配置覆盖两/三相机 × 16/32/64/128/160 个有效语言 token × FP16/W8A16。它们使用当前运行时 Graph 路径、TP8、原生十步去噪和 32 个专家中的 Top4 路由，调用 `rpu_backend.adapters.lingbot_vla_v2.runtime.build_lingbot_vla_v2` 与 `get_action`。公开三相机 T3 保留为独立的两精度对照；旧 exact-Z2 已迁入 [legacy/exact_z2/](../legacy/exact_z2/README.md)。

| 输入范围 | FP16 | W8A16 | 输入环境变量 |
|---|---|---|---|
| 两相机, T16 | [FP16](2cam_t16/fp16.toml) | [W8A16](2cam_t16/w8a16.toml) | `LINGBOT2_2CAM_T16_OBSERVATION` |
| 两相机, T32 | [FP16](2cam_t32/fp16.toml) | [W8A16](2cam_t32/w8a16.toml) | `LINGBOT2_2CAM_T32_OBSERVATION` |
| 两相机, T64 | [FP16](2cam_t64/fp16.toml) | [W8A16](2cam_t64/w8a16.toml) | `LINGBOT2_2CAM_T64_OBSERVATION` |
| 两相机, T128 | [FP16](2cam_t128/fp16.toml) | [W8A16](2cam_t128/w8a16.toml) | `LINGBOT2_2CAM_T128_OBSERVATION` |
| 两相机, T160 | [FP16](2cam_t160/fp16.toml) | [W8A16](2cam_t160/w8a16.toml) | `LINGBOT2_2CAM_T160_OBSERVATION` |
| 三相机, T16 | [FP16](3cam_t16/fp16.toml) | [W8A16](3cam_t16/w8a16.toml) | `LINGBOT2_3CAM_T16_OBSERVATION` |
| 三相机, T32 | [FP16](3cam_t32/fp16.toml) | [W8A16](3cam_t32/w8a16.toml) | `LINGBOT2_3CAM_T32_OBSERVATION` |
| 三相机, T64 | [FP16](3cam_t64/fp16.toml) | [W8A16](3cam_t64/w8a16.toml) | `LINGBOT2_3CAM_T64_OBSERVATION` |
| 三相机, T128 | [FP16](3cam_t128/fp16.toml) | [W8A16](3cam_t128/w8a16.toml) | `LINGBOT2_3CAM_T128_OBSERVATION` |
| 三相机, T160 | [FP16](3cam_t160/fp16.toml) | [W8A16](3cam_t160/w8a16.toml) | `LINGBOT2_3CAM_T160_OBSERVATION` |

原公开参考使用 [3cam T3 FP16](3cam_t3/fp16.toml) 或 [W8A16](3cam_t3/w8a16.toml)，`LINGBOT2_PUBLIC_CASE` 指向固定样例目录。T 是 language mask 中实际为 true 的数量，不是填充后的张量宽度。

<a id="weights"></a>
<a id="weight-sources"></a>
## 官方来源与原始存储精度

| 资产 | 官方下载 / 源码 | 原始存储精度与用途 |
|---|---|---|
| v2 6B 父权重 | [robbyant/lingbot-vla-v2-6b](https://huggingface.co/robbyant/lingbot-vla-v2-6b)；[官方源码](https://github.com/robbyant/lingbot-vla-v2) | 版本 `11c703bf6a5c1f45b3b69168482da11fdbba53d7` 的主要模型 safetensors 为 **F32**，见 [HF 元数据](https://huggingface.co/api/models/robbyant/lingbot-vla-v2-6b)。它不是 FP16/W8 下载 |
| 训练 YAML 与输入 | 对应交付/运行记录中的匹配 YAML、公开 T3 case 或真实 observation | 非权重量化产物；必须与所选相机、token 数和掩码匹配，单独下载模型不能补齐这些输入 |

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
hf download robbyant/lingbot-vla-v2-6b --revision 11c703bf6a5c1f45b3b69168482da11fdbba53d7 --local-dir "$RPU_MODEL_CACHE/lingbot-vla-v2-6b"
```

<a id="weight-recipes"></a>
## 模块精度与配方

| 配置组 | 模块实际 W/A 精度 | 输入父权重 → 转换方式与命令 |
|---|---|---|
| 所有 `*/fp16.toml` | Vision、Text、Action 投影 W16/A16；不是原文件的 F32 存储精度 | 上述 F32 父权重 → 加载时转 FP16；[完整命令](#fp16-loading)，无低比特量化 |
| 所有 `*/w8a16.toml` | Vision block Q/K/V/O/FC1/FC2、Text Q/K/V/O/Gate/Up/Down、Action Q/K/V/O 及共享/路由专家 Gate/Up/Down 均 W8/A16，逐输出通道 INT8 + FP16 scale；Vision patch/merger、Action router/IO/条件投影、norm/bias/embedding 保留浮点，设备权重为 FP16 | 同一 F32 父权重 → [加载时量化命令](#quantization)；不生成量化目录，没有离线 CLI |

两种配方均支持下表中的相机/长度模板；旧 exact-Z2、历史 BF16 父资产和未列出的 W4 实验不包含在本页配方中。实现入口为 [runtime.py](../../../../python/rpu_backend/adapters/lingbot_vla_v2/runtime.py) 的 `_quantize_text_w8a16()`、[convert.py](../../../../python/rpu_backend/adapters/lingbot_vla_v2/convert.py) 的 `build_expert_moe()` / `pack_sparse_expert_projection()` 以及共享 [Vision 安装器](../../../../python/rpu_backend/adapters/qwen3_vl/vision.py)。

<a id="quantization"></a>
## 加载时量化

<a id="fp16-loading"></a>
### FP16：只做浮点类型转换

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
export LINGBOT2_TRAINING_CONFIG=/path/to/matching-training.yaml
export LINGBOT2_PUBLIC_CASE=/path/to/frozen-3cam-t3-case
cp examples/configs/lingbot/v2/3cam_t3/fp16.toml /tmp/lingbot2_fp16.toml
# run.output_dir = 本次独立输出目录；保留 precision="fp16"。
vi /tmp/lingbot2_fp16.toml
python examples/run.py --config /tmp/lingbot2_fp16.toml
```

其他 FP16 相机/长度变体使用对应 `fp16.toml`、匹配该相机数与长度的输入，以及下表列出的 observation 环境变量。

以下命令在仓库根目录执行。量化在最后一行加载模型时发生，不生成独立量化目录。先替换实际路径，再按注释用 `vi` 编辑副本并保存；`--check-config` 只校验配置，不能代替最后的运行命令。

### 当前 v2 W8：三相机公开 T3

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
# 上述缓存根下放完整原始 F32 目录 lingbot-vla-v2-6b。
export LINGBOT2_TRAINING_CONFIG=/path/to/matching-training.yaml
export LINGBOT2_PUBLIC_CASE=/path/to/frozen-3cam-t3-case
cp examples/configs/lingbot/v2/3cam_t3/w8a16.toml /tmp/lingbot2_w8.toml
# run.output_dir = 本次独立输出目录。input 的三个变量由上面 export 解析。
vi /tmp/lingbot2_w8.toml
python examples/run.py --config /tmp/lingbot2_w8.toml
```

<a id="当前-v2-w8两相机-t16-observation"></a>
### 当前 v2 W8：两相机 T16 观测输入

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
export LINGBOT2_TRAINING_CONFIG=/path/to/matching-training.yaml
export LINGBOT2_2CAM_T16_OBSERVATION=/path/to/real-2cam-t16-observation.pt
cp examples/configs/lingbot/v2/2cam_t16/w8a16.toml /tmp/lingbot2_2cam_t16_w8.toml
# run.output_dir = 本次独立输出目录。保留 cameras=2 和 expected_language_tokens=16。
vi /tmp/lingbot2_2cam_t16_w8.toml
python examples/run.py --config /tmp/lingbot2_2cam_t16_w8.toml
```

其他相机/T 模板使用匹配的输入，并导出 `input.observation` 指定的变量，如 `LINGBOT2_3CAM_T160_OBSERVATION`。**v2 FP16 无需量化**，同目录 `fp16.toml` 使用原始父权重。**v1 混合浮点无需量化，测试仍延期**；旧 exact-Z2 的掩码/安装合同保持独立。

**实现位置**：[examples/runners/lingbot2/optimized.py](../../../runners/lingbot2/optimized.py)；[python/rpu_backend/adapters/lingbot_vla_v2/runtime.py](../../../../python/rpu_backend/adapters/lingbot_vla_v2/runtime.py)；[python/rpu_backend/adapters/lingbot_vla_v2/convert.py](../../../../python/rpu_backend/adapters/lingbot_vla_v2/convert.py)；[python/rpu_backend/adapters/qwen3_vl/vision.py](../../../../python/rpu_backend/adapters/qwen3_vl/vision.py)。`run_example_config()` → `build_lingbot_vla_v2()`；Text 用 `_quantize_text_w8a16()`，Action 用 `build_expert_moe()` / `pack_sparse_expert_projection()`，Vision 经共享安装器。runner 从 TOML 选择冷量化参数，不需手设额外量化环境变量，也没有离线转换 CLI。

## 运行

设置权重、训练 YAML 及所选 observation 环境变量；使用 T3 时设置 `LINGBOT2_PUBLIC_CASE`。

```bash
python examples/run.py --config examples/configs/lingbot/v2/3cam_t3/w8a16.toml --check-config
python examples/run.py --config examples/configs/lingbot/v2/3cam_t3/w8a16.toml
```

`--check-config` 仅检查配置，无需 Torch 或输入资产。`~` 和环境变量在执行时展开，未解析变量报错。每种冷配置使用新进程。

源码安装读取两份 `src/*_current.ref`；Release 包读取 wheel 中匹配的资产。两者均校验 manifest 哈希，拒绝额外 REF、冲突覆盖和不完整的源码资产。安装方法见 [运行库](../../../../docs/runtime_assets.md)。

<a id="input-modes"></a>
## 输入方式

`[input]` 恰选一种：

- `case`：目录含 `case.json`，指向带校验和的输入/参考张量。须声明十步去噪、策略后处理之前的完整 55 维归一化输出，以及与训练 YAML 匹配的对齐掩码。支持公开 `input.safetensors` / `golden.safetensors` 格式。
- `observation`：含下表字段的 `.pt` 或 `.safetensors` CPU 张量字典；`.pt` 使用 `weights_only=True` 读取。
- `images`：两/三张图片路径、真实 `instruction` 和含 1–55 个有限值的 `state`。状态按公开 API 补零至 55 维。默认精确 CPU FP32 resize 及 checkpoint processor/tokenizer 在加载模型前运行一次。可选 `noise` 指向 CPU 张量文件；否则按 `run.seed` 生成一次 `[1,50,55]` 噪声并复用。

observation 字段如下：

| 字段 | CPU 类型与形状 |
|---|---|
| `images` (case 别名 `pixel_values`) | Float32 `[2 or 3,256,1536]`, 来自 256×256 相机图片 |
| `image_grid_thw` | Int64 `[cameras,3]`, 每行 `[1,16,16]` |
| `img_masks` | Boolean `[cameras]`, 所有配置相机均有效 |
| `lang_tokens`, `lang_masks` | Int64 和 Boolean `[1,L]`, `1 <= L <= max_language_tokens`, 至少一个有效 token |
| `state` | Float32 `[1,55]` |
| `noise` (case 别名 `initial_noise`) | Float32 `[1,50,55]` |

`cameras` 与 `expected_language_tokens` 可选，用于断言自定义输入。`max_language_tokens` 默认 72，可显式提高至 160；T128/T160 模板设 160。该值约束存储并填充分词输出，不截断原指令。指令超过上限会在模型加载前失败；须显式增加上限或提供另一条完整指令。输出为归一化 `[1,50,55]`，不执行机器人专用动作后处理。

原始图片输入可用以下字段替换模板的 `case`/`observation`：

```toml
images = ["/path/to/front.png", "/path/to/wrist.png"]
instruction = "Move the red object into the tray."
state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
cameras = 2
# 未确认该 tokenizer 的实际 token 数时，省略 expected_language_tokens。
```

## 运行配置

- `[lingbot2].precision` 为 `fp16` 或 `w8a16`。W8 量化 Language、Vision 和 Action 支持的投影；router、patch、merger、Action-IO 保留 FP16。
- `instruction_residency="auto"` 由 `GraphRuntimePolicy`、SDK 和匹配的算子资产共同选择快照重放；不满足条件时使用普通 Graph。`"off"` 关闭快照，仍保留 Graph。旧快照环境开关会报错。
- 训练 YAML 决定对齐和注意力掩码，缺省字段遵循官方解析默认值。输入须匹配 Top4、32 专家、36 层 Action 和长度 50。
- AUTO 填充预算：短输入为 15；T128/T160 及三相机 T64 为 63。规划器选择物理布局，保留实际语言 token 和 mask。
- 默认 `warmup=2`、`runs=5`、`cpu.threads=1`。另有五次 A/B/A 与 READY 准备，共 12 次调用。准备后 Graph 身份保持稳定，重放及提交计数递增；OFF 或回退路径的 prepared miss 可递增。
- `run.output_dir` 保存 `actions.pt`、`report.json`、执行计划和 Graph 状态。`--profile` 在常规运行后增加一次诊断调用。

实验性 Prefill KV 配对、QK-norm/RoPE 融合、稀疏 KV 镜像和路由替换保持关闭。
