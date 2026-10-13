# LingBot v2 配置

[模型入口](../README.md) · [配置索引](../../CONFIGURATIONS.md) · [准备与运行](#使用)

本目录通过模型 adapter 运行 LingBot v2 6B，输出 `[1,50,55]` 归一化动作。配置覆盖两/三相机、五种文本长度与 FP16/W8A16，另有三相机 T3 固定样例。

## 配置说明

<a id="选择输入配置"></a>
<a id="weight-recipes"></a>
<a id="配置与权重对应关系"></a>

T 表示 `lang_masks` 中有效语言 token 的数量。所有模板使用 TP8、十步去噪、36 层 Action，以及 32 个路由专家中的 Top4 加共享专家。

| 输入 | FP16 基准 | W8A16 变体 | 权重获取 |
|---|---|---|---|
| 两相机 T16 | [FP16](2cam_t16/fp16.toml) | [W8A16](2cam_t16/w8a16.toml) | [共用 6B FP32 权重](#weights) |
| 两相机 T32 | [FP16](2cam_t32/fp16.toml) | [W8A16](2cam_t32/w8a16.toml) | [共用 6B FP32 权重](#weights) |
| 两相机 T64 | [FP16](2cam_t64/fp16.toml) | [W8A16](2cam_t64/w8a16.toml) | [共用 6B FP32 权重](#weights) |
| 两相机 T128 | [FP16](2cam_t128/fp16.toml) | [W8A16](2cam_t128/w8a16.toml) | [共用 6B FP32 权重](#weights) |
| 两相机 T160 | [FP16](2cam_t160/fp16.toml) | [W8A16](2cam_t160/w8a16.toml) | [共用 6B FP32 权重](#weights) |
| 三相机 T16 | [FP16](3cam_t16/fp16.toml) | [W8A16](3cam_t16/w8a16.toml) | [共用 6B FP32 权重](#weights) |
| 三相机 T32 | [FP16](3cam_t32/fp16.toml) | [W8A16](3cam_t32/w8a16.toml) | [共用 6B FP32 权重](#weights) |
| 三相机 T64 | [FP16](3cam_t64/fp16.toml) | [W8A16](3cam_t64/w8a16.toml) | [共用 6B FP32 权重](#weights) |
| 三相机 T128 | [FP16](3cam_t128/fp16.toml) | [W8A16](3cam_t128/w8a16.toml) | [共用 6B FP32 权重](#weights) |
| 三相机 T160 | [FP16](3cam_t160/fp16.toml) | [W8A16](3cam_t160/w8a16.toml) | [共用 6B FP32 权重](#weights) |
| 三相机 T3 固定样例 | [FP16](3cam_t3/fp16.toml) | [W8A16](3cam_t3/w8a16.toml) | [共用 6B FP32 权重](#weights) |

全部模板共用一份官方 FP32 权重，相机数和文本长度只改变输入与执行布局。`lingbot2.precision="fp16"` 在加载时转 FP16；`"w8a16"` 对同一源权重量化下文列出的投影。

<a id="fp16-基准做了什么"></a>

### FP16 基准

以同一相机数、同一文本长度下的 `fp16.toml` 为基准。Vision、Text、Action 的设备投影权重和主路径激活为 FP16，router、Action IO 与条件计算保留浮点路径。

这些模板已开启以下优化，W8 配方共用：

| 阶段 | 主要执行设置 |
|---|---|
| Vision | 多相机 batch；patch embedding 在设备；融合 merger、SPM RoPE；图像特征直接写入前缀缓冲 |
| Prefill | 复用前缀准备缓冲；Vision/Prefill fast replay；VLM replay；K/V 使用对齐后的写入路径 |
| Action | 稀疏 Top4 专家执行；十步去噪图内展开；专家 replay 与 AdaRMS 广播融合；FP16 Top4 路由 |
| Graph | 保留 Vision/Prefill/Action Graph；`instruction_residency="auto"` 按 SDK 和程序条件选择快照重放 |

`instruction_residency="off"` 关闭快照重放，仍使用普通 Graph，模型算术相同。实际策略和选择原因写入运行摘要。

<a id="w8a16-相对-fp16-的增量"></a>

### W8A16 的增量

同一输入目录下，两份 TOML 的计算配方只改变 `lingbot2.precision`，输出目录随之改变。runner 据此设置四类 W8 开关，从 `0` 改为 `1`：

| 开关，均带 `RPU_LINGBOT2_` 前缀 | 量化的模块 |
|---|---|
| `VISION_W8A16` | Vision block 的 Q/K/V/O、FC1/FC2 |
| `PREFILL_W8A16` | Text 每层 Q/K/V/O、Gate/Up/Down |
| `BASE_W8A16` | Action Attention 的 Q/K/V/O，以及共享专家 Gate/Up/Down |
| `EXPERT_W8A16` | 路由专家的 Gate/Up/Down |

量化采用逐输出通道 INT8 权重、FP16 scale、FP16 激活。路由专家的 Gate/Up 为每个专家分别保存通道 scale；Down 的通道 scale 在专家间共享。Vision patch/merger、Action router/IO/条件投影、norm/bias/embedding 继续使用浮点。Top4 的专家数量和路由算法保持相同。

还有一项执行选择随精度变化：`RPU_WALL_OSS_FAST_REPLAY` 在 FP16 为 `1`，W8 为 `0`；Vision、Prefill、Expert 自身的 replay 开关在两档均开启。稀疏专家权重分别按 FP16 或 INT8 物理布局打包，W8 配方仍走稀疏 Top4 路径。

量化只产生内存中的运行资产，无离线导出步骤。实现见 [runner 的配方映射](../../../runners/lingbot2/optimized.py)、[Text 与 Vision 安装](../../../../python/rpu_backend/adapters/lingbot_vla_v2/runtime.py)、[Action 权重转换](../../../../python/rpu_backend/adapters/lingbot_vla_v2/convert.py)。

## 准备与运行

<a id="weight-sources"></a>
<a id="weights"></a>
<a id="权重准备"></a>

### 获取共用权重

源权重为 [robbyant/lingbot-vla-v2-6b](https://huggingface.co/robbyant/lingbot-vla-v2-6b)，固定 revision `11c703bf6a5c1f45b3b69168482da11fdbba53d7` 的主要 safetensors 存储为 FP32。官方源码见 [robbyant/lingbot-vla-v2](https://github.com/robbyant/lingbot-vla-v2)。

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
hf download robbyant/lingbot-vla-v2-6b \
  --revision 11c703bf6a5c1f45b3b69168482da11fdbba53d7 \
  --local-dir "$RPU_MODEL_CACHE/lingbot-vla-v2-6b"
```

按所选输入准备匹配训练 YAML；模型文件、YAML 和 case 中的掩码应配套。本目录直接读取 `input.checkpoint`；[exact-Z2](../legacy/exact_z2/README.md) 使用另一层级的 registry 路径。

### 绑定权重、训练配置与输入

| 资产 | 字段 | 模板中的路径 |
|---|---|---|
| 完整 checkpoint，含 config、processor/tokenizer | `input.checkpoint` | `${RPU_MODEL_CACHE}/lingbot-vla-v2-6b` |
| 配对训练 YAML | `input.training_config` | `${LINGBOT2_TRAINING_CONFIG}` |
| 两/三相机 T16/32/64/128/160 observation | `input.observation` | 按相机数与 T 命名，如 `${LINGBOT2_2CAM_T16_OBSERVATION}` |
| 三相机 T3 固定样例目录 | `input.case` | `${LINGBOT2_PUBLIC_CASE}` |

训练 YAML 决定对齐和注意力掩码；由所选输入的交付包或运行记录取得配对 YAML 和 observation/case。环境变量指向已有文件，路径中的 `~` 与变量在执行时展开。

<a id="quantization"></a>
<a id="使用"></a>

### 运行

在仓库根目录、已安装项目依赖的设备环境执行。每种配置用新进程运行。

<a id="fp16-loading"></a>

#### FP16：三相机 T3 固定样例

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
export LINGBOT2_TRAINING_CONFIG=/path/to/matching-training.yaml
export LINGBOT2_PUBLIC_CASE=/path/to/frozen-3cam-t3-case
cp examples/configs/lingbot/v2/3cam_t3/fp16.toml /tmp/lingbot2_fp16.toml
# 修改 run.output_dir 为本次输出目录。
vi /tmp/lingbot2_fp16.toml
python examples/run.py --config /tmp/lingbot2_fp16.toml --check-config
python examples/run.py --config /tmp/lingbot2_fp16.toml
```

#### W8A16：同一三相机 T3 输入

复用上面的 checkpoint、YAML 和 case，只替换模板：

```bash
cp examples/configs/lingbot/v2/3cam_t3/w8a16.toml /tmp/lingbot2_w8.toml
vi /tmp/lingbot2_w8.toml
python examples/run.py --config /tmp/lingbot2_w8.toml
```

<a id="当前-v2-w8两相机-t16-observation"></a>

#### W8A16：两相机 T16 observation

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
export LINGBOT2_TRAINING_CONFIG=/path/to/matching-training.yaml
export LINGBOT2_2CAM_T16_OBSERVATION=/path/to/real-2cam-t16-observation.pt
cp examples/configs/lingbot/v2/2cam_t16/w8a16.toml /tmp/lingbot2_2cam_t16_w8.toml
vi /tmp/lingbot2_2cam_t16_w8.toml
python examples/run.py --config /tmp/lingbot2_2cam_t16_w8.toml
```

其他输入选择配置表中的模板，并导出其中 `input.observation` 对应的变量。运行所需的 main/expansion 两份 REF 由 `release/runtime-dependencies.json` 绑定。安装独立 runtime 包后，设置 `RPU_KERNEL_LIB_PATH` 与 `RPU_KERNEL_LIB_PATH_EXPANSION` 指向配套文件；示例会校验两份文件的 SHA-256。准备方法见[算子库说明](../../../../docs/runtime_assets.md)。

<a id="input-modes"></a>
<a id="输入方式"></a>

### 输入方式

`[input]` 在 `case`、`observation`、`images` 中选择一种：

- `case`：目录含 `case.json`，绑定输入/参考张量及校验和，支持 `input.safetensors` / `golden.safetensors`。参考对应十步去噪、动作后处理前的完整 55 维输出与配对掩码。
- `observation`：`.pt` 或 `.safetensors` CPU 张量字典，字段见下表；这种输入只提供观测，不附带 CPU 参考。
- `images`：两/三张图片、真实 `instruction` 和 1–55 维有限 `state`。状态补零到 55 维；使用 CPU FP32 resize 与 checkpoint processor/tokenizer，在模型加载前准备输入。可选 `noise` 文件，否则由 `run.seed` 生成一次噪声并复用。

observation 恰好包含以下七个字段：

| 字段 | CPU 类型与形状 |
|---|---|
| `images` | Float32 `[2或3,256,1536]`，由 256×256 相机图片生成；case 中别名为 `pixel_values` |
| `image_grid_thw` | Int64 `[cameras,3]`，每行 `[1,16,16]` |
| `img_masks` | Boolean `[cameras]`，均为 true |
| `lang_tokens`、`lang_masks` | Int64 / Boolean `[1,L]`，`1 <= L <= max_language_tokens`，至少一个有效 token |
| `state` | Float32 `[1,55]` |
| `noise` | Float32 `[1,50,55]`；case 中别名为 `initial_noise` |

原始图片输入可用下面字段替换模板中的 `case` 或 `observation`。实际 token 数未知时省略 `expected_language_tokens`：

```toml
images = ["/path/to/front.png", "/path/to/wrist.png"]
instruction = "Move the red object into the tray."
state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
cameras = 2
```

<a id="相机数文本长度与填充"></a>

### 常用参数

| 字段 | 模板值 | 省略时 | 范围与关联 |
|---|---|---|---|
| `lingbot2.precision` | `"fp16"` / `"w8a16"` | 必填 | 模块差异见配置说明 |
| `lingbot2.instruction_residency` | `"auto"` | `"auto"` | `auto/off`，控制快照重放 |
| `input.checkpoint/training_config` | 上表环境变量 | 必填 | 配对权重与 YAML |
| `input.case/observation/images` | 按模板填写其一 | 三者选一 | `images` 还需 `instruction` 与 `state` |
| `input.cameras` | 2 或 3 | 从实际输入取得 | 填写时校验相机数，所有相机 mask 有效 |
| `input.expected_language_tokens` | 目录中的 T 值 | 只要求至少一个有效 token | 填写时校验 `lang_masks` 的有效数量 |
| `input.max_language_tokens` | T128/T160 为 160，其余未配置 | 72 | 1–160，限制语言存储宽度 |
| `rpu_execution.prefill.padding_budget` | T128/T160 和三相机 T64 为 63，其余未配置 | 15 | 非负整数；AUTO 在预算内选物理前缀 |
| `rpu_execution.model.num_cores` | 8 | 8 | 固定 TP8 |
| `cpu.threads` | 1 | 1 | 正整数，CPU 线程数 |
| `run.warmup/runs/seed` | 2 / 5 / 0 | 2 / 5 / 0 | warmup≥1、runs≥1、seed≥0 |
| `run.output_dir` | 模板对应结果目录 | 必填 | 每次运行使用独立目录 |

语言存储上限、有效 token 数和物理前缀分别控制存储、语义与执行布局。超出语言上限的指令会报错；增加物理填充不增加有效 token，也不改变 mask。两种精度的同输入模板使用相同填充设置。日志、诊断和公共执行字段见[公共参数说明](../../PARAMETERS.md)。

## 输出与重放

`run.output_dir` 保存 `actions.pt`、执行计划和 Graph 状态。默认预热 2 次、正式运行 5 次；准备阶段另有五次调用。准备后 Graph 身份保持稳定，重放及提交计数递增；OFF 或回退路径的 prepared miss 可递增。`--profile` 和 `--hwperf` 各在常规运行后增加一次诊断调用；跟踪文件另存。`--check-config` 只检查 TOML，不读取模型或输入资产。

实验性 Prefill KV 配对、QK-norm/RoPE 融合、稀疏 KV 镜像和路由替换保持关闭。
