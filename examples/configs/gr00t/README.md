# GR00T-N1.7 配置

[配置索引](../CONFIGURATIONS.md) · [准备与运行](#使用)

本目录运行 GR00T-N1.7 DROID 策略。四份模板覆盖三相机 224/256 分辨率与 FP16/W8A16 两种精度；共同使用 B1、TP8、40 步动作、默认 4 步去噪。

## 配置说明

<a id="weight-recipes"></a>
<a id="配置与权重对应关系"></a>

| 图片尺寸 | FP16 基准 | W8A16 变体 | 权重获取 |
|---|---|---|---|
| 三图 224×224 | [fp16.toml](n1_7_droid/224/fp16.toml) | [w8a16.toml](n1_7_droid/224/w8a16.toml) | [共用 GR00T 父权重与 Qwen 配置文件](#weights) |
| 三图 256×256 | [fp16.toml](n1_7_droid/256/fp16.toml) | [w8a16.toml](n1_7_droid/256/w8a16.toml) | 同一组资产，更换匹配分辨率的 observation |

四份模板复用同一 BF16 父权重，精度在加载时选择，W8 不生成单独的权重目录。

<a id="fp16-基准做了什么"></a>

### FP16 基准

以 [224 FP16](n1_7_droid/224/fp16.toml) 为基准，执行顺序为 Vision/Text 主干 → VL encoder → DiT 动作去噪。投影权重和设备主路径激活为 FP16；CPU 预处理和输出快照各用自己的浮点类型。

模板的 `[rpu_execution]` 留空，runner 根据三相机配置自动应用以下优化，FP16 和 W8 共用：

| 部分 | 已开启的设置与作用 |
|---|---|
| Vision | 多相机 batch、融合 merger、设备 RoPE 路径 |
| Text | partial MRoPE、位置数据缓存、KV cache 复用及 fast replay |
| VL encoder / DiT | 合法布局使用 SPM 内 K/V，减少 DDR 中转；跨注意力 K/V 与 V 转置结果在首步刷新，后续去噪步复用 |
| 条件计算 | 加载时预计算 AdaLN 与固定时间投影表；推理时按去噪步读取 |
| 去噪与通信 | 多步去噪展开在 Graph 中，使用 XOR3 all-gather 完成 Attention/MLP 残差通信 |

前缀容量默认按真实 observation 长度向上 16 对齐，最小 256、最大 512。`RPU_GR00T_QWEN3VL_SPM_Z1` 在这四份模板的自动策略中为 `0`。

配置到环境的映射见 [config.py](../../runners/gr00t/config.py)，加载与运行见 [runner](../../runners/gr00t/runner.py) 和 [runtime.py](../../../python/rpu_backend/adapters/gr00t/runtime.py)。

<a id="w8a16-相对-fp16-的增量"></a>

### W8A16 的增量

同分辨率下，`example.dtype` 从 `"fp16"` 改为 `"w8a16"`，同时切换对应 `profile_id` 和输出目录。runner 据此开启 `RPU_GR00T_W8A16`，在加载时将下列投影改为逐输出通道 INT8 权重、FP16 scale、FP16 激活：

| 模块 | 改为 W8 的投影 | 仍为浮点的部分 |
|---|---|---|
| Text | Q/K/V/O、Gate/Up/Down | embedding、norm、bias |
| Vision | 每个 block 的 Q/K/V/O、FC1/FC2 | patch embedding、merger、norm、bias |
| VL encoder | Q/K/V/O、两层 FFN | norm、bias |
| DiT | Q/K/V/O、FF1/FF2、AdaLN 条件投影 | norm、bias |
| 动作连接层 | `se2/ae2a/ae2t/ae3/dec1/dec2/po1/po2` | `se1/ae1`，其输入维补齐后为 K144 |

动作连接层名称对应状态编码、动作/时间编码、动作解码与输出调制。AdaLN 表仍预计算为 FP16，但计算该表时使用所安装的 W8 投影。

W8 模板还自动开启两项执行优化：

| 自动开关，均带 `RPU_GR00T_` 前缀 | FP16 → W8 | 作用 |
|---|---|---|
| `FF1_RESIDENT_W8` | `0` → `1` | 布局容量允许时，首步将第 0 层 FF1 权重载入 SPM，后续去噪步复用；实际选择由完整 SPM 预算决定 |
| `SELF_QKV_MULTIBASE` | `0` → `1` | Self-Attention 的 Q/K/V 三次投影改为一次联合启动，保留独立输出和原 ACC16 算术 |

FP16 与 W8 同时改变量化和这两项执行策略。开关细节见[运行参数说明](../../../docs/runtime_config.md)。

## 准备与运行

<a id="weight-sources"></a>
<a id="weights"></a>
<a id="quantization"></a>
<a id="权重准备"></a>

### 获取共用权重与配置

在仓库根目录执行，四份配置只需下载一次：

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
hf download nvidia/GR00T-N1.7-3B \
  --revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
  --local-dir "$RPU_MODEL_CACHE/GR00T-N1.7-3B"
hf download Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --include '*.json' '*.txt' '*.jinja' \
  --local-dir "$RPU_MODEL_CACHE/Qwen3-VL-2B-Instruct"
```

[NVIDIA 父权重](https://huggingface.co/nvidia/GR00T-N1.7-3B)的上述版本存储为 BF16。模型源码见 [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)。FP16 在加载时转换浮点类型；W8 使用同一目录，在安装模型时完成量化。

| 资产 | 用途 | 配置字段与路径 |
|---|---|---|
| NVIDIA GR00T-N1.7-3B | 提供主干和动作模块的实际权重 | `example.registry_alias="gr00t-n1d7-3b"` → `$RPU_MODEL_CACHE/GR00T-N1.7-3B` |
| Qwen3-VL-2B-Instruct 的配置与 tokenizer | 提供架构配置、processor 与分词文件 | `input.qwen3vl_registry_alias="qwen3-vl-2b"` → `$RPU_MODEL_CACHE/Qwen3-VL-2B-Instruct` |
| 真实 CPU observation | 图片、指令和状态的预处理结果 | `input.observation` 指向 `.pt` 文件 |

Qwen 目录只用于配置和预处理；策略的 `backbone.model.*` 与 Action 权重均从 GR00T checkpoint 读取。

<a id="使用"></a>

### 输入与运行

在已安装项目及模型依赖的 设备环境中，以 224 W8 为例：

```bash
cp examples/configs/gr00t/n1_7_droid/224/w8a16.toml /tmp/gr00t_w8.toml
# input.observation 填真实三图 224/T32 文件，run.output_dir 填本次目录。
vi /tmp/gr00t_w8.toml
python examples/run.py --config /tmp/gr00t_w8.toml --check-config
python examples/run.py --config /tmp/gr00t_w8.toml
```

FP16 或 256 输入从对应模板开始，权重目录不变。

observation 为 CPU tensor 字典，也接受 `{"inputs": 字典, "meta": ...}` 包装。必需字段为 `input_ids`、`attention_mask`、`pixel_values`、`image_grid_thw`、`state`。输入 B1、无 token padding；一个 user 消息先放三图再放 instruction，末尾带 assistant generation prompt。图片用匹配的 Qwen3-VL processor 预处理，状态为有限浮点 `[1,1,132]`。可选的 `mm_token_type_ids`、`embodiment_id` 与实际输入一致。

<a id="分辨率文本长度和执行参数"></a>

### 分辨率与文本长度

256×256 模板相对同精度 224 模板改变 `input.image_size`、`input.observation` 和输出目录；量化范围、父权重与去噪步数相同。尺寸和文本长度改变真实计算量，需要同时准备匹配的 observation。

| 项目 | 224 配置 | 256 配置 |
|---|---|---|
| 每图 `image_grid_thw` | `[1,14,14]` | `[1,16,16]` |
| 三图 `pixel_values` 形状 | `[588,1536]` | `[768,1536]` |
| T32 / T64 / T128 的真实前缀 P | 193 / 225 / 289 | 238 / 270 / 334 |

T 是纯 instruction token 数，不含图像和消息模板；P 是完整前缀长度。

### 常用参数

| 字段 | 模板值 | 省略时 | 范围与关联 |
|---|---|---|---|
| `input.observation` | 对应尺寸/T32 的占位路径 | 必填 | 准备与分辨率、文本长度、状态匹配的 CPU 字典 |
| `input.qwen3vl_registry_alias` | `"qwen3-vl-2b"` | 必填 | 指向上表配置/tokenizer 目录 |
| `input.cameras` | 3 | 三相机模板必填 | 固定 3 |
| `input.image_size` | 224 或 256 | 三相机模板必填 | 与图片 grid 及 `pixel_values` 一致 |
| `input.text_tokens` | 32 | 三相机模板必填 | 32/64/128，更换匹配 observation |
| `input.action_horizon` | 40 | 三相机模板必填 | 固定 40 |
| `input.num_steps` | 4 | 三相机模板必填 | 4 或 10，加载时确定去噪计算量及预计算表 |
| `input.embodiment_id` | 20 | 20 | DROID；observation 中给定时必须一致 |
| `input.max_prefix_len` | 未配置 | `max(256, ceil16(P))` | 显式值覆盖真实 P、为 16 的倍数且不超过 512 |
| `input.seed` | 0 | 0 | 初始去噪噪声种子 |

模板 `cpu.threads=8`、`run.warmup/runs/seed=2/6/0`。这些公共字段的省略值、输出目录和诊断选项见[公共参数说明](../PARAMETERS.md)。

## 输出

输出动作张量与运行摘要，产物保存在 `run.output_dir`。
