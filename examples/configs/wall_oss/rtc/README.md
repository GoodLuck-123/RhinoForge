# Wall-OSS RTC 配置

本组运行客户 WRC `wrc_sf_v728_0_15000`，使用三张 336×336 图片、动作长度 20 和六步去噪。RTC（Real-Time Chunking）通过已有动作前缀与推理延迟衔接相邻动作块。登记状态为 `CONTROLLED_EVALUATION`。

<a id="选择配置"></a>

<a id="配置与权重对应"></a>

<a id="weight-recipes"></a>

## 配置说明

| 配置 | 比较起点 | 权重获取 |
|---|---|---|
| [fp16.toml](fp16.toml) | FP16 基准 | [原始 WRC 交付](#run-fp16) |
| [w8a16.toml](w8a16.toml) | FP16 主投影改 W8，保留固定浮点边界 | [原始 WRC + 配对 W8](#run-rtc-w8) |

<a id="fp16-基准与-w8-增量"></a>

`fp16.toml` 使用 `example.dtype="fp16"`，Vision、Language 和 Action 主投影为 FP16。输入前缀范围为 `[599,640]`，准备 P640、每段 C320 的预填充图；六步去噪和 Graph replay 沿用 RTC 执行路径。模板中的 `RPU_WALL_OSS_RTC_ALLOW_NUMERIC_BLOCKED="1"` 显式选择这组受控配置。

切换 W8 时，`example.dtype` 改为 `"w8a16"`，`profile_id/registry_alias` 同步选择 W8，`input.checkpoint` 改为量化目录并新增 `input.fp16_checkpoint`。Vision block、Language 和 Action decoder 的 Q/K/V/O/Gate/Up/Down 改为逐输出通道 INT8 权重、FP16 scale 和 FP16 激活。

RTC 保留两个固定浮点边界：Vision block17 Down 和 Language layer2 Down 读取原始 FP16 权重，并以 ACC32 累加、BF16 输出；其余相关激活保持 A16。patch/merger、embedding、norm/bias、动作预处理和输出回投保持浮点，设备部分 FP16、CPU 归一化 FP32。W8 文件中额外量化的动作预处理副本不用于这条路径。

这组配置的累加边界由 RTC 实现固定，base 的分阶段 ACC 参数属于另一入口。权重安装发生在加载时，推理阶段复用已安装权重和准备好的图。实现见 [runner](../../../runners/wall_oss/runner.py)、[适配器](../../../../python/rpu_backend/adapters/wall_oss/__init__.py)和 [W8 转换器](../../../../python/rpu_backend/quant/convert_wall_oss_w8a16.py)。

<a id="准备权重并运行"></a>

<a id="quantization"></a>

<a id="weight-sources"></a>

<a id="weights"></a>

## 准备与运行

<a id="run-fp16"></a>

### 获取原始 WRC

原始权重、normalizer、processor、源码和输入从同一 WRC 交付获取。公开 [Wall-OSS 0.5](../base/README.md#weights) 是不同资产；本组转换的 `--src` 始终使用 WRC 原始目录。W8 目录由本仓库转换器生成，FP16 目录继续提供浮点模块和预处理权重。

将交付原始目录记为 `WRC_SRC`。FP16 的 `input.checkpoint` 直接填写该目录。

```bash
WRC_SRC=/absolute/path/model-cache/wrc_sf_v728_0_15000
```

<a id="run-rtc-w8"></a>

### 转换 W8

```bash
WRC_W8=/absolute/path/model-cache/wrc_v728_w8a16
python python/rpu_backend/quant/convert_wall_oss_w8a16.py \
  --src "$WRC_SRC" --dst "$WRC_W8"
```

W8 模板的 `input.checkpoint` 填 `WRC_W8` 实际路径，`input.fp16_checkpoint` 填 `WRC_SRC` 实际路径；保留两个配对目录。

<a id="使用"></a>

### 绑定输入并运行

```bash
WRC_CONFIG=examples/configs/wall_oss/rtc/w8a16.toml
cp "$WRC_CONFIG" /tmp/wall_rtc.toml
# 按上一节填写 checkpoint；FP16 改选 fp16.toml，只填原始 checkpoint。
# input.images/instruction/proprioception = 配对输入；run.output_dir = 结果目录。
vi /tmp/wall_rtc.toml
python examples/run.py --config /tmp/wall_rtc.toml
```

<a id="输入与执行参数"></a>

### 输入与常调参数

填写三张图片、指令与 26 维 `proprioception`。相机顺序为 `face_view`、`left_wrist_view`、`right_wrist_view`。每路图片经 processor 后为 576 个 patch，三图共 1728 个；336×336 对应 grid `[1,24,24]`。

| 字段 | 模板值 | 可选范围 | 修改时的关联 |
|---|---|---|---|
| `input.images` | 三个占位路径 | 三张实际图片 | 处理后满足每图 576 patch |
| `input.instruction` | `Move the object to the target.` | 非空文本 | 与图片、状态共同决定实际前缀 |
| `input.prefix_length_range` | `[599,640]` | 本组固定范围 | 约束图像 token 展开后的长度；模板输入实际为 622 |
| `input.inference_delay` | 0 | 0–10 的整数 | 非零时同时提供 `rtc_prefix_actions` |
| `input.rtc_prefix_actions` | 省略 | 有限浮点 `[L,26]` 或 `[1,L,26]` | 使用物理动作空间，delay≤L≤20；delay=0 时省略 |
| `run.warmup/runs/seed` | 2 / 3 / 0 | 预热≥0、采样≥1、seed≥0 | 按所需运行次数调整 |
| `cpu.threads` | 模板未指定 | 正整数 | 省略时继承运行环境 |

公共字段见[参数说明](../../PARAMETERS.md)，运行库见[算子库说明](../../../../docs/runtime_assets.md)。更换冷配置后使用新进程。
