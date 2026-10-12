# Wall-OSS 快递分拣配置

本组使用客户 WRC `wrc_sf_v728_0_15000` 和配对分拣输入，固定 batch1、TP8、三图 448×252、动作长度 20、六步去噪。按输入文件重放完整分拣请求。

<a id="选择配置"></a>

<a id="配置与权重对应"></a>

<a id="weight-recipes"></a>

## 配置说明

| 配置 | 相对上一档的变化 | 权重获取 |
|---|---|---|
| [fp16.toml](fp16.toml) | FP16 基准 | [原始 WRC 交付](#run-fp16) |
| [w8a16.toml](w8a16.toml) | 主投影 W8，保留固定浮点边界 | [原始 WRC + W8](#run-sorting-w8) |
| [w8a16_action_w4a16_g32.toml](w8a16_action_w4a16_g32.toml) | Action expert1 改为 G32 INT4，调整驻留和 KV 路径 | [再准备动作 W4 目录](#客户-wrc-分拣-w8--action-w4-pgrp) |

<a id="fp16-基准与配置增量"></a>

### FP16 基准

`wall_oss_sorting.precision="fp16"` 已包含设备 patch embedding、三图处理、merger 融合、Vision 行复用、部分 MRoPE、预填充 KV 直接写入、SiLU×Mul 融合、六步去噪展开、Action 权重驻留与 KV 复用，以及 Graph replay。实际前缀范围为 599–640，图准备为 P640、C320×2。

三档均保留 RTC 的混合浮点边界：Vision block17 Down 与 Language layer2 Down 使用原始 FP16 权重、ACC32 累加和 BF16 输出。patch/merger、norm/bias、embedding、动作预处理/输出回投保持浮点，设备部分 FP16、CPU 部分 FP32。这组入口不设置 base 的分阶段 ACC 开关。

### W8A16：三个组件主投影改为 INT8

相对 FP16，`wall_oss_sorting.precision` 改为 `"w8a16"`，新增 `input.w8_checkpoint`，`input.fp16_checkpoint` 不变。Vision、Language 和 Action 的 Q/K/V/O/Gate/Up/Down 改为 W8A16，保留上述两个浮点 Down 边界。W8 使用逐输出通道 INT8 权重、FP16 scale 和激活。

运行器同时将 FP16 专用传输与 MLP 路径切换为量化路径；缓存、融合去噪和 Action 权重驻留仍启用。转换器离线量化的动作预处理二维权重不在运行中使用，原始目录继续提供这些浮点项。

### 动作 W4：在 W8 上替换动作专家

`wall_oss_sorting.precision` 改为 `"w8a16_action_w4a16"`，新增 `input.action_w4_checkpoint`。只从该目录加载 Action expert1 的 Q/K/V/O/Gate/Up/Down，使用 K 维每 32 个值一组的 INT4 权重和 FP16 scale；激活仍为 FP16。Vision/Language 继续从 W8 目录加载。

这项配置使用整数 W4-pgrp。转换器也生成 Language 的 W4 投影，但本入口只从该目录读取 Action。运行器还关闭 Action 权重驻留、KV owner fanout 和对应的浮点复用路径。

实现见[分拣入口](../../../runners/wall_oss/sorting.py)、[执行设置](../../../runners/wall_oss/sorting_engine.py)、[W8 转换器](../../../../python/rpu_backend/quant/convert_wall_oss_w8a16.py)及 [W4 转换器](../../../../python/rpu_backend/quant/convert_wall_oss.py)。使用下面的完整命令生成 G32 权重。

<a id="准备权重并运行"></a>

<a id="quantization"></a>

<a id="weight-sources"></a>

<a id="weights"></a>

## 准备与运行

<a id="run-fp16"></a>

### 获取原始 WRC

模型、normalizer、源码和分拣 batch 从同一 `wrc_sf_v728_0_15000` 客户交付获取。公开 [Wall-OSS base](../base/README.md#weights) 使用另一套权重。三档的 `input.fp16_checkpoint` 均填写原始目录，`input.batch` 填配对分拣输入。

```bash
WRC_SRC=/absolute/path/model-cache/wrc_sf_v728_0_15000
```

<a id="run-sorting-w8"></a>

### 转换 W8

```bash
WRC_W8=/absolute/path/model-cache/wrc_v728_w8a16
python python/rpu_backend/quant/convert_wall_oss_w8a16.py \
  --src "$WRC_SRC" --dst "$WRC_W8"
```

W8 和动作 W4 配置都将 `input.w8_checkpoint` 填为 `WRC_W8` 实际路径。

<a id="客户-wrc-分拣-w8--action-w4-pgrp"></a>

### 转换动作 W4

动作 W4 在配置比较上接续 W8，转换仍直接读取原始 `WRC_SRC`：

```bash
WRC_W4=/absolute/path/model-cache/wrc_v728_action_w4a16_pgrp
python -m rpu_backend.quant.convert_wall_oss \
  --src "$WRC_SRC" --dst "$WRC_W4" --group-size 32
```

`input.action_w4_checkpoint` 填 `WRC_W4` 实际路径，同时保留原始和 W8 两个目录。转换器生成两个 decoder 的 G32 INT4 文件。

<a id="使用"></a>

### 绑定输入并运行

```bash
SORTING_CONFIG=examples/configs/wall_oss/sorting/w8a16.toml
cp "$SORTING_CONFIG" /tmp/wall_sorting.toml
# input.fp16_checkpoint = WRC_SRC 实际路径。
# W8/动作 W4 另填 w8_checkpoint；动作 W4 再填 action_w4_checkpoint。
# input.batch = 配对分拣 batch；run.output_dir = 本次结果目录。
vi /tmp/wall_sorting.toml
python examples/run.py --config /tmp/wall_sorting.toml
```

<a id="输入与执行参数"></a>

### 输入与常调参数

batch 为 CPU 字典：uint8 `images [3,252,448,3]`，依次为 face/left/right；非空 `instruction`；有限浮点 `state [26]`、`noise [1,20,26]`；bool `state_mask/dof_mask [26]`，前 20 项为 true。state 和动作前缀使用模型的物理空间。图片和文本处理后的实际前缀处于 599–640。

| 字段 | 模板值或入口默认 | 可选范围 | 修改时的关联 |
|---|---|---|---|
| `input.batch` | 占位路径 | 配对实际 batch | 保持上述图像、状态、mask 和噪声形状 |
| `input.inference_delay` | 0 | 0 / 1 / 3 / 10 | 非零时 batch 另有浮点 `prefix_actions [20,26]` |
| `cpu.threads` | 未指定 | 正整数 | 省略时继承运行环境 |
| `run.warmup/runs` | 2 / 6 | 预热至少 1，采样至少 1 | 按所需运行次数调整 |

公共字段见[参数说明](../../PARAMETERS.md)，运行库见[算子库说明](../../../../docs/runtime_assets.md)。每种冷配置用新进程。
