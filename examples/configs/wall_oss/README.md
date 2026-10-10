# Wall-OSS 配置
[配置索引](../CONFIGURATIONS.md) · [使用](#使用)

<a id="weights"></a>
<a id="weight-sources"></a>
## 权重来源

官方文件的存储 dtype 与本仓库运行时的 FP16/W8 等选项分开列示；`FP16` 配置不表示官方发布文件就是 FP16。

| 身份 | 名称与发布者 / 下载渠道 | 文件存储精度 | 本地目录与配置绑定 |
|---|---|---|---|
| 官方公开发布 | X Square Robot：[wall-oss-0.5](https://huggingface.co/x-square-robot/wall-oss-0.5/tree/b7dc23a82dbf70a65859c9fdd596703234597028)；[官方源码](https://github.com/X-Square-Robot/wall-x) | BF16 + FP32（官方 safetensors 元数据） | `wall-oss-0.5`；完整模型、processor/tokenizer、normalizer |
| 当前公开基础模型父 | base / legacy 使用上述公开模型身份；固定 source revision 与文件校验和 | 原文件浮点；运行时转 FP16 / FP32 | base：`input.fp16_checkpoint`；legacy：`$RPU_MODEL_CACHE/wall-oss-0.5` |
| 当前客户 WRC 父 | 原交付方：`wrc_sf_v728_0_15000`，未核实官方公开下载；不可由公开 base 替代 | 原始 dtype 由交付清单指定 | RTC：`input.checkpoint` / `input.fp16_checkpoint`；sorting：`input.fp16_checkpoint` |
| 公开量化 / 本仓库转换 | 使用本仓库转换配方；下列 CLI 从各自原始父转换 | W8：2D linear INT8，浮点保留项 FP16；W4：两个 decoder expert 的投影均为 INT4 | 公开：`wall-oss-0.5-w8a16` / `-w4a16` / `-w4a16-pgrp`；客户：`wrc_v728_w8a16` / `wrc_v728_action_w4a16_pgrp` |

<a id="weight-recipes"></a>
## 模块精度与配方

W/A 与 ACC 定义见[精度记法](../WEIGHTS.md#precision-notation)。 NVFP4A16 为 E2M1 四位权重、沿 K 每 16 个值一组的 E4M3FN 八位 scale、FP32 tensor scale，激活仍为 FP16；

| 配方与模板 | 加载后 Vision / Language | 加载后 Action / IO 与保留浮点部分 | 所需源权重、转换方式与完整命令 |
|---|---|---|---|
| base FP16：[两图224](base/2cam/224/fp16.toml) / [两图256](base/2cam/256/fp16.toml) / [三图224](base/3cam/224/fp16.toml) / [三图256](base/3cam/256/fp16.toml) | Vision、Language expert0 主投影 W16A16 | Action expert1 主投影 W16A16；patch/merger、norm/bias、embedding、动作预处理/输出回投保持浮点：设备路径 FP16，host 预处理/归一化 FP32；不使用 W8 文件中 action_preprocessor 的 INT8 副本 | 公开原父；直接加载；[完整命令](base/README.md#run-fp16) |
| base W8：[两图224](base/2cam/224/w8a16.toml) / [两图256](base/2cam/256/w8a16.toml) / [三图224](base/3cam/224/w8a16.toml) / [三图256](base/3cam/256/w8a16.toml) | Vision block 与 Language expert0 Q/K/V/O/Gate/Up/Down W8A16 | Action expert1 相同七类投影 W8A16；patch/merger、norm/bias、embedding、动作预处理/输出回投保持浮点：设备路径 FP16，host 预处理/归一化 FP32；不使用 W8 文件中 action_preprocessor 的 INT8 副本 | 公开原父 → 离线 W8；[完整命令](#公开-base-w8a16)，[python/rpu_backend/quant/convert_wall_oss_w8a16.py](../../../python/rpu_backend/quant/convert_wall_oss_w8a16.py) |
| base W8 + 动作 NVFP4：[两图224](base/2cam/224/w8a16_action_nvfp4.toml) / [两图256](base/2cam/256/w8a16_action_nvfp4.toml) / [三图224](base/3cam/224/w8a16_action_nvfp4.toml) / [三图256](base/3cam/256/w8a16_action_nvfp4.toml) | Vision / Language 同 base W8 | Action expert1 Q/K/V/O/Gate/Up/Down 全为 NVFP4A16，包含 K/V；浮点保留项同上 | 公开原父 + 配对 W8；动作加载时 `_load_expert1_decoder()` → `_pack_nvfp4_rpu()`，无 NVFP4 导出目录；[完整命令](#公开-base-w8--action-nvfp4) |
| legacy FP16：[legacy/fp16.toml](legacy/fp16.toml) | Vision / Language 主投影 W16A16 | Action 主投影 W16A16；浮点保留项见下文 | 公开原父；直接加载；[完整命令](legacy/README.md#run-fp16) |
| legacy W8：[legacy/w8a16.toml](legacy/w8a16.toml) | Vision / Language 主投影 W8A16 | Action 主投影 W8A16；浮点保留项见下文 | 公开原父 → 离线 W8；[完整命令](#公开-base-legacy-w8a16)，[python/rpu_backend/quant/convert_wall_oss_w8a16.py](../../../python/rpu_backend/quant/convert_wall_oss_w8a16.py) |
| legacy W4 G0：[legacy/w4a16.toml](legacy/w4a16.toml) | Vision 从配对 W8 目录加载；Language decoder W4A16 | Action decoder W4A16；Q/K/V/O/Gate/Up/Down 逐通道 INT4，FP16 scale，运行时打包 pgrp 布局；浮点保留项见下文 | 公开原父 + W8；离线 `--group-size 0`；[完整命令](#公开-base-legacy-w4逐通道-group0)，[convert_wall_oss.py](../../../python/rpu_backend/quant/convert_wall_oss.py) |
| legacy W4 G32：[legacy/w4a16_pgrp.toml](legacy/w4a16_pgrp.toml) | Vision W8A16；Language Q/K/V/O/Gate/Up/Down 均为 G32 W4A16 | Action decoder 全部七类投影 G32 W4A16；scale FP16；浮点保留项见下文 | 公开原父 + W8；离线 G32；[完整命令](#公开-base-legacy-w4-pgrp混合-group32)，[convert_wall_oss.py](../../../python/rpu_backend/quant/convert_wall_oss.py) |
| RTC FP16：[rtc/fp16.toml](rtc/fp16.toml) | Vision / Language 主投影 W16A16；保留既有混合浮点边界 | Action 主投影 W16A16；浮点保留项见下文 | 客户 WRC 原父；直接加载；[完整命令](rtc/README.md#run-fp16) |
| RTC W8：[rtc/w8a16.toml](rtc/w8a16.toml) | Vision / Language 主投影 W8A16，但 Vision block17 Down 与 Language layer2 Down 保留原父 FP16，使用既有 ACC32→BF16 输出边界，其余相关激活 A16 | Action decoder W8A16；浮点保留项见下文 | 客户 WRC 原父 + 配对离线 W8；[完整命令](#run-rtc-w8)，[python/rpu_backend/quant/convert_wall_oss_w8a16.py](../../../python/rpu_backend/quant/convert_wall_oss_w8a16.py) |
| 分拣 FP16：[sorting/fp16.toml](sorting/fp16.toml) | Vision / Language 主投影 W16A16；保留既有混合浮点边界 | Action 主投影 W16A16；浮点保留项见下文 | 客户 WRC 原父；直接加载；[完整命令](sorting/README.md#run-fp16) |
| 分拣 W8：[sorting/w8a16.toml](sorting/w8a16.toml) | Vision / Language 主投影 W8A16，但 Vision block17 Down 与 Language layer2 Down 保留原父 FP16，使用既有 ACC32→BF16 输出边界，其余相关激活 A16 | Action decoder W8A16；浮点保留项见下文 | 客户 WRC 原父 + 配对离线 W8；[完整命令](#run-sorting-w8)，[python/rpu_backend/quant/convert_wall_oss_w8a16.py](../../../python/rpu_backend/quant/convert_wall_oss_w8a16.py) |
| 分拣 W8 + 动作 W4：[sorting/w8a16_action_w4a16.toml](sorting/w8a16_action_w4a16.toml) | 与分拣 W8 相同，Language 不改读 W4 文件 | 仅 Action expert1 Q/K/V/O/Gate/Up/Down 为 G32 W4A16；FP16 scale；非 NVFP4；浮点保留项见下文 | 客户 WRC 原父 + W8 + G32 W4；只从 W4 目录取动作专家；[完整命令](#客户-wrc-分拣-w8--action-w4-pgrp)，[convert_wall_oss.py](../../../python/rpu_backend/quant/convert_wall_oss.py) |

**各配方的浮点保留项：** patch/merger、norm/bias、embedding、动作预处理/输出回投保持浮点：设备路径 FP16，host 预处理/归一化 FP32；不使用 W8 文件中 action_preprocessor 的 INT8 副本。离线 W8 转换器会量化除 `embed_tokens` 外的全部浮点 2D `.weight`；运行器只取所需 INT8 投影，其余胶水权重仍从配对原父加载。

<a id="quantization"></a>
## 下载与转换

以下命令在仓库根目录执行。先把 `/path/to/...` 换成实际路径；`vi` 这一步编辑刚复制的 TOML，保存后再执行下一行。每个代码块是一个独立精度方案，只执行所需方案；已有同身份量化目录时跳过对应转换行。

下载固定 revision，记录源权重校验和。

<a id="download-parent"></a>
### 下载公开基础模型原始权重

```bash
SRC=/path/to/model-cache/wall-oss-0.5
hf download x-square-robot/wall-oss-0.5 --revision b7dc23a82dbf70a65859c9fdd596703234597028 --local-dir "$SRC"
```

<a id="公开-base-w8a16"></a>
### 公开基础模型 W8A16

```bash
SRC=/path/to/model-cache/wall-oss-0.5  # 原始公开 base 父目录
DST=/path/to/model-cache/wall-oss-0.5-w8a16  # 新 W8 输出目录
python python/rpu_backend/quant/convert_wall_oss_w8a16.py --src "$SRC" --dst "$DST"
cp examples/configs/wall_oss/base/3cam/224/w8a16.toml /tmp/wall_base_w8.toml
# input.fp16_checkpoint = SRC 实际路径；input.w8_checkpoint = DST 实际路径。
# input.batch_template = 匹配的真实 batch 路径模板（保留 {text_tokens} 等占位符）。
# run.output_dir = 本次目录；输入变化时按本页说明处理旧 padding_rows。
vi /tmp/wall_base_w8.toml
python examples/run.py --config /tmp/wall_base_w8.toml
```

<a id="公开-base-w8--action-nvfp4"></a>
### 公开基础模型 W8 + 动作 NVFP4

```bash
SRC=/path/to/model-cache/wall-oss-0.5  # 原始公开 base 父目录
DST=/path/to/model-cache/wall-oss-0.5-w8a16  # 新 W8 输出目录
python python/rpu_backend/quant/convert_wall_oss_w8a16.py --src "$SRC" --dst "$DST"
cp examples/configs/wall_oss/base/3cam/224/w8a16_action_nvfp4.toml /tmp/wall_base_nvfp4.toml
# input.fp16_checkpoint = SRC 实际路径；input.w8_checkpoint = DST 实际路径。
# input.batch_template = 匹配的真实 batch 路径模板（保留 {text_tokens} 等占位符）。
# run.output_dir = 本次目录；输入变化时按本页说明处理旧 padding_rows。
vi /tmp/wall_base_nvfp4.toml
python examples/run.py --config /tmp/wall_base_nvfp4.toml
```

<a id="公开-base-legacy-w8a16"></a>
### 公开基础模型旧版 W8A16

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
SRC="$RPU_MODEL_CACHE/wall-oss-0.5"  # 原始公开 base 父目录
DST="$RPU_MODEL_CACHE/wall-oss-0.5-w8a16"  # W8 registry 输出目录
python python/rpu_backend/quant/convert_wall_oss_w8a16.py --src "$SRC" --dst "$DST"
cp examples/configs/wall_oss/legacy/w8a16.toml /tmp/wall_legacy_w8.toml
# input.images/instruction/proprioception/state_mask = 配对真实输入。
# 在 [run] 添加 output_dir；保留原 registry_alias 和 fp16_registry_alias。
# DST 按上述 registry 目录名解析，不另添加 input.checkpoint。
vi /tmp/wall_legacy_w8.toml
python examples/run.py --config /tmp/wall_legacy_w8.toml
```

<a id="公开-base-legacy-w4逐通道-group0"></a>
### 公开基础模型旧版 W4：逐通道 group0

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
SRC="$RPU_MODEL_CACHE/wall-oss-0.5"  # 原始公开 base 父目录
DST="$RPU_MODEL_CACHE/wall-oss-0.5-w4a16"  # 此精度的 registry 输出目录
python python/rpu_backend/quant/convert_wall_oss_w8a16.py --src "$SRC" --dst "$RPU_MODEL_CACHE/wall-oss-0.5-w8a16"
python -m rpu_backend.quant.convert_wall_oss --src "$SRC" --dst "$DST" --group-size 0
cp examples/configs/wall_oss/legacy/w4a16.toml /tmp/wall_legacy_g0.toml
# input.images = 真实 face/left/right 图片；input.instruction/proprioception/state_mask = 配对输入。
# 在 [run] 添加 output_dir；保留 example.registry_alias 和 input.fp16_registry_alias。
# DST 按 registry 目录名直接解析，无需把它填进不存在的 input.checkpoint 字段。
vi /tmp/wall_legacy_g0.toml
python examples/run.py --config /tmp/wall_legacy_g0.toml
```

<a id="公开-base-legacy-w4-pgrp混合-group32"></a>
### 公开基础模型旧版 W4-pgrp：group32

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
SRC="$RPU_MODEL_CACHE/wall-oss-0.5"  # 原始公开 base 父目录
DST="$RPU_MODEL_CACHE/wall-oss-0.5-w4a16-pgrp"  # 此精度的 registry 输出目录
python python/rpu_backend/quant/convert_wall_oss_w8a16.py --src "$SRC" --dst "$RPU_MODEL_CACHE/wall-oss-0.5-w8a16"
python -m rpu_backend.quant.convert_wall_oss --src "$SRC" --dst "$DST" --group-size 32
cp examples/configs/wall_oss/legacy/w4a16_pgrp.toml /tmp/wall_legacy_g32.toml
# input.images = 真实 face/left/right 图片；input.instruction/proprioception/state_mask = 配对输入。
# 在 [run] 添加 output_dir；保留 example.registry_alias 和 input.fp16_registry_alias。
# DST 按 registry 目录名直接解析，无需把它填进不存在的 input.checkpoint 字段。
vi /tmp/wall_legacy_g32.toml
python examples/run.py --config /tmp/wall_legacy_g32.toml
```

<a id="run-rtc-w8"></a>
### 客户 WRC RTC W8A16

```bash
SRC=/path/to/model-cache/wrc_sf_v728_0_15000  # 客户 WRC 原始目录，不能换成公开 base
DST=/path/to/model-cache/wrc_v728_w8a16  # 新 W8 输出
python python/rpu_backend/quant/convert_wall_oss_w8a16.py --src "$SRC" --dst "$DST"
cp examples/configs/wall_oss/rtc/w8a16.toml /tmp/wall_rtc_w8.toml
# input.checkpoint = DST 实际路径；input.fp16_checkpoint = SRC 实际路径。
# input.images/instruction/proprioception = 配对三图、指令与状态。
# 在 [run] 添加 output_dir；保留原受控 opt-in 和 prefix 范围。
vi /tmp/wall_rtc_w8.toml
python examples/run.py --config /tmp/wall_rtc_w8.toml
```

<a id="run-sorting-w8"></a>
### 客户 WRC 分拣 W8A16

```bash
SRC=/path/to/model-cache/wrc_sf_v728_0_15000  # 原始客户 WRC 父目录
DST=/path/to/model-cache/wrc_v728_w8a16
python python/rpu_backend/quant/convert_wall_oss_w8a16.py --src "$SRC" --dst "$DST"
cp examples/configs/wall_oss/sorting/w8a16.toml /tmp/wall_sorting_w8.toml
# input.fp16_checkpoint = SRC 实际路径；input.w8_checkpoint = DST 实际路径。
# input.batch = 同交付真实分拣 batch；run.output_dir = 本次独立目录。
vi /tmp/wall_sorting_w8.toml
python examples/run.py --config /tmp/wall_sorting_w8.toml
```

<a id="客户-wrc-分拣-w8--action-w4-pgrp"></a>
### 客户 WRC 分拣 W8 + 动作 W4-pgrp

```bash
SRC=/path/to/model-cache/wrc_sf_v728_0_15000  # 原始客户 WRC 父目录
DST=/path/to/model-cache/wrc_v728_w8a16
python python/rpu_backend/quant/convert_wall_oss_w8a16.py --src "$SRC" --dst "$DST"
W4_DST=/path/to/model-cache/wrc_v728_action_w4a16_pgrp
python -m rpu_backend.quant.convert_wall_oss --src "$SRC" --dst "$W4_DST" --group-size 32
cp examples/configs/wall_oss/sorting/w8a16_action_w4a16.toml /tmp/wall_sorting_w4.toml
# input.fp16_checkpoint = SRC 实际路径；input.w8_checkpoint = DST 实际路径。
# input.action_w4_checkpoint = W4_DST 实际路径。
# input.batch = 同交付真实分拣 batch；run.output_dir = 本次独立目录。
vi /tmp/wall_sorting_w4.toml
python examples/run.py --config /tmp/wall_sorting_w4.toml
```

**FP16 无需量化。** 源/量化目录保持兄弟关系，以便 `source` 元数据解析。公开 base 与客户 WRC 不能混用；group0 与 group32 使用不同 scale 配方。转换计算使用 CPU，命令需在匹配的后端宿主环境中执行。

**实现位置**：[python/rpu_backend/quant/convert_wall_oss_w8a16.py](../../../python/rpu_backend/quant/convert_wall_oss_w8a16.py)；[convert_wall_oss.py](../../../python/rpu_backend/quant/convert_wall_oss.py)。W8/W4 CLI 分别调用各模块的 `convert_checkpoint()`。W4 默认 G32，两个 decoder expert 的投影均为 INT4；用 `--group-size 0` 选择逐通道 scale。

**实现位置**：[python/rpu_backend/adapters/wall_oss/action.py](../../../python/rpu_backend/adapters/wall_oss/action.py)；[python/rpu_backend/adapters/wall_oss/llm.py](../../../python/rpu_backend/adapters/wall_oss/llm.py)。Action NVFP4 没有独立导出目录；选对应 TOML 后，加载时由 `_load_expert1_decoder()` 调用 `_pack_nvfp4_rpu()`，Vision/Prefill 读取 W8。

## 选择配置

| 组别 | 配置覆盖 / 区别 | 推荐配置 |
|---|---|---|
| [base](base/README.md) | 开源0.5；2/3图×224/256×FP16/W8/Action NVFP4，共12份；同文件改T/ACC | [3cam/224/fp16](base/3cam/224/fp16.toml) |
| [RTC](rtc/README.md) | 受控 WRC v728 release，FP16/W8；原固定混合累加 | [fp16](rtc/fp16.toml) |
| [sorting](sorting/README.md) | WRC v728快递分拣；FP16/W8/Action W4-pgrp，评价范围受控 | [w8](sorting/w8a16.toml) |
| [legacy](legacy/README.md) | 原registry FP16/W8/decoder W4/W4-pgrp，原profile输入 | 按下级指南显式选择 |

各组分别绑定 checkpoint、输入和精度。RTC 使用固定混合累加；sorting Action 使用 W4-pgrp，base Action 使用 NVFP4。

## 使用

选择上表对应 README，准备其实际输入与配对权重，再复制 TOML、修改资产路径和输出目录：

```bash
python examples/run.py --config /path/to/wall_base.toml --check-config
python examples/run.py --config /path/to/wall_base.toml
```

RTC/sorting显式选择各自 TOML，统一使用上述命令。`--check-config` 只验证配置，实际运行需空闲板卡、匹配环境/native 和既有设备锁。

## 可修改参数与限制

base 的 text length、组件 ACC 与CPU线程在同一文件修改；其余组按各自登记范围修改。输入变化必须真实替换 tensor/图片/文本，不用声明数字替代实际输入。全族只依赖 rhino-ops main/DDR 两份正式 REF，见 [算子库规则](../../../docs/runtime_assets.md)。支持状态见 [模型矩阵](../../../docs/model_support.md)。

<a id="路径迁移"></a>
复用旧格式请显式选择 legacy/RTC 配置。
