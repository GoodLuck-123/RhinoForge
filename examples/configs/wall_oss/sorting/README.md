# Wall-OSS 快递分拣配置
[上级选型](../README.md) · [使用](#使用)

入口：`python examples/run.py --config <file>`；推荐 [w8a16.toml](w8a16.toml)。使用 WRC `wrc_sf_v728_0_15000` 和配对量化权重，与开源 Wall-OSS base 的资产不同。

<a id="weights"></a>
<a id="weight-sources"></a>
## 权重来源

官方文件的存储 dtype 与本仓库运行时的 FP16/W8 等选项分开列示；`FP16` 配置不表示官方发布文件就是 FP16。

| 身份 | 名称与发布者 / 下载渠道 | 文件存储精度 | 本地目录与配置绑定 |
|---|---|---|---|
| 官方公开发布（另一资产） | X Square Robot：[wall-oss-0.5](https://huggingface.co/x-square-robot/wall-oss-0.5/tree/b7dc23a82dbf70a65859c9fdd596703234597028)；[官方源码](https://github.com/X-Square-Robot/wall-x) | BF16 + FP32（官方 safetensors 元数据） | `wall-oss-0.5`；完整模型、processor/tokenizer、normalizer |
| 当前客户 WRC 父 | 原交付方：`wrc_sf_v728_0_15000`，未核实官方公开下载；不可由公开 base 替代 | 原始 dtype 由交付清单指定 | RTC：`input.checkpoint` / `input.fp16_checkpoint`；sorting：`input.fp16_checkpoint` |
| 公开量化 / 本仓库转换 | 使用本仓库转换配方；下列 CLI 从本组原始父转换 | W8：2D linear INT8；W4：两个 decoder expert 的投影均为 INT4；浮点保留项 FP16 | `wrc_v728_w8a16` → `input.w8_checkpoint`；`wrc_v728_action_w4a16_pgrp` → `input.action_w4_checkpoint` |

<a id="weight-recipes"></a>
## 模块精度与配方

W/A 与 ACC 定义见[精度记法](../../WEIGHTS.md#precision-notation)。 W4 G0 是逐通道 INT4，G32 是 K 维每 32 值一组；两者 scale 为 FP16，均不是 NVFP4。

| 配方与模板 | 加载后 Vision / Language | 加载后 Action / IO 与保留浮点部分 | 所需源权重、转换方式与完整命令 |
|---|---|---|---|
| 分拣 FP16：[fp16.toml](fp16.toml) | Vision / Language 主投影 W16A16；保留既有混合浮点边界 | Action 主投影 W16A16；浮点保留项见下文 | 客户 WRC 原父；直接加载；[完整命令](#run-fp16) |
| 分拣 W8：[w8a16.toml](w8a16.toml) | Vision / Language 主投影 W8A16，但 Vision block17 Down 与 Language layer2 Down 保留原父 FP16，使用既有 ACC32→BF16 输出边界，其余相关激活 A16 | Action decoder W8A16；浮点保留项见下文 | 客户 WRC 原父 + 配对离线 W8；[完整命令](#run-sorting-w8)，[python/rpu_backend/quant/convert_wall_oss_w8a16.py](../../../../python/rpu_backend/quant/convert_wall_oss_w8a16.py) |
| 分拣 W8 + 动作 W4：[w8a16_action_w4a16.toml](w8a16_action_w4a16.toml) | 与分拣 W8 相同，Language 不改读 W4 文件 | 仅 Action expert1 Q/K/V/O/Gate/Up/Down 为 G32 W4A16；FP16 scale；非 NVFP4；浮点保留项见下文 | 客户 WRC 原父 + W8 + G32 W4；只从 W4 目录取动作专家；[完整命令](#客户-wrc-分拣-w8--action-w4-pgrp)，[convert_wall_oss.py](../../../../python/rpu_backend/quant/convert_wall_oss.py) |

**各配方的浮点保留项：** patch/merger、norm/bias、embedding、动作预处理/输出回投保持浮点：设备路径 FP16，host 预处理/归一化 FP32；不使用 W8 文件中 action_preprocessor 的 INT8 副本。离线 W8 转换器会量化除 `embed_tokens` 外的全部浮点 2D `.weight`；运行器只取所需 INT8 投影，其余胶水权重仍从配对原父加载。

<a id="quantization"></a>
## 下载与转换

以下命令在仓库根目录执行。先把 `/path/to/...` 换成实际路径；`vi` 这一步编辑刚复制的 TOML，保存后再执行下一行。每个代码块是一个独立精度方案，只执行所需方案；已有同身份量化目录时跳过对应转换行。

<a id="run-fp16"></a>
### FP16：原始浮点父权重直接运行

```bash
cp examples/configs/wall_oss/sorting/fp16.toml /tmp/wall_sorting_fp16.toml
# input.fp16_checkpoint = 原交付 WRC 目录；input.batch = 同交付真实分拣 batch。
# run.output_dir = 本次独立目录；保留原输入和受控范围。
vi /tmp/wall_sorting_fp16.toml
python examples/run.py --config /tmp/wall_sorting_fp16.toml
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

**实现位置**：[python/rpu_backend/quant/convert_wall_oss_w8a16.py](../../../../python/rpu_backend/quant/convert_wall_oss_w8a16.py)；[convert_wall_oss.py](../../../../python/rpu_backend/quant/convert_wall_oss.py)。W8/W4 CLI 分别调用各模块的 `convert_checkpoint()`。W4 默认 G32，两个 decoder expert 的投影均为 INT4；用 `--group-size 0` 选择逐通道 scale。

## 选择配置

| 配置 | Vision / Prefill | Action | 区别 |
|---|---|---|---|
| [FP16](fp16.toml) | FP16 | FP16 | 浮点权重 |
| [W8](w8a16.toml) | W8A16 | W8A16 | 推荐配置 |
| [W8 + Action W4](w8a16_action_w4a16.toml) | W8A16 | packed W4A16-pgrp | 非NVFP4，非W8A8 |

三档保持既有ACC32/BF16边界，均为evaluation-only，不继承base的分阶段ACC切换能力。固定TP8/B1、三图448×252、H20、六步去噪、P640/C320×2；真实prefix需在599..640。

## 使用

复制模板，填写 `fp16_checkpoint`、batch和output_dir；W8另填 `w8_checkpoint`，Action W4另填 `action_w4_checkpoint`。FP16 checkpoint在三档均必填，用于配置/配对身份：

```bash
cp examples/configs/wall_oss/sorting/w8a16.toml /tmp/wall_sorting.toml
python examples/run.py --config /tmp/wall_sorting.toml --check-config
python examples/run.py --config /tmp/wall_sorting.toml
```

输出 `report.json`、`actions.pt`。每个冷配置用新进程和独立输出目录，运行需匹配 native、空闲板卡及既有设备锁；`--check-config` 仅检查配置。

采样设置示例：`cpu.threads=8`、`run.warmup=2`、`run.runs=5`；模板默认6次正式运行。

## 可修改参数

| 字段 | 默认 / 可选 | 条件 |
|---|---|---|
| `input.inference_delay` | 0 / 1 / 3 / 10 | 非零须提供前次物理动作前缀 |
| `input.batch` | 必填 | 不生成/裁剪图像或指令凑prefix |
| `cpu.threads` | 可选正整数 | 省略继承运行环境 |
| `run.warmup/runs` | 2 / 6 | warmup≥1，runs≥1；常规运行关闭跟踪 |

## 输入与限制

batch为CPU字典，字段：uint8 `images [3,252,448,3]`（face/left/right），非空instruction，浮点state `[26]`、noise `[1,20,26]`，bool state_mask/dof_mask `[26]`（前20 true）；非零delay另需浮点 `prefix_actions [20,26]`。浮点必须有限；state和前次动作使用物理模型空间。

CPU 比较、分段 profile 和 RTC 变输入验证须单独准备匹配的参考与输入。库依赖见 [正式 main/DDR 两份 REF](../../../../docs/runtime_assets.md)。
