# Wall-OSS 原 registry 配置
[上级选型](../README.md) · [使用](#使用)

<a id="weights"></a>
<a id="weight-sources"></a>
## 权重来源

官方文件的存储 dtype 与本仓库运行时的 FP16/W8 等选项分开列示；`FP16` 配置不表示官方发布文件就是 FP16。

| 身份 | 名称与发布者 / 下载渠道 | 文件存储精度 | 本地目录与配置绑定 |
|---|---|---|---|
| 官方公开发布 | X Square Robot：[wall-oss-0.5](https://huggingface.co/x-square-robot/wall-oss-0.5/tree/b7dc23a82dbf70a65859c9fdd596703234597028)；[官方源码](https://github.com/X-Square-Robot/wall-x) | BF16 + FP32（官方 safetensors 元数据） | `wall-oss-0.5`；完整模型、processor/tokenizer、normalizer |
| 当前公开基础模型父 | base / legacy 使用上述公开模型身份；固定 source revision 与文件校验和 | 原文件浮点；运行时转 FP16 / FP32 | base：`input.fp16_checkpoint`；legacy：`$RPU_MODEL_CACHE/wall-oss-0.5` |
| 公开量化 / 本仓库转换 | 使用本仓库转换配方；下列 CLI 从本组原始父转换 | W8：2D linear INT8；W4：两个 decoder expert 的投影均为 INT4；浮点保留项 FP16 | `$RPU_MODEL_CACHE/wall-oss-0.5-w8a16` / `wall-oss-0.5-w4a16` / `wall-oss-0.5-w4a16-pgrp` → 各模板 registry 别名 |

<a id="weight-recipes"></a>
## 模块精度与配方

W/A 与 ACC 定义见[精度记法](../../WEIGHTS.md#precision-notation)。 W4 G0 是逐通道 INT4，G32 是 K 维每 32 值一组；两者 scale 为 FP16，均不是 NVFP4。

| 配方与模板 | 加载后 Vision / Language | 加载后 Action / IO 与保留浮点部分 | 所需源权重、转换方式与完整命令 |
|---|---|---|---|
| legacy FP16：[fp16.toml](fp16.toml) | Vision / Language 主投影 W16A16 | Action 主投影 W16A16；浮点保留项见下文 | 公开原父；直接加载；[完整命令](#run-fp16) |
| legacy W8：[w8a16.toml](w8a16.toml) | Vision / Language 主投影 W8A16 | Action 主投影 W8A16；浮点保留项见下文 | 公开原父 → 离线 W8；[完整命令](#公开-base-legacy-w8a16)，[python/rpu_backend/quant/convert_wall_oss_w8a16.py](../../../../python/rpu_backend/quant/convert_wall_oss_w8a16.py) |
| legacy W4 G0：[w4a16.toml](w4a16.toml) | Vision 从配对 W8 目录加载；Language decoder W4A16 | Action decoder W4A16；Q/K/V/O/Gate/Up/Down 逐通道 INT4，FP16 scale，运行时打包 pgrp 布局；浮点保留项见下文 | 公开原父 + W8；离线 `--group-size 0`；[完整命令](#公开-base-legacy-w4逐通道-group0)，[convert_wall_oss.py](../../../../python/rpu_backend/quant/convert_wall_oss.py) |
| legacy W4 G32：[w4a16_pgrp.toml](w4a16_pgrp.toml) | Vision W8A16；Language Q/K/V/O/Gate/Up/Down 均为 G32 W4A16 | Action decoder 全部七类投影 G32 W4A16；scale FP16；浮点保留项见下文 | 公开原父 + W8；离线 G32；[完整命令](#公开-base-legacy-w4-pgrp混合-group32)，[convert_wall_oss.py](../../../../python/rpu_backend/quant/convert_wall_oss.py) |

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

<a id="run-fp16"></a>
### FP16：原始浮点父权重直接运行

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
cp examples/configs/wall_oss/legacy/fp16.toml /tmp/wall_legacy_fp16.toml
# 原始公开父目录为 $RPU_MODEL_CACHE/wall-oss-0.5，保留 registry_alias。
# input.images/instruction/proprioception/state_mask 填配对输入；在 [run] 添加 output_dir。
vi /tmp/wall_legacy_fp16.toml
python examples/run.py --config /tmp/wall_legacy_fp16.toml
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

**FP16 无需量化。** 源/量化目录保持兄弟关系，以便 `source` 元数据解析。公开 base 与客户 WRC 不能混用；group0 与 group32 使用不同 scale 配方。转换计算使用 CPU，命令需在匹配的后端宿主环境中执行。

**实现位置**：[python/rpu_backend/quant/convert_wall_oss_w8a16.py](../../../../python/rpu_backend/quant/convert_wall_oss_w8a16.py)；[convert_wall_oss.py](../../../../python/rpu_backend/quant/convert_wall_oss.py)。W8/W4 CLI 分别调用各模块的 `convert_checkpoint()`。W4 默认 G32，两个 decoder expert 的投影均为 INT4；用 `--group-size 0` 选择逐通道 scale。

## 选择配置

| 配置 | 精度范围 / 区别 | 登记状态 |
|---|---|---|
| [FP16](fp16.toml) | 原浮点配方 | SUPPORTED_LIMITED，精确profile |
| [W8](w8a16.toml) | 原INT8配方 | EXPERIMENTAL |
| [W4](w4a16.toml) | Vision W8、Prefill/Action decoder per-channel W4、glue FP16 | EXPERIMENTAL |
| [W4-pgrp](w4a16_pgrp.toml) | Vision W8、decoder group-wise W4、glue FP16 | EXPERIMENTAL |

两种 W4 都量化 Prefill/Action decoder，区别在按通道或按组计算 scale；它们与 base 的 Action-only NVFP4、sorting 配方不同。

## 使用

通过 `examples/run.py --config <TOML>` 显式选择旧 registry 配置：

```bash
python examples/run.py --config examples/configs/wall_oss/legacy/w4a16_pgrp.toml --check-config
python examples/run.py --config /path/to/wall_legacy.toml
```

填写原checkpoint配对、face/left/right三张图片、指令与26维状态，保持真实prefix范围 `[545,640]`。

采样设置示例：`cpu.threads=8`、`run.warmup=2`、`run.runs=5`；模板默认3次正式运行。

## 参数与限制

可修改图片、状态、prompt 和原 profile 允许的 run 参数；冷配置变化后使用新进程。`--check-config` 仅检查配置。库依赖见 [正式 main/DDR 两份 REF](../../../../docs/runtime_assets.md)。
