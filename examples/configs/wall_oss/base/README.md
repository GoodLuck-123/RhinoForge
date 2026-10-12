# Wall-OSS base 配置
[上级选型](../README.md) · [使用](#使用)

入口：`python examples/run.py --config <file>`；推荐 [3cam/224/fp16.toml](3cam/224/fp16.toml)。固定 B1、TP8、H32、十步去噪。

<a id="weights"></a>
<a id="weight-sources"></a>
## 权重来源

官方文件的存储 dtype 与本仓库运行时的 FP16/W8 等选项分开列示；`FP16` 配置不表示官方发布文件就是 FP16。

| 身份 | 名称与发布者 / 下载渠道 | 文件存储精度 | 本地目录与配置绑定 |
|---|---|---|---|
| 官方公开发布 | X Square Robot：[wall-oss-0.5](https://huggingface.co/x-square-robot/wall-oss-0.5/tree/b7dc23a82dbf70a65859c9fdd596703234597028)；[官方源码](https://github.com/X-Square-Robot/wall-x) | BF16 + FP32（官方 safetensors 元数据） | `wall-oss-0.5`；完整模型、processor/tokenizer、normalizer |
| 当前公开基础模型父 | base / legacy 使用上述公开模型身份；固定 source revision 与文件校验和 | 原文件浮点；运行时转 FP16 / FP32 | base：`input.fp16_checkpoint`；legacy：`$RPU_MODEL_CACHE/wall-oss-0.5` |
| 公开量化 / 本仓库转换 | 使用本仓库转换配方；下列 CLI 从本组原始父转换 | W8：2D linear INT8，浮点保留项 FP16 | `wall-oss-0.5-w8a16` → `input.w8_checkpoint`；动作 NVFP4 仅在内存生成 |

<a id="weight-recipes"></a>
## 模块精度与配方

W/A 与 ACC 定义见[精度记法](../../WEIGHTS.md#precision-notation)。 NVFP4A16 为 E2M1 四位权重、沿 K 每 16 个值一组的 E4M3FN 八位 scale、FP32 tensor scale，激活仍为 FP16；

| 配方与模板 | 加载后 Vision / Language | 加载后 Action / IO 与保留浮点部分 | 所需源权重、转换方式与完整命令 |
|---|---|---|---|
| base FP16：[两图224](2cam/224/fp16.toml) / [两图256](2cam/256/fp16.toml) / [三图224](3cam/224/fp16.toml) / [三图256](3cam/256/fp16.toml) | Vision、Language expert0 主投影 W16A16 | Action expert1 主投影 W16A16；patch/merger、norm/bias、embedding、动作预处理/输出回投保持浮点：设备路径 FP16，host 预处理/归一化 FP32；不使用 W8 文件中 action_preprocessor 的 INT8 副本 | 公开原父；直接加载；[完整命令](#run-fp16) |
| base W8：[两图224](2cam/224/w8a16.toml) / [两图256](2cam/256/w8a16.toml) / [三图224](3cam/224/w8a16.toml) / [三图256](3cam/256/w8a16.toml) | Vision block 与 Language expert0 Q/K/V/O/Gate/Up/Down W8A16 | Action expert1 相同七类投影 W8A16；patch/merger、norm/bias、embedding、动作预处理/输出回投保持浮点：设备路径 FP16，host 预处理/归一化 FP32；不使用 W8 文件中 action_preprocessor 的 INT8 副本 | 公开原父 → 离线 W8；[完整命令](#公开-base-w8a16)，[python/rpu_backend/quant/convert_wall_oss_w8a16.py](../../../../python/rpu_backend/quant/convert_wall_oss_w8a16.py) |
| base W8 + 动作 NVFP4：[两图224](2cam/224/w8a16_action_nvfp4.toml) / [两图256](2cam/256/w8a16_action_nvfp4.toml) / [三图224](3cam/224/w8a16_action_nvfp4.toml) / [三图256](3cam/256/w8a16_action_nvfp4.toml) | Vision / Language 同 base W8 | Action expert1 Q/K/V/O/Gate/Up/Down 全为 NVFP4A16，包含 K/V；浮点保留项同上 | 公开原父 + 配对 W8；动作加载时 `_load_expert1_decoder()` → `_pack_nvfp4_rpu()`，无 NVFP4 导出目录；[完整命令](#公开-base-w8--action-nvfp4) |

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
cp examples/configs/wall_oss/base/3cam/224/fp16.toml /tmp/wall_base_fp16.toml
# input.fp16_checkpoint = 已下载 wall-oss-0.5 的绝对目录。
# input.batch_template = 匹配真实 batch 模板（保留 {text_tokens}）；run.output_dir = 本次目录。
# 更换输入时按本页说明处理 padding_rows/padding_budget。
vi /tmp/wall_base_fp16.toml
python examples/run.py --config /tmp/wall_base_fp16.toml
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

**FP16 无需量化。** 源/量化目录保持兄弟关系，以便 `source` 元数据解析。公开 base 与客户 WRC 不能混用；group0 与 group32 使用不同 scale 配方。转换计算使用 CPU，命令需在匹配的后端宿主环境中执行。

**实现位置**：[python/rpu_backend/quant/convert_wall_oss_w8a16.py](../../../../python/rpu_backend/quant/convert_wall_oss_w8a16.py)；[convert_wall_oss.py](../../../../python/rpu_backend/quant/convert_wall_oss.py)。W8/W4 CLI 分别调用各模块的 `convert_checkpoint()`。W4 默认 G32，两个 decoder expert 的投影均为 INT4；用 `--group-size 0` 选择逐通道 scale。

**实现位置**：[python/rpu_backend/adapters/wall_oss/action.py](../../../../python/rpu_backend/adapters/wall_oss/action.py)；[python/rpu_backend/adapters/wall_oss/llm.py](../../../../python/rpu_backend/adapters/wall_oss/llm.py)。Action NVFP4 没有独立导出目录；选对应 TOML 后，加载时由 `_load_expert1_decoder()` 调用 `_pack_nvfp4_rpu()`，Vision/Prefill 读取 W8。

## 选择配置

| 权重组合 | 两图224 | 两图256 | 三图224 | 三图256 |
|---|---|---|---|---|
| FP16 | [配置](2cam/224/fp16.toml) | [配置](2cam/256/fp16.toml) | [配置](3cam/224/fp16.toml) | [配置](3cam/256/fp16.toml) |
| W8A16 | [配置](2cam/224/w8a16.toml) | [配置](2cam/256/w8a16.toml) | [配置](3cam/224/w8a16.toml) | [配置](3cam/256/w8a16.toml) |
| W8 + Action NVFP4 | [配置](2cam/224/w8a16_action_nvfp4.toml) | [配置](2cam/256/w8a16_action_nvfp4.toml) | [配置](3cam/224/w8a16_action_nvfp4.toml) | [配置](3cam/256/w8a16_action_nvfp4.toml) |

两图为 face/left wrist，三图增加 right wrist。混合档仅 Action 为 NVFP4A16，Vision/Prefill 为 W8；base 测试模板与 [legacy](../legacy/README.md)、[RTC](../rtc/README.md) 的配方和登记范围不同。

## 使用

<a id="运行与输入"></a>
复制模板，填写 FP16 与配对 W8 checkpoint、已有 batch、输出目录：

```bash
cp examples/configs/wall_oss/base/3cam/224/w8a16.toml /tmp/wall_base.toml
python examples/run.py --config /tmp/wall_base.toml --check-config
python examples/run.py --config /tmp/wall_base.toml
```

`input.batch_template` 展开 cameras/image_size/text_tokens，必须包含 `{text_tokens}`；固定 `input.batch` 与它互斥。入口读取已有输入，输出 `outputs.pt`。冷配置使用新进程，运行需空闲板卡及既有设备锁；`--check-config` 仅检查配置。

## 可修改参数

| 字段 | 默认 / 选项 | 条件 |
|---|---|---|
| `input.text_tokens` | 32；已测32/64/96/128/160 | instruction 有效 token 数，真实输入必须匹配 |
| `input.text_length_scope` | instruction / non_image | 后者数完整 prompt 非图像 token |
| `rpu_execution.prefill/vision/action.linear_acc32` | false=ACC16 / true=ACC32 | 分阶段冷绑定；权重/输出 dtype 不变 |
| `cpu.threads` | 模板按 T32 已测格设置；省略默认8 | 对其他长度/ACC需重新选择 |
| `rpu_execution.prefill.chunk_size` | auto | padding_rows/padding_budget 与完整 prompt 相关 |
| `wall_oss.vision_window` | 224专用 | 开启需 row reuse；256不允许 |
| `wall_oss.vision_pair` | 256专用前两图合批 | 三图第三路单独执行；224不允许 |
| `wall_oss.vision_row_reuse` | true | 行复用及对应传输路径 |
| `run.warmup/runs/seed` | 2 / 5 / 0 | seed 不替换 batch 中 noise |

更换输入后先移除旧 `padding_rows/padding_budget` 使用 AUTO，复现时再采用对应格参数。Action chunk 仅接受 auto/32；ACC32 的执行路径与 ACC16 不同。256输入由 processor 处理为252×252。

## 输入与限制

batch 恰含六项：`images` 为 uint8 `[N,H,W,3]`、顺序 face/left/right 的前N路；非空 `instruction`；有限浮点 `proprioception [26]`、`noise [1,32,26]`、`dof_mask [1,32,26]`；`state_mask` 为 bool `[26]` 前7 true，其余false，dof_mask 扩展同一七维 mask。当前几何使用 `berkeley_autolab_ur5` 前七个槽位。

库依赖为 [正式 main/DDR 两份 REF](../../../../docs/runtime_assets.md)，`[runtime]` 可指定同身份安装副本；运行时还需匹配配套 native。
