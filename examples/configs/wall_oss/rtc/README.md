# Wall-OSS 受控 RTC 配置
[上级选型](../README.md) · [使用](#使用)

<a id="weights"></a>
<a id="weight-sources"></a>
## 权重来源

官方文件的存储 dtype 与本仓库运行时的 FP16/W8 等选项分开列示；`FP16` 配置不表示官方发布文件就是 FP16。

| 身份 | 名称与发布者 / 下载渠道 | 文件存储精度 | 本地目录与配置绑定 |
|---|---|---|---|
| 官方公开发布（另一资产） | X Square Robot：[wall-oss-0.5](https://huggingface.co/x-square-robot/wall-oss-0.5/tree/b7dc23a82dbf70a65859c9fdd596703234597028)；[官方源码](https://github.com/X-Square-Robot/wall-x) | BF16 + FP32（官方 safetensors 元数据） | `wall-oss-0.5`；完整模型、processor/tokenizer、normalizer |
| 当前客户 WRC 父 | 原交付方：`wrc_sf_v728_0_15000`，未核实官方公开下载；不可由公开 base 替代 | 原始 dtype 由交付清单指定 | RTC：`input.checkpoint` / `input.fp16_checkpoint`；sorting：`input.fp16_checkpoint` |
| 公开量化 / 本仓库转换 | 使用本仓库转换配方；下列 CLI 从本组原始父转换 | W8：2D linear INT8，浮点保留项 FP16 | `wrc_v728_w8a16` → `input.checkpoint`；原父 → `input.fp16_checkpoint` |

<a id="weight-recipes"></a>
## 模块精度与配方

W/A 与 ACC 定义见[精度记法](../../WEIGHTS.md#precision-notation)。 RTC 的 BF16 溢出边界不改变其余线性 A16 路径。

| 配方与模板 | 加载后 Vision / Language | 加载后 Action / IO 与保留浮点部分 | 所需源权重、转换方式与完整命令 |
|---|---|---|---|
| RTC FP16：[fp16.toml](fp16.toml) | Vision / Language 主投影 W16A16；保留既有混合浮点边界 | Action 主投影 W16A16；浮点保留项见下文 | 客户 WRC 原父；直接加载；[完整命令](#run-fp16) |
| RTC W8：[w8a16.toml](w8a16.toml) | Vision / Language 主投影 W8A16，但 Vision block17 Down 与 Language layer2 Down 保留原父 FP16，使用既有 ACC32→BF16 输出边界，其余相关激活 A16 | Action decoder W8A16；浮点保留项见下文 | 客户 WRC 原父 + 配对离线 W8；[完整命令](#run-rtc-w8)，[python/rpu_backend/quant/convert_wall_oss_w8a16.py](../../../../python/rpu_backend/quant/convert_wall_oss_w8a16.py) |

**各配方的浮点保留项：** patch/merger、norm/bias、embedding、动作预处理/输出回投保持浮点：设备路径 FP16，host 预处理/归一化 FP32；不使用 W8 文件中 action_preprocessor 的 INT8 副本。离线 W8 转换器会量化除 `embed_tokens` 外的全部浮点 2D `.weight`；运行器只取所需 INT8 投影，其余胶水权重仍从配对原父加载。

<a id="quantization"></a>
## 下载与转换

以下命令在仓库根目录执行。先把 `/path/to/...` 换成实际路径；`vi` 这一步编辑刚复制的 TOML，保存后再执行下一行。每个代码块是一个独立精度方案，只执行所需方案；已有同身份量化目录时跳过对应转换行。

<a id="run-fp16"></a>
### FP16：原始浮点父权重直接运行

```bash
cp examples/configs/wall_oss/rtc/fp16.toml /tmp/wall_rtc_fp16.toml
# input.checkpoint = 原交付 WRC 目录；input.images/instruction/proprioception = 配对输入。
# 保留受控 opt-in / prefix 范围；在 [run] 添加本次 output_dir。
vi /tmp/wall_rtc_fp16.toml
python examples/run.py --config /tmp/wall_rtc_fp16.toml
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

**FP16 无需量化。** 源/量化目录保持兄弟关系，以便 `source` 元数据解析。公开 base 与客户 WRC 不能混用；group0 与 group32 使用不同 scale 配方。转换计算使用 CPU，命令需在匹配的后端宿主环境中执行。

**实现位置**：[python/rpu_backend/quant/convert_wall_oss_w8a16.py](../../../../python/rpu_backend/quant/convert_wall_oss_w8a16.py)；[convert_wall_oss.py](../../../../python/rpu_backend/quant/convert_wall_oss.py)。W8/W4 CLI 分别调用各模块的 `convert_checkpoint()`。W4 默认 G32，两个 decoder expert 的投影均为 INT4；用 `--group-size 0` 选择逐通道 scale。

## 选择配置

| 配置 | 资产 / 精度 | 状态 |
|---|---|---|
| [fp16.toml](fp16.toml)，推荐配置 | 受控WRC v728 FP16 | CONTROLLED_EVALUATION |
| [w8a16.toml](w8a16.toml) | 配对WRC v728 W8，原固定混合累加边界 | CONTROLLED_EVALUATION |

两份配置均为三图、动作长度20、6步去噪，使用固定累加边界。W8需要配对的FP16 checkpoint提供CPU胶水权重。

## 使用

复制配置，填写checkpoint、三张RGB图片、指令、27维状态与输出目录。图片顺序为 `face_view`、`left_wrist_view`、`right_wrist_view`；推荐三张336×336图片：

```bash
python examples/run.py --config /path/to/wall_rtc.toml --check-config
python examples/run.py --config /path/to/wall_rtc.toml
```

采样设置示例：`cpu.threads=8`、`run.warmup=2`、`run.runs=5`；模板默认3次正式运行。

## 可修改参数与限制

- 每路图片经processor处理后必须恰为576个patch，三图共1728个patch；336×336对应grid `[1,24,24]`。其他宽高须确认处理后的grid，不能只按图片总面积判断。
- `input.prefix_length_range=[599,640]` 是图片token展开后的实际前缀长度范围，含两端；该范围只准备P640/C320图。默认指令 `Move the object to the target.` 配合上述三图得到622；更换指令或状态后须仍处于范围内。
- 默认 `input.inference_delay=0`。更改图片几何、前缀范围或执行配置后使用新进程，运行仍受板卡锁保护。

`--check-config`只检查配置结构，不读取权重或板卡。算子库使用main/DDR两个REF，见 [算子库规则](../../../../docs/runtime_assets.md)。
