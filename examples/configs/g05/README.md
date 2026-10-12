# G0.5 K1 配置
[使用步骤](#使用) · [旧配置](legacy/README.md)

六份模板覆盖两/三图和三种精度；每份均可修改 ACC16/ACC32、BASE/HIGH 和文本长度。入口为 `python examples/run.py --config <file>`。固定 batch1、TP8、每相机一帧、十步 FM，输出 `[1,32,27]`。

<a id="weights"></a>
<a id="weight-sources"></a>
## 权重来源

官方文件的存储 dtype 与本仓库运行时的 FP16/W8 等选项分开列示；`FP16` 配置不表示官方发布文件就是 FP16。

| 身份 | 名称与发布者 / 下载渠道 | 文件存储精度 | 本地目录与配置绑定 |
|---|---|---|---|
| 官方公开发布 | 未核实这份 G0.5 K1 `vla_rtc` release 的官方公开 HF / GitHub 权重下载 | 未核实；不从 `fp16` 运行标签推断文件 dtype | 无可替代的公开下载目录 |
| 当前实际父 | 客户 / 受控交付方提供的原始 K1 checkpoint，须配同版 release 源码、processor 和输入 | 原始 dtype 由交付清单指定；运行前按模块转 FP16 / FP32 | `/path/to/vla_rtc_release/checkpoint` → `input.checkpoint`；源码 → `input.release_root` |
| 公开量化 / 当前量化产物 | 无已核实兼容公开量化地址；本仓库加载时转换原始父 | 内存中 INT8 或 NVFP4 + 浮点保留项 | 不导出量化目录；仍填写原始 `input.checkpoint` |

<a id="weight-recipes"></a>
## 模块精度与配方

W/A 与 ACC 定义见[精度记法](../WEIGHTS.md#precision-notation)。 NVFP4A16 为 E2M1 四位权重、沿 K 每 16 个值一组的 E4M3FN 八位 scale、FP32 tensor scale，激活仍为 FP16； BASE/HIGH 以及 ACC 选项不改变此处的权重位宽。

| 配方与模板 | 加载后 Vision / Language | 加载后 Action / IO 与保留浮点部分 | 所需源权重、转换方式与完整命令 |
|---|---|---|---|
| FP16：[两图](2cam/fp16.toml) / [三图](3cam/fp16.toml) | Vision / Language 主投影 W16A16 | Action 主投影 W16A16；Vision patch/merger、Language norm/bias、GDN conv/标量与 `in_proj_a/b` 保留 FP16；嵌入不量化、输出转 FP16 送设备；设备 IO W16A16，CPU AdaLN/时间条件 FP32 | 原始 K1；直接加载；[完整命令](#run-fp16) |
| W8A16：[两图](2cam/w8a16.toml) / [三图](3cam/w8a16.toml) | Vision Q/K/V/O/FC1/FC2；Language MLP、全注意力 Q/K/V/O、GDN QKV/Z/Out 为 W8A16 | 动作 Q/K/V/O/Gate/Up/Down 及 attention gate 为 W8A16；Vision patch/merger、Language norm/bias、GDN conv/标量与 `in_proj_a/b` 保留 FP16；嵌入不量化、输出转 FP16 送设备；设备 IO W16A16，CPU AdaLN/时间条件 FP32 | 原始 K1；加载时 `quantize_g05_text_weights()` 与 Vision 转换；[完整命令](#run-w8)，[python/rpu_backend/adapters/g05/quant.py](../../../python/rpu_backend/adapters/g05/quant.py) |
| W8 + 动作 NVFP4：[两图](2cam/w8a16_nvfp4.toml) / [三图](3cam/w8a16_nvfp4.toml) | 同 W8A16 | 动作 Q/O/Gate/Up/Down 为 NVFP4A16（`striped_v2`）；K/V/attention gate 仍 W8A16；其余同上 | 原始 K1；加载时 `prepare_g05_action_nvfp4()`；[完整命令](#w8--action-nvfp4) |
| 旧版受控 FP16：[legacy/fp16.toml](legacy/fp16.toml) | 原登记浮点路径 | 无权重量化；保留原 K1 的 FP16 设备 / FP32 host 边界 | 同 K1 父 + 配对预处理 batch；[旧版完整命令](legacy/README.md#run-fp16) |

<a id="quantization"></a>
## 加载时量化

以下命令在仓库根目录执行。量化在最后一行加载模型时发生，不生成独立量化目录。先替换实际路径，再按注释用 `vi` 编辑副本并保存；`--check-config` 只校验配置，不能代替最后的运行命令。

<a id="run-fp16"></a>
### FP16：原始 K1 直接运行

```bash
cp examples/configs/g05/2cam/fp16.toml /tmp/g05_fp16.toml
# input.checkpoint = 原始 K1；input.release_root = 配对源码；input.source_golden = 配对 golden。
# [libraries].native/main/expansion 填实际路径，或删除整个 [libraries] 表使用当前环境。
# run.output_dir = 本次独立目录；三图改复制 3cam/fp16.toml 并配三图输入。
vi /tmp/g05_fp16.toml
python examples/run.py --config /tmp/g05_fp16.toml
```

<a id="run-w8"></a>
### W8A16

```bash
cp examples/configs/g05/2cam/w8a16.toml /tmp/g05_w8.toml
# input.checkpoint = 客户原始未量化 K1 checkpoint 目录。
# input.release_root = 同一 release 源码目录；input.source_golden = 配对真实 golden 文件。
# [libraries] 的 native/main/expansion 三项改为实际路径，或删除整个 [libraries] 表以使用当前环境。
# run.output_dir = 本次输出目录。保留本模板 precision、相机和输入模式。
vi /tmp/g05_w8.toml
python examples/run.py --config /tmp/g05_w8.toml
```

<a id="w8--action-nvfp4"></a>
### W8 + 动作 NVFP4

```bash
cp examples/configs/g05/2cam/w8a16_nvfp4.toml /tmp/g05_nvfp4.toml
# input.checkpoint = 客户原始未量化 K1 checkpoint 目录。
# input.release_root = 同一 release 源码目录；input.source_golden = 配对真实 golden 文件。
# [libraries] 的 native/main/expansion 三项改为实际路径，或删除整个 [libraries] 表以使用当前环境。
# run.output_dir = 本次输出目录。保留本模板 precision、相机和输入模式。
vi /tmp/g05_nvfp4.toml
python examples/run.py --config /tmp/g05_nvfp4.toml
```

三图改复制 `3cam/` 的同精度模板，并使用三图配对输入。**FP16 无需量化**。两种量化均读取原始 K1 父权重，不把 checkpoint 改成其他模型的量化资产。

**实现位置**：[examples/runners/g05/optimized.py](../../runners/g05/optimized.py)；[python/rpu_backend/adapters/g05/runtime.py](../../../python/rpu_backend/adapters/g05/runtime.py)；[python/rpu_backend/adapters/g05/quant.py](../../../python/rpu_backend/adapters/g05/quant.py)；[python/rpu_backend/adapters/qwen3_5/vision.py](../../../python/rpu_backend/adapters/qwen3_5/vision.py)。`run_example_config()` → `patch_g05_policy_for_rpu()`；Text/Action W8 调用 `quantize_g05_text_weights()`，NVFP4 调用 `prepare_g05_action_nvfp4()`，Vision 由 `_convert_vision_block_weights_for_rpu()` 内部量化。上述模块由运行入口调用。

## 原 G05 源码集成入口

[base/fp16.toml](base/fp16.toml) 保留原官方 G05 源码集成配置。按其中固定 revision 准备
OpenGalaxea/G05 与 GalaxeaVLA，构建 CPU `G05PolicyQwen35`，再调用
`rpu_backend.adapters.g05.patch_g05_policy_for_rpu(policy, max_seq_len=2048)`，
使用其连续动作、离散动作与 CoT 请求路径。此配置的 `--check-config` 用于检查原身份和
集成要求；本页 K1 动作 runner 需要另外准备配对资产与 batch。

## 选择配置

两图使用 `head_rgb + left_wrist_rgb`；三图增加 `right_wrist_rgb`。

| 精度 | Vision / Text | Action | 默认 ACC |
|---|---|---|---|
| FP16 | FP16 | FP16 | ACC32 |
| W8A16 | W8A16 | W8A16 | ACC16 |
| W8 + Action NVFP4 | W8A16 | Q/O/Gate/Up/Down 为 NVFP4A16；K/V/attention gate 为 W8A16 | ACC16 |

六份模板分别为：

| 配置 | 默认 ACC |
|---|---|
| [2cam/fp16.toml](2cam/fp16.toml) | ACC32 |
| [2cam/w8a16.toml](2cam/w8a16.toml) | ACC16 |
| [2cam/w8a16_nvfp4.toml](2cam/w8a16_nvfp4.toml) | ACC16 |
| [3cam/fp16.toml](3cam/fp16.toml) | ACC32 |
| [3cam/w8a16.toml](3cam/w8a16.toml) | ACC16 |
| [3cam/w8a16_nvfp4.toml](3cam/w8a16_nvfp4.toml) | ACC16 |

## 使用

复制模板，填写 `[input]` 的 `checkpoint`、`release_root`、`source_golden` 和 `[run]` 的 `output_dir`。checkpoint 使用原始未量化 `vla_rtc`；入口安装所选权重格式。`[libraries]` 同时填写 `native/main/expansion`，或删除整个表使用当前运行库，见[算子库规则](../../../docs/runtime_assets.md)。

```bash
cp examples/configs/g05/2cam/w8a16.toml /tmp/g05.toml
# 编辑资产路径和 run.output_dir。
python examples/run.py --config /tmp/g05.toml --check-config
python examples/run.py --config /tmp/g05.toml
```

`--check-config` 检查配置结构。实际运行在空闲板卡持设备锁，每种精度/运行库使用新进程。结果包含 `actions.pt`、`input_batch.pt`；多长度按顺序各起独立进程，失败即停止，结果分存于 `text_N/`；`stdout.log` 和 `process_exit_code` 记录子进程结果。模板默认六次采样；可通过 `run.runs` 修改正式采样次数。

## 修改参数

| 字段 | 默认 / 可选值 | 作用 |
|---|---|---|
| `g05.linear_accumulation` | 默认见上表；`acc16/acc32` | 三种精度均支持，安装前选择 |
| `g05.math_precision` | `base/high`；默认 base | HIGH 使用高精度非线性与 Newton RMSNorm，激活/输出仍为 FP16 |
| `input.text_lengths` | `[32,52,64,128,192,256]`；允许 32–256 的整数 | 单长度可改用 `text_tokens`，两字段互斥 |
| `input.source_step` | 默认 1（RTC）；可选 0（no-RTC） | 选择 `source_golden` 中对应输入；两者分别准备 CPU 参考 |
| `g05.vision_camera_batch` | `auto` | W8/NVFP4 可选 1 或相机数 2/3；FP16 仅 1 |
| `rpu_execution.components.language_model.prefill.chunk_size` | `auto` | 显式值需通过运行时容量检查 |
| `cpu.threads` | 8 | CPU 线程数 |
| `run.warmup/runs/seed` | 2 / 6 / 7 | CPU 比较使用保存的实际 FM 噪声 |

`auto` 应用[selected_profiles.json](selected_profiles.json)中匹配的相机分组与 Text chunk；未匹配则逐相机执行/自动 chunk，显式值优先。HIGH 沿用 BASE 调度。

## 输入

`source="synthetic"` 使用 checkpoint 匹配的 `source_golden` 中的真实 K1 图像、状态和 RTC 前缀，任务文本由原 release processor 编码。本轮用合成指令覆盖指定长度；自定义 `instruction` 只接受一个匹配的实际长度。公开入口不运行 CPU 参考。

`source="batch"` 需填写 `batch`，移除 `source_golden/source_step/instruction/text_lengths`；它重放保存输入的实际单长度与相机顺序，改 `text_tokens` 不改变已有 token。

T 为实际 TEXT/COT/PRED_TEXT 计数，含模板及图像边界文本，排除 IMAGE/STATE；两图 P=T+129，三图 P=T+193。不能用有效 PAD、空图或截断序列凑长度。文本范围测量不扩展生产 API 支持。

<a id="推荐配置与路径迁移"></a>

旧登记输入见[legacy](legacy/README.md)。
