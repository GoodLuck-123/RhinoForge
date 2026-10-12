# HyVLA 配置
[配置索引](../CONFIGURATIONS.md) · [使用](#使用)

入口：`python examples/run.py --config <file>`；推荐 [3cam/fp16.toml](3cam/fp16.toml)。固定 B1、TP8、十步 action；原登记 fixture 仍可显式使用 [hy_embodied/umi/fp16.toml](../hy_embodied/umi/fp16.toml)。

<a id="weights"></a>
<a id="weight-sources"></a>
## 权重来源

官方文件的存储 dtype 与本仓库运行时的 FP16/W8 等选项分开列示；`FP16` 配置不表示官方发布文件就是 FP16。

| 身份 | 名称与发布者 / 下载渠道 | 文件存储精度 | 本地目录与配置绑定 |
|---|---|---|---|
| 官方发布 / 当前实际父 | 腾讯：[Hy-Embodied-0.5-VLA-UMI](https://huggingface.co/tencent/Hy-Embodied-0.5-VLA-UMI/tree/3f53d1f8d2bc587c523cfdc9f1041ceee42c2524) | BF16（官方 safetensors 元数据） | `Hy-Embodied-0.5-VLA-UMI` → `input.checkpoint`；目录名区分大小写，保留 tokenizer/归一化文件 |
| 公开量化 / 当前转换 | 使用本仓库转换配方；优化 HyVLA 由本仓库加载时转换 | 内存中 INT8 / NVFP4 + 浮点保留项 | 无独立量化目录；使用同一完整原始父 |

<a id="weight-recipes"></a>
## 模块精度与配方

W/A 与 ACC 定义见[精度记法](../WEIGHTS.md#precision-notation)。 NVFP4A16 为 E2M1 四位权重、沿 K 每 16 个值一组的 E4M3FN 八位 scale、FP32 tensor scale，激活仍为 FP16； 相机数/文本长度与驻留 scale 层数不改变源目录；驻留选项只改变既有 scale 的存放位置。

| 配方与模板 | 加载后 Vision / Language | 加载后 Action / IO 与保留浮点部分 | 所需源权重、转换方式与完整命令 |
|---|---|---|---|
| FP16：[两图](2cam/fp16.toml) / [三图](3cam/fp16.toml) | ViT 与 VLM text/vision 两分支的主投影 W16A16 | 实际动作 `_v` 分支主投影 W16A16；ViT patch/merger、norm/bias/final norm 为 FP16；host token embedding、state/时间预处理为 FP32；图内动作 IO 为 W16A16 | 原始 UMI；直接加载；[完整命令](#run-fp16) |
| W8A16：[两图](2cam/w8a16.toml) / [三图](3cam/w8a16.toml) | ViT Q/K/V/O/FC1/FC2、VLM 两分支 Q/K/V/O/Gate/Up/Down 为 W8A16 | 动作 `_v` Q/K/V/O/Gate/Up/Down 为 W8A16；ViT patch/merger、norm/bias/final norm 为 FP16；host token embedding、state/时间预处理为 FP32；图内动作 IO 为 W16A16 | 原始 UMI；加载时 `_quant()` / `_q8()`；[完整命令](#run-w8)，[python/rpu_backend/adapters/hy_vla/weights.py](../../../python/rpu_backend/adapters/hy_vla/weights.py) |
| W8 + 动作 NVFP4：[两图](2cam/w8a16_nvfp4.toml) / [三图](3cam/w8a16_nvfp4.toml) | 同 W8A16 | 动作 `_v` Q/K/V/O/Gate/Up/Down 全为 NVFP4A16，包含 K/V；其余浮点保留项同上 | 原始 UMI；加载时 `quantize_expert_nvfp4_cpu()`（先舍入 FP16）；[完整命令](#w8--action-nvfp4)，[python/rpu_backend/adapters/hy_vla/nvfp4.py](../../../python/rpu_backend/adapters/hy_vla/nvfp4.py) |

<a id="quantization"></a>
## 加载时量化

以下命令在仓库根目录执行。量化在最后一行加载模型时发生，不生成独立量化目录。先替换实际路径，再按注释用 `vi` 编辑副本并保存；`--check-config` 只校验配置，不能代替最后的运行命令。

<a id="download-parent"></a>
### 下载官方 BF16 父权重

```bash
SRC=/path/to/model-cache/Hy-Embodied-0.5-VLA-UMI
hf download tencent/Hy-Embodied-0.5-VLA-UMI --revision 3f53d1f8d2bc587c523cfdc9f1041ceee42c2524 --local-dir "$SRC"
```

<a id="run-fp16"></a>
### FP16：加载时转换浮点 dtype，无需量化

```bash
cp examples/configs/hy_vla/2cam/fp16.toml /tmp/hyvla_fp16.toml
# input.checkpoint = 下载所得 UMI 完整目录的绝对路径。
# [libraries].native/main/expansion 填实际路径，或删除整个 [libraries] 表。
# run.output_dir = 本次目录；保留 source="synthetic" 无需额外 batch。
vi /tmp/hyvla_fp16.toml
python examples/run.py --config /tmp/hyvla_fp16.toml
```

<a id="run-w8"></a>
### W8A16

```bash
cp examples/configs/hy_vla/2cam/w8a16.toml /tmp/hyvla_w8.toml
# input.checkpoint = 原始 Hy-Embodied-0.5-VLA-UMI 完整目录的绝对路径。
# [libraries] 的 native/main/expansion 三项改为实际路径，或删除整个 [libraries] 表以使用当前环境。
# run.output_dir = 本次独立输出目录。保留 source="synthetic" 时无需另配 batch 文件。
vi /tmp/hyvla_w8.toml
python examples/run.py --config /tmp/hyvla_w8.toml
```

<a id="w8--action-nvfp4"></a>
### W8 + 动作 NVFP4

```bash
cp examples/configs/hy_vla/2cam/w8a16_nvfp4.toml /tmp/hyvla_nvfp4.toml
# input.checkpoint = 原始 Hy-Embodied-0.5-VLA-UMI 完整目录的绝对路径。
# [libraries] 的 native/main/expansion 三项改为实际路径，或删除整个 [libraries] 表以使用当前环境。
# run.output_dir = 本次独立输出目录。保留 source="synthetic" 时无需另配 batch 文件。
vi /tmp/hyvla_nvfp4.toml
python examples/run.py --config /tmp/hyvla_nvfp4.toml
```

三图使用 `3cam/` 同精度模板。以上直接填写绝对 checkpoint 路径，不要求 `RPU_MODEL_CACHE`；若保留 registry 别名，则需设置该变量并准备 `Hy-Embodied-0.5-VLA-UMI` 子目录。**FP16 无需量化**。

**实现位置**：[examples/runners/hy_vla/optimized.py](../../runners/hy_vla/optimized.py)；[python/rpu_backend/adapters/hy_vla/weights.py](../../../python/rpu_backend/adapters/hy_vla/weights.py)；[python/rpu_backend/adapters/hy_vla/nvfp4.py](../../../python/rpu_backend/adapters/hy_vla/nvfp4.py)。`run_example_config()` → `HyEmbodiedPolicy.from_checkpoint(...).to("rpu")` → `load_hy_vla_weights()`；W8 调用 `_quant()` / `_q8()`，NVFP4 调用 `quantize_expert_nvfp4_cpu()`。patch embedding、merger、final norm 保持浮点。实现模块不是 CLI。

## 选择配置

| 权重组合 | 两图 | 三图 | 区别 |
|---|---|---|---|
| FP16 | [配置](2cam/fp16.toml) | [配置](3cam/fp16.toml) | Vision/VLM/Action 浮点权重 |
| W8A16 | [配置](2cam/w8a16.toml) | [配置](3cam/w8a16.toml) | 三组件 INT8 权重 |
| W8 + Action NVFP4 | [配置](2cam/w8a16_nvfp4.toml) | [配置](3cam/w8a16_nvfp4.toml) | Vision/VLM W8，Action NVFP4A16 |

两图使用 `top_head`、`hand_left`，三图增加 `hand_right`。当前 CLI 使用 TP8，未提供 ACC 模式切换。

## 使用

复制模板，填写 `input.checkpoint` 和 `[libraries].native/main/expansion`，设置独立输出目录：

```bash
cp examples/configs/hy_vla/2cam/w8a16_nvfp4.toml /tmp/hyvla.toml
python examples/run.py --config /tmp/hyvla.toml --check-config
python examples/run.py --config /tmp/hyvla.toml
```

synthetic 使用 checkpoint tokenizer 和 `run.seed` 生成 RGB/state/noise，保存 `input_batch.pt`。每个冷配置使用新进程和独立输出目录；运行需空闲板卡及既有设备锁，`--check-config` 仅检查配置。

## 可修改参数

| 字段 | 默认 / 范围 | 条件 |
|---|---|---|
| `input.text_tokens` | 32；1..64 | tokenizer 有效 token 数 |
| `input.prefix_len` | auto | `ceil16(2 + cameras×58 + T)`；显式值为合法 16 对齐 bucket |
| `input.source` | synthetic / batch | batch 模式提供真实保存输入 |
| `input.instruction` | 默认长提示词 | 至少含 T 个 token；会截断到 T，再 PAD 到64槽 |
| `cpu.threads` | 12，正整数 | 本轮实际12；2026-10-04旧矩阵8 |
| `run.warmup/runs/seed` | 3 / 6 / 0 | warmup≥3、runs≥1、seed≥0 |
| `input.reference` | 可省略 | 入口仅接受同输入身份的 设备 reference；CPU 数值比较需单独准备同输入参考 |

自动 prefix：T16/32/48/64 时两图 P144/160/176/192，三图 P192/208/224/240。显式 prefix 不小于自动值、不超过完整64语言槽容量；padding 始终不可见。

## 输入与限制

synthetic 提示词截断可能删掉 assistant 后缀，仅用于固定输入检查；输出记录实际 token 和截断状态。

重放保存输入时设置 `source="batch"` 和 `batch` 路径，移除 `instruction`；cameras/text_tokens、mask 和位置必须匹配文件内容。

三种精度共用 [正式 main/DDR 两份 REF](../../../docs/runtime_assets.md)。
