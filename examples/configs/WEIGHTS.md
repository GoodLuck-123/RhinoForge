# 模型权重来源、精度与转换

先从[逐模型入口](#sources)找到对应模型，再依次查看“官方来源与存储精度”“配置需要的模块精度”“下载与转换命令”。每个模型指南都把源权重、转换产物和消费该产物的 TOML 对应起来；按表中的模块范围和分组选择配方。文中的“父权重”指转换所用的原始模型文件，“加载时量化”指运行入口读取这些文件后在内存中转换。

<a id="precision-notation"></a>
## 精度记法

W 表示权重、A 表示线性算子输入激活。W16/A16 为 FP16；W8/A16 为逐输出通道对称 INT8 权重、FP16 scale 和激活。ACC16/ACC32 单独选择累加类型。低位格式、分组和保留浮点的模块见各模型配方表。

## 精度字段如何阅读

| 名称 | 本文含义 | 示例 |
|---|---|---|
| 官方发布的存储精度 | 下载文件中张量的 dtype，按官方模型文件元数据或已记录的原始文件核对；与当前运行配置分开 | BF16、FP32，或 BF16 主体与少量 FP32 张量 |
| 转换产物的存储格式 | 本地转换脚本写入的整数权重、浮点权重、scale 与布局；安装时可能继续量化或重新打包 | Pi0.5 的离线包保留浮点 Vision，W8 配置加载时再转换 Vision；具体差别见模块表 |
| 配置实际使用的模块精度 | 进入当前后端的 Vision、文本主干、Action/专家、head 等各自的权重与激活精度 | `W8A16` 表示该模块 INT8 权重、FP16 激活；未列入量化范围的模块继续使用浮点 |
| 运算累加精度 | 内核的累加模式，由对应 `linear_acc32` 等配置控制 | ACC32 指累加精度，权重存储 dtype 由 checkpoint 决定 |

`G32/G128` 表示量化分组大小。AWQ/GPTQ 的 INT4 权重可打包为 INT32/UINT8 张量，逻辑位宽由配方和 scale/zero-point 决定。NVFP4 与 INT4 使用不同格式，转换须使用对应工具。

## 按这四步准备

1. **选择 TOML 和配方行。** 查清模型大小/任务版本，以及各模块的精度；相机数、长度和核数相同配方可共用权重，具体范围见模型表。
2. **下载表中指定的源资产。** 官方浮点父、第三方 AWQ、官方 GPTQ 和客户交付包分别标注；GitHub 源码仓库不自动包含模型权重。下载命令和目标缓存目录在各模型指南中列出。
3. **按“转换方式”操作。** 离线转换执行所列脚本生成新目录；加载时量化直接读取浮点父并由运行入口转换，不预先生成量化目录；公开量化包直用时跳过本地量化。
4. **绑定输出并运行。** 将产物放入表中 registry 对应目录，或填写指定 `input.*checkpoint` 字段；按所列 TOML 运行。下载、转换、输入准备和模型推理是不同步骤，`--check-config` 不会执行前三者。

交付包无公开地址时标为“未核实公开下载地址”；仅支持加载时量化时标为“无离线 CLI”。

<a id="sources"></a>
## 逐模型查看官方来源与量化配方

每行的“官方来源”给出发布者、下载链接、存储精度和本地目录；“模块精度”覆盖该模型各类 TOML 的量化范围、浮点保留项及转换方式；“完整命令”注明脚本路径、输入、输出和运行配置。原始权重及输入资产不随本仓库提交。

| 模型 / 版本 | 官方下载与文件存储精度 | 各配置所需模块精度 | 下载、转换脚本与用法 |
|---|---|---|---|
| Qwen3 普通款 | [Qwen 官方](https://huggingface.co/Qwen/Qwen3-4B)：BF16；[全部尺寸及历史 Instruct 来源](qwen3/README.md#weight-sources) | [FP16 / W8 / W4、head 是否量化](qwen3/README.md#weight-recipes) | [完整命令](qwen3/README.md#quantization) |
| Qwen3-VL | [官方浮点父](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)：BF16；[cyankiwi AWQ](https://huggingface.co/cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit) 是第三方 W4/G32；[各尺寸来源](qwen3_vl/README.md#weight-sources) | [运行时 W8/W4、仅文本量化、AWQ 直用](qwen3_vl/README.md#weight-recipes) | [完整命令](qwen3_vl/README.md#quantization) |
| Qwen3.5 稠密模型 0.8–27B | [官方 27B](https://huggingface.co/Qwen/Qwen3.5-27B)及[各尺寸](qwen3_5/README.md#weight-sources)：BF16 主体＋少量 FP32 | [FP16 / W8 / W4、文本主干与 GDN/Vision 范围](qwen3_5/README.md#weight-recipes) | [完整命令](qwen3_5/README.md#quantization) |
| Qwen3.5 仅 MLP RTN / QuantTrio AWQ | [官方 2B 浮点父](https://huggingface.co/Qwen/Qwen3.5-2B)用于 RTN；[QuantTrio 社区 2B/4B/9B AWQ](qwen3_5/README.md#weight-sources)为公开量化包 | [MLP-only G32 与 AWQ G128 的首层、Attention/GDN 范围](qwen3_5/README.md#text-w4-recipes)，纯文本 B1/TP8 | [RTN 转换](qwen3_5/README.md#cmd-mlp-only-g32) / [AWQ 下载直用](qwen3_5/README.md#cmd-quanttrio-awq) |
| Qwen3.5 35B-A3B MoE | [官方父](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)：BF16＋FP32；[官方 GPTQ](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4)：路由专家 W4/G128；已供 mixed 包来源另列 | [RTN / GPTQ / mixed E4M3 的专家与非专家区别](qwen3_5/README.md#weight-recipes) | [完整命令](qwen3_5/README.md#quantization) |
| 历史 legacy27 | 已供 BF16 父；公开来源未核实，不能替代为官方 Qwen3.5-27B | [历史混合 W8/FP16 配方](qwen3_5/README.md#weight-recipes) | [归档说明与命令](qwen3_5/legacy/27b/README.md) |
| Llama 3.2-1B | [Meta 官方](https://huggingface.co/meta-llama/Llama-3.2-1B)：BF16 | [FP16 运行范围](llama/README.md#weight-recipes) | [下载后直接运行](llama/README.md#quantization) |
| Gemma4 E4B-it | [Google 官方](https://huggingface.co/google/gemma-4-E4B-it)：BF16 | [设备 FP16 与 CPU 浮点组件](gemma4/README.md#weight-recipes) | [下载后直接运行](gemma4/README.md#quantization) |
| DINOv3 ViT-B | [Meta 官方](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)：FP32 | [FP16 视觉编码器](dinov3/README.md#weight-recipes) | [下载后直接运行](dinov3/README.md#quantization) |
| Pi0.5 LIBERO | [LeRobot 微调权重](https://huggingface.co/lerobot/pi05_libero_finetuned_v044)：FP32＋BF16；不是 `pi05_base` | [四种精度的 Vision / Prefill / Action 配方](pi05/README.md#weight-recipes) | [下载及离线转换命令](pi05/README.md#quantization) |
| G0.5 K1 | 当前 `vla_rtc` 为受控交付；精确父的公开地址与发布 dtype 未核实 | [FP16 / W8 / 动作 NVFP4](g05/README.md#weight-recipes) | [取得配对交付包后加载时转换](g05/README.md#quantization) |
| Wall-OSS 基础模型 | [x-square-robot 官方](https://huggingface.co/x-square-robot/wall-oss-0.5)：FP32＋BF16 | [基础与旧版 W8 / W4 / NVFP4](wall_oss/README.md#weight-recipes) | [下载及各配方完整命令](wall_oss/README.md#quantization) |
| Wall-OSS RTC / 分拣 | WRC 为客户交付；公开下载与原存储 dtype 未核实，不能用公开 base 父替代 | [RTC / 分拣模块配方与配对目录](wall_oss/README.md#weight-recipes) | [WRC 转换及运行命令](wall_oss/README.md#quantization) |
| HyVLA | [腾讯 UMI 官方父](https://huggingface.co/tencent/Hy-Embodied-0.5-VLA-UMI)：BF16 | [FP16 / W8 / 动作 NVFP4](hy_vla/README.md#weight-recipes) | [下载与加载时转换](hy_vla/README.md#quantization) |
| HY Embodied 原 UMI | [腾讯同一官方父](https://huggingface.co/tencent/Hy-Embodied-0.5-VLA-UMI)：BF16；另需原配对输入包 | [原 UMI 浮点运行范围](hy_embodied/README.md#weight-recipes) | [下载与输入绑定](hy_embodied/README.md#quantization) |
| GR00T N1.7-3B | [NVIDIA 官方](https://huggingface.co/nvidia/GR00T-N1.7-3B)：BF16；[配套 Qwen 配置来源](gr00t/README.md#weight-sources) | [FP16 / W8 与保留 FP16 的输入投影](gr00t/README.md#weight-recipes) | [下载与加载时转换](gr00t/README.md#quantization) |
| RhinoVLA v1 | [HuixiAI 官方 `.ckpt`](https://huggingface.co/HuixiAI/RhinoVLA)：运行时为 FP16，原文件存储 dtype 未核实 | [v1 与 v3 分别说明](rhinovla/README.md#weight-recipes) | [固定下载及输入准备](rhinovla/v1/README.md#quantization) |
| RhinoVLA v3 | 客户 96D 权重与 factory；公开下载及发布 dtype 未核实 | [专家 W8 / 完整 W8 / 完整流水线范围](rhinovla/README.md#weight-recipes) | [加载时转换与调用者参数](rhinovla/v3/README.md#quantization) |
| InternVLA-N1 NavDP | [InternRobotics 官方](https://huggingface.co/InternRobotics/InternVLA-N1-w-NavDP)：BF16 | [System2 W8 与浮点 Vision / RGBD / NavDP](internvla/README.md#weight-recipes) | [提取、转换、准备输入的完整命令](internvla/README.md#quantization) |
| LingBot v2 | [Robbyant 官方](https://huggingface.co/robbyant/lingbot-vla-v2-6b)：FP32；旧 exact-Z2 BF16 派生父另列 | [FP16 / W8 的 Vision、文本与 MoE 范围](lingbot/v2/README.md#weight-recipes) | [下载与加载时转换](lingbot/v2/README.md#quantization) |
| LingBot v1 | [Robbyant 官方](https://huggingface.co/robbyant/lingbot-vla-4b)：FP32 | [CPU FP32 + 设备 FP16 混合执行](lingbot/README.md#weight-recipes) | [来源与使用说明](lingbot/README.md#quantization) |
| HALO 专家组件 | [官方预训练](https://huggingface.co/qshou-coder/halo_pt_weight) / [官方微调](https://huggingface.co/qshou-coder/halo_ft_weight)：BF16；当前已供 CoT EMA 资产的公开亲缘未核实 | [组件范围及不能替代的资产](halo/README.md#weight-recipes) | [原配对资产的运行说明](halo/README.md#quantization) |

<a id="quantization"></a>
## 先设置仓库与模型目录

下面直接给出调用命令。先激活已安装本项目及模型依赖的 Python 环境，把这两处 `/path/to` 换成实际目录，再执行所选模型、所选精度的一个命令块。**`SRC` 是已有的原始权重目录，`DST` 是本次生成的量化目录。** 输出目录应为新目录。

```bash
cd /path/to/rpu_backend
export RPU_MODEL_CACHE=/path/to/model-cache
```

离线命令只生成权重。Qwen 的 `DST` 使用模板实际查找的缓存路径；转换完成后选择对应 TOML。原始权重尚未下载时，先看本页[权重来源](#sources)。Wall 的交付量化脚本需在已安装匹配后端的环境运行。

| 要执行什么 | 仓库中的转换脚本 / 方式 | 本页完整命令 |
|---|---|---|
| Qwen3 W8 / 加载时 W4 | W8：[convert_qwen3.py](../../python/rpu_backend/quant/convert_qwen3.py)；W4：运行入口加载时量化 | [Qwen3](#cmd-qwen3) |
| Qwen3-VL W8、AWQ→W4、32B W4、仅文本 W8 | [convert_qwen3_vl.py](../../python/rpu_backend/quant/convert_qwen3_vl.py)；仅文本 W8 用 [convert_qwen3.py](../../python/rpu_backend/quant/convert_qwen3.py) | [Qwen3-VL](#cmd-qwen3-vl) |
| Qwen3.5 稠密模型 W8/W4，含官方27B | [convert_qwen3_5.py](../../python/rpu_backend/quant/convert_qwen3_5.py) | [稠密模型](#cmd-qwen35-dense) |
| Qwen3.5 2B MLP-only RTN / 2B/4B/9B AWQ | RTN：Dense 脚本 `--profile w4_mlp_only`；QuantTrio AWQ：直接下载 | [四个文本配方](#cmd-qwen35-text-w4) |
| Qwen3.5 MoE RTN W4 / 官方 GPTQ | RTN：[convert_qwen3_5_moe_rtn.py](../../python/rpu_backend/quant/convert_qwen3_5_moe_rtn.py)；GPTQ：直接下载 | [MoE](#cmd-qwen35-moe) |
| Pi0.5 W8 / 动作 NVFP4 | [convert_pi05.py](../../python/rpu_backend/quant/convert_pi05.py) | [Pi0.5](#cmd-pi05) |
| Wall-OSS W8 / W4 / 分拣动作 W4 | W8：[convert_wall_oss_w8a16.py](../../python/rpu_backend/quant/convert_wall_oss_w8a16.py)；W4：[convert_wall_oss.py](../../python/rpu_backend/quant/convert_wall_oss.py) | [Wall-OSS](#cmd-wall) |
| InternVLA System2 W8 配对包 | 由资产提供方准备完整 bundle | [输入与运行](#cmd-internvla) |
| LingBot v2、G05、HyVLA、GR00T、RhinoVLA 的加载时量化 | 无独立离线 CLI；运行入口按 TOML 加载并转换，源码位置见各模型页 | [运行命令](#runtime-quantization) |
| 历史 legacy27 | [convert_qwen3_5.py](../../python/rpu_backend/quant/convert_qwen3_5.py)；必须选择历史混合配方 | [旧资产转换](#cmd-legacy27) |

<a id="offline"></a>
<a id="cmd-qwen3"></a>
## Qwen3

**W8A16，以下以4B为例。** `SRC` 可改为已有官方4B浮点目录；`DST` 对应 `text/4b/w8a16.toml`。独立 head 必须使用 `--quant-lm-head`。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3-4B"
DST="$RPU_MODEL_CACHE/qwen3_lmhead_int8_ckpts/qwen3-4b-w8a16-lmhead-int8"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3.py \
  --src "$SRC" --dst "$DST" --quant-lm-head
```

**W4/G32 在加载时转换，不先生成 W4 目录。** 把浮点父放在 `$RPU_MODEL_CACHE/Qwen3-4B` 后，下面分别是保留 FP16 head、量化 head 的运行命令，选择所需的一条；需要可用板卡。

```bash
# Text7 W4，head FP16
python examples/run.py --config examples/configs/qwen3/text/4b/w4a16.toml
# Text7 与 head 均 W4
python examples/run.py --config examples/configs/qwen3/text/4b/w4a16_lm_head.toml
```

其他大小的精确缓存目录、32B分层加载与源码见 [Qwen3 命令说明](qwen3/README.md#quantization)。

<a id="cmd-qwen3-vl"></a>
## Qwen3-VL

**runtime W8，4B示例：** 输入官方浮点 checkpoint，输出供 `qwen3_vl/vl/4b/w8a16.toml` 使用。这个脚本必须按下方 `python -m` 形式调用。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3-VL-4B-Instruct"
DST="$RPU_MODEL_CACHE/Qwen3-VL-4B-Instruct-runtime-w8"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=python python -m rpu_backend.quant.convert_qwen3_vl \
  --bits 8 --src "$SRC" --dst "$DST"
```

**2/4/8B runtime W4，以下以4B为例：** 输入是指定的公开 AWQ 包，不能填写普通浮点父目录。输出供 `qwen3_vl/vl/4b/w4a16.toml` 使用。

```bash
# 已有同一 AWQ 包时跳过下载。
hf download cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit \
  --local-dir "$RPU_MODEL_CACHE/Qwen3-VL-4B-Instruct-AWQ"
SRC="$RPU_MODEL_CACHE/Qwen3-VL-4B-Instruct-AWQ"
DST="$RPU_MODEL_CACHE/Qwen3-VL-4B-Instruct-runtime-w4-awq"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=python python -m rpu_backend.quant.convert_qwen3_vl \
  --bits 4 --src "$SRC" --dst "$DST"
```

**32B runtime W4：** 此项输入官方32B浮点父，输出供 `qwen3_vl/vl/32b/w4a16.toml` 使用。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3-VL-32B-Instruct"
DST="$RPU_MODEL_CACHE/Qwen3-VL-32B-Instruct-runtime-w4"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=python python -m rpu_backend.quant.convert_qwen3_vl \
  --bits 4 --src "$SRC" --dst "$DST"
```

**2B Text-only W8：** 保留 Vision/head 为浮点，输出供 `qwen3_vl/vl/2b/w8a16_text.toml` 使用。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3-VL-2B-Instruct"
DST="$RPU_MODEL_CACHE/Qwen3-VL-2B-Instruct-w8a16-text"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3.py \
  --src "$SRC" --dst "$DST" --skip-modules model.visual lm_head
```

独立 Text-AWQ 模板直接使用下载的 AWQ，无需上面的 runtime 转换。其他大小、32B W8、Text-AWQ 与源码见 [Qwen3-VL 命令说明](qwen3_vl/README.md#quantization)。

<a id="cmd-qwen35-dense"></a>
## Qwen3.5 Dense：官方27B W8 / W4

以下两种精度分别输出到对应公开模板查找的目录。`SRC` 必须是官方 `Qwen/Qwen3.5-27B` 浮点 checkpoint。0.8/2/4/9B 的相同转换方法和各自缓存目录见 [Dense 命令说明](qwen3_5/README.md#quantization)。

**W8A16：** 输出供 `qwen3_5/text/27b/w8a16.toml` 和相应 VL 模板使用。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3.5-27B"
DST="$RPU_MODEL_CACHE/qwen3_5_quant_ckpts/qwen3_5-27b-w8a16"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3_5.py \
  --src "$SRC" --dst "$DST" --method w8a16
```

**W4A16/G32：** 输出供 `qwen3_5/text/27b/w4a16.toml` 和相应 VL 模板使用。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3.5-27B"
DST="$RPU_MODEL_CACHE/qwen3_5_quant_ckpts/qwen3_5-27b-w4a16"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3_5.py \
  --src "$SRC" --dst "$DST" --method w4a16_pgrp --group-size 32
```

<a id="cmd-qwen35-text-w4"></a>
## Qwen3.5 仅 MLP RTN / QuantTrio AWQ

四个配方均为纯文本、B1、TP8；短示例和固定长度模板共用各自权重。普通 uniform W4 不能通过重命名充当 MLP-only。四个 alias 与缓存目录见[模型指南](qwen3_5/README.md#text-w4-recipes)。

**2B MLP-only RTN G32** 使用官方浮点父，量化全部 24 层的 72 个 MLP gate/up/down，Attention/GDN 及其他部分 FP16。先[下载父](qwen3_5/README.md#cmd-mlp-only-g32)，再执行：

```bash
SRC="$RPU_MODEL_CACHE/Qwen3.5-2B"
DST="$RPU_MODEL_CACHE/qwen3_5-2b-w4a16-g32-mlp-only"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3_5.py \
  --src "$SRC" --dst "$DST" --profile w4_mlp_only
python examples/run.py --config examples/configs/qwen3_5/text/2b/w4_mlp_only_g32.toml
```

**QuantTrio AWQ G128** 是社区原包，2B/4B/9B 仅第 1 层起的 MLP 量化，分别 69/93/93 个投影；首层 MLP 和 Attention/GDN FP16。源 qweight/qzeros 为 I32 打包容器，逻辑权重为 UINT4，scale 为 FP16。下载后直接运行，不做 RTN 转换或再量化：

```bash
SIZE=2  # 可选：2、4、9
hf download "QuantTrio/Qwen3.5-${SIZE}B-AWQ" \
  --local-dir "$RPU_MODEL_CACHE/Qwen3.5-${SIZE}B-AWQ"
python examples/run.py --config "examples/configs/qwen3_5/text/${SIZE}b/quanttrio_awq_g128.toml"
```

loader 不写回源配置/分片，保留整数码、direct zero-point、原 FP16 scale，未量化参数直接转 FP16。需要匹配的 native 扩展；下载、转换和 `--check-config` 只检查配置与资产格式，不代表设备验证。

<a id="cmd-qwen35-moe"></a>
## Qwen3.5 35B-A3B MoE

**RTN W4/G128：** 从官方未量化35B-A3B父生成新 checkpoint，供 `qwen3_5/text/35b_a3b/rtn_w4a16.toml` 及对应 VL 使用。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3.5-35B-A3B"
DST="$RPU_MODEL_CACHE/Qwen3.5-35B-A3B-RTN-W4A16-G128"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3_5_moe_rtn.py \
  --src "$SRC" --dst "$DST" --threads 8
```

**官方 GPTQ W4：** 已是量化权重，直接下载，不执行本地 RTN 转换。

```bash
hf download Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --local-dir "$RPU_MODEL_CACHE/Qwen3.5-35B-A3B-GPTQ-Int4"
```

`mixed_e4m3.toml` 使用原已供 mixed 包，尚无可公开复现的完整 producer，不能将官方 block-FP8 包改名替代。[配方与源码](qwen3_5/README.md#quantization)。

<a id="cmd-pi05"></a>
## Pi0.5

输入均为原始 LIBERO 微调父目录。**W8A16：** 转换完成后将 Pi0.5 W8 模板的 `input.checkpoint` 设置为 `DST`。

```bash
SRC="$RPU_MODEL_CACHE/pi05_libero_finetuned"
DST="$RPU_MODEL_CACHE/pi05_libero_finetuned-w8a16-vlm-expert"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_pi05.py \
  --src "$SRC" --dst "$DST"
```

**W8 + Action NVFP4：** 将 `w8a16_action_nvfp4.toml` 的 `input.checkpoint` 设置为 `DST`。Prefill W8A8 + Action NVFP4 模板也使用这个目录，不另生成 W8A8 权重。

```bash
SRC="$RPU_MODEL_CACHE/pi05_libero_finetuned"
DST="$RPU_MODEL_CACHE/pi05_action_nvfp4"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_pi05.py \
  --src "$SRC" --dst "$DST" --action-w4 --action-w4-format nvfp4
```

下载固定父、配置修改和后续运行见 [Pi0.5 命令说明](pi05/README.md#quantization)。旧 `int4_g128` 格式不能用于 NVFP4 模板。

<a id="cmd-wall"></a>
## Wall-OSS

**开源 base W8：** 源目录和输出保持兄弟关系。将 base 模板的 `input.fp16_checkpoint` / `input.w8_checkpoint` 分别设置为 `SRC` / `DST`。

```bash
SRC="$RPU_MODEL_CACHE/wall-oss-0.5"
DST="$RPU_MODEL_CACHE/wall-oss-0.5-w8a16"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_wall_oss_w8a16.py \
  --src "$SRC" --dst "$DST"
```

**RTC / sorting W8：** 输入必须是原 WRC 交付父。RTC 模板的 `input.fp16_checkpoint` 填 `SRC`、`input.checkpoint` 填 `DST`；sorting 模板的 `input.fp16_checkpoint` 填 `SRC`、`input.w8_checkpoint` 填 `DST`。

```bash
SRC="$RPU_MODEL_CACHE/wrc_sf_v728_0_15000"
DST="$RPU_MODEL_CACHE/wrc_sf_v728_0_15000-w8a16"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_wall_oss_w8a16.py \
  --src "$SRC" --dst "$DST"
```

**sorting Action W4/G32：** 从原始 WRC 父转换两个 decoder expert 的全部投影为 INT4；运行时只取 Action expert。把 `DST` 填入 sorting 模板的 `input.action_w4_checkpoint`，Vision/Prefill 仍使用上一步 W8 目录。

```bash
SRC="$RPU_MODEL_CACHE/wrc_sf_v728_0_15000"
DST="$RPU_MODEL_CACHE/wrc_sf_v728_0_15000-action-w4-g32"
python -m rpu_backend.quant.convert_wall_oss \
  --src "$SRC" --dst "$DST" --group-size 32
```

**legacy decoder W4，逐通道 group0：** 使用开源 base 父，生成八类投影 INT4。

```bash
SRC="$RPU_MODEL_CACHE/wall-oss-0.5"
DST="$RPU_MODEL_CACHE/wall-oss-0.5-w4a16"
python -m rpu_backend.quant.convert_wall_oss \
  --src "$SRC" --dst "$DST" --group-size 0
```

**legacy decoder W4-pgrp/G32：** 两个 decoder expert 的全部投影均使用 G32 INT4，包含 expert0 Down；模块范围见 [Wall 模块配方](wall_oss/README.md#weight-recipes)。

```bash
SRC="$RPU_MODEL_CACHE/wall-oss-0.5"
DST="$RPU_MODEL_CACHE/wall-oss-0.5-w4a16-pgrp"
python -m rpu_backend.quant.convert_wall_oss \
  --src "$SRC" --dst "$DST" --group-size 32
```

legacy 还需同父 W8 目录提供 Vision，按[legacy 配置说明](wall_oss/legacy/README.md#quantization)放置资产。base Action NVFP4 无离线导出命令，按下方[加载时运行](#runtime-quantization)处理。

<a id="cmd-internvla"></a>
## InternVLA：System2 W8 配对包

向资产提供方取得 `internvla_n1-controlled/` 完整包，包括逐通道 W8 decoder、浮点视觉/NavDP 组件、输入 fixture 和 manifest。固定 checkpoint 与源码下载命令、W/A 配方及运行步骤见 [InternVLA](internvla/README.md#quantization)。

<a id="cmd-legacy27"></a>
## 历史 legacy27：W8 主干 / FP16 GDN

使用原已供 exact legacy BF16 父及原始配置；公开父下载来源未核实。该配方与官方 Qwen3.5-27B 分开使用。

```bash
SRC=/path/to/original-exact-legacy27-bf16
DST="$RPU_MODEL_CACHE/Qwen3.8-27B-w8a16"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3_5.py \
  --src "$SRC" --dst "$DST" --profile w8_main_fp16_gdn_legacy
```

输出供历史 `qwen3_5/legacy/27b/` 模板使用；新输出不保证与历史分片字节一致。[来源、准入与复现](qwen3_5/README.md#quantization)。

<a id="runtime-quantization"></a>
## 加载时量化：填写配置后直接运行

这些模型没有独立的量化导出命令。下列命令会安装量化权重并运行模型，**需要可用板卡，并准备相应模板要求的输入**，不生成离线 checkpoint。先复制模板，按每块列明的字段编辑，再运行；`--check-config` 不执行转换。

**LingBot v2 W8，公开三相机T3：** 修改下方两个路径；原权重在前面设置的 `$RPU_MODEL_CACHE/lingbot-vla-v2-6b`。

```bash
export LINGBOT2_TRAINING_CONFIG=/path/to/paired-training.yaml
export LINGBOT2_PUBLIC_CASE=/path/to/3cam-t3-case
cp examples/configs/lingbot/v2/3cam_t3/w8a16.toml /tmp/lingbot2_w8.toml
# 将 run.output_dir 改为本次输出目录。
vi /tmp/lingbot2_w8.toml
python examples/run.py --config /tmp/lingbot2_w8.toml
```

**G0.5 W8：** 原始 K1 父、配对 release/输入和三份库路径都在配置中填写。

```bash
cp examples/configs/g05/2cam/w8a16.toml /tmp/g05_w8.toml
# 填 input.checkpoint、input.release_root、input.source_golden、run.output_dir。
# [libraries] 填 native/main/expansion 实际路径，或删除整表使用当前环境。
vi /tmp/g05_w8.toml
python examples/run.py --config /tmp/g05_w8.toml
```

**HyVLA W8：** 直接填写原浮点父目录。

```bash
cp examples/configs/hy_vla/2cam/w8a16.toml /tmp/hyvla_w8.toml
# 填 input.checkpoint、run.output_dir，以及 [libraries] 的 native/main/expansion。
vi /tmp/hyvla_w8.toml
python examples/run.py --config /tmp/hyvla_w8.toml
```

G0.5 / HyVLA 的 Action NVFP4 各有完整独立调用块：[G0.5](g05/README.md#quantization)、[HyVLA](hy_vla/README.md#quantization)。

**GR00T N1.7 W8：** 缓存中须有原 GR00T 权重和匹配的 Qwen3-VL 配套资产，见[来源](gr00t/README.md#weights)。

```bash
cp examples/configs/gr00t/n1_7_droid/w8a16.toml /tmp/gr00t_w8.toml
# 填 input.observation（真实CPU输入文件）及 run.output_dir。
vi /tmp/gr00t_w8.toml
python examples/run.py --config /tmp/gr00t_w8.toml
```

**Wall-OSS base W8 + Action NVFP4：** 先完成上方 base W8 转换，再填配对目录。

```bash
cp examples/configs/wall_oss/base/3cam/224/w8a16_action_nvfp4.toml /tmp/wall_nvfp4.toml
# 填 input.fp16_checkpoint、input.w8_checkpoint、input.batch_template 和 run.output_dir。
# batch_template 保留 {cameras}/{image_size}/{text_tokens}，展开后须指向真实输入文件。
vi /tmp/wall_nvfp4.toml
python examples/run.py --config /tmp/wall_nvfp4.toml
```

**RhinoVLA v3 expert W8：** 需客户 checkpoint 与可导入的可信 factory。

```bash
cp examples/configs/rhinovla/v3/expert_w8a16.toml /tmp/rhino_expert_w8.toml
# 填 input.checkpoint/runtime_factory/request/factory_config，并在 [run] 添加 output_dir。
# 保留模板的量化开关；factory JSON 必须匹配这些开关。
vi /tmp/rhino_expert_w8.toml
python examples/run.py --config /tmp/rhino_expert_w8.toml
```

其他 W8 范围的完整命令见 [RhinoVLA v3](rhinovla/v3/README.md#quantization)。上述模型的转换实现路径和调用函数放在各模型命令之后，便于检查源码。

<a id="no-local-converter"></a>
## 不需要量化转换的配置

当前 Llama、Gemma4、DINOv3、HY Embodied 原 UMI、HALO、RhinoVLA v1、LingBot v1 模板直接使用各自浮点资产，没有需要先调用的量化脚本。Qwen3-VL 独立 Text-AWQ 与 MoE 官方 GPTQ 使用对应公开量化包。

## 下载与缓存

各族命令在仓库根目录执行，使用已安装本项目及所需 Torch、Transformers、safetensors 的 Python 环境。[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) 提供 `hf download`；受限仓库需先按模型页面获得访问权限，再通过本机 HF 登录配置访问。

先把 `RPU_MODEL_CACHE` 设为实际缓存目录。每个族的表格写出该目录下的相对路径，例如：

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
mkdir -p "$RPU_MODEL_CACHE"
hf download Qwen/Qwen3-4B --local-dir "$RPU_MODEL_CACHE/Qwen3-4B"
```

本地目录名由 [model registry](../../python/rpu_backend/model_registry.py) 解析，可能与 HF 仓库的大小写不同。各族使用其 README 指定的路径字段；`--model` 当前仅供 Qwen3.5 MoE 使用，不是所有模板的通用覆盖参数。下载完整 snapshot，保留 config、权重 index/全部分片、generation config、tokenizer 和 processor；不要只复制一个权重文件。

## 转换与运行

各模型指南页首先给出转换或加载时量化命令、输入输出及模板。按该节选择独立 CPU 环境或已安装后端的环境：`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 可以关闭 Torch 的自动后端加载，但不阻止脚本显式导入后端；Wall 转换器 需匹配的后端环境。目标目录须不存在；保留原 checkpoint，将输出放入新目录，再选择匹配精度的模板。

AWQ、GPTQ、RTN、NVFP4、混合 E4M3，以及“只量化 Text7”与“量化 Text7 + head/Vision”是不同配方。改变 TOML 的 dtype 或改目录名不能完成格式转换；缺少公开 producer/来源的已供资产在各族说明中明确列出，不提供猜测的下载或转换命令。

完成资产准备后，按[配置指南](README.md)运行。`--check-config` 检查模板语法和身份，不会下载、转换或验证权重内容；公开来源可下载也不扩大本项目登记的尺寸、精度或执行范围。
