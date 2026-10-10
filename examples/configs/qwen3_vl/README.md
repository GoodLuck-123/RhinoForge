# Qwen3-VL 配置
[配置索引](../CONFIGURATIONS.md) · [使用](#使用)

`text`只生成文字，`vl`完整图文生成，`vision`只运行视觉编码器。图文入口从 [vl/2b/fp16.toml](vl/2b/fp16.toml) 开始；按下表选择模型尺寸与精度。

<a id="weights"></a>
<a id="weight-sources"></a>
## 权重来源与存储类型

下表按发布仓库的 config 与 safetensors 元数据核对（2026-10-10）；**官方父存储为 BF16，FP16 是本项目加载后的运行精度**。路径均相对于 `$RPU_MODEL_CACHE`，需完整下载 config、tokenizer、processor 和全部分片。

| 权重 / 用途 | 磁盘存储精度 | 发布者与下载来源 | 下载到的目录 |
|---|---|---|---|
| Qwen3-VL-2B-Instruct：浮点模板与转换父 | BF16 | Qwen 官方 [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-VL-2B-Instruct) | `Qwen3-VL-2B-Instruct` |
| Qwen3-VL-4B-Instruct：浮点模板与转换父 | BF16 | Qwen 官方 [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct) | `Qwen3-VL-4B-Instruct` |
| Qwen3-VL-8B-Instruct：浮点模板与转换父 | BF16 | Qwen 官方 [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-VL-8B-Instruct) | `Qwen3-VL-8B-Instruct` |
| Qwen3-VL-32B-Instruct：浮点模板与转换父 | BF16 | Qwen 官方 [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-VL-32B-Instruct) | `Qwen3-VL-32B-Instruct` |
| 2B AWQ：Text-AWQ / 运行时 W4 的输入 | Text7 为 **INT4/G32**，I32 packed；scale 与未量化 companion 为 BF16；shape 为 I64 | **第三方 cyankiwi** [Qwen3-VL-2B-Instruct-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit) · [量化声明](https://huggingface.co/cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit/blob/main/config.json) | `Qwen3-VL-2B-Instruct-AWQ` |
| 4B AWQ：Text-AWQ / 运行时 W4 的输入 | Text7 为 **INT4/G32**，I32 packed；scale 与未量化 companion 为 BF16；shape 为 I64 | **第三方 cyankiwi** [Qwen3-VL-4B-Instruct-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit) · [量化声明](https://huggingface.co/cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit/blob/main/config.json) | `Qwen3-VL-4B-Instruct-AWQ` |
| 8B AWQ：仅运行时 W4 的转换输入（无独立 Text-AWQ 模板） | Text7 为 **INT4/G32**，I32 packed；scale 与未量化 companion 为 BF16；shape 为 I64 | **第三方 cyankiwi** [Qwen3-VL-8B-Instruct-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit) · [量化声明](https://huggingface.co/cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit/blob/main/config.json) | `Qwen3-VL-8B-Instruct-AWQ` |
| 本仓库运行时 W8/W4 派生包 | Text7/head：INT8 或 packed INT4，FP16 scale；其余 FP16 | 本仓库离线转换，不是 Qwen 官方或 cyankiwi 原包 | `Qwen3-VL-{SIZE}B-Instruct-runtime-w8`；2/4/8B 的 `-runtime-w4-awq`；32B 的 `-runtime-w4` |
| 2B Text-W8 派生包 | Text7 INT8 + FP16 scale；其余 FP16 | 本仓库离线转换 | `Qwen3-VL-2B-Instruct-w8a16-text` |
| 32B legacy Text-W8 已供包 | Text7 INT8 + FP16 scale；head/Vision 等 FP16 | 原历史包；下方可生成新同格式资产，不保证历史字节相同 | `Qwen3-VL-32B-Instruct-w8a16-full-20260816` |

32B 的 BF16 父只用于转换；没有32B FP16运行模板。直接消费 AWQ 的 `w4a16_text_awq.toml` 仅有2/4B，8B AWQ须先转换为运行时 W4。AWQ 包使用发布者自己的完整 companion，不能把官方 BF16 的 embedding/norm/Vision 随意拼入它。

<a id="weight-recipes"></a>
## 模块精度与配方

**Text7** = 每层 Attention 的 Q/K/V/O 投影与 MLP 的 gate/up/down 投影。下面 A16 为 FP16 主路径激活；W8 为逐输出通道对称 INT8，W4 为 G32 对称 INT4。所有长度变体沿用其基础模板的资产与配方，核数/累加精度不生成另一份权重。

| config 配方 | 运行时模块权重 / 激活 | 输入来源 | 获取方式、源码与完整命令 |
|---|---|---|---|
| `text/`、`vl/`、`vision/` 的 2/4/8B `fp16*.toml` | 实际执行的 Text7/head/embedding/norm/Vision 全 FP16；A16 | 对应官方 BF16 父 | 不量化，加载时转 FP16；[examples/runners/qwen3_vl/runner.py](../../runners/qwen3_vl/runner.py)；[完整下载/运行](#cmd-fp16) |
| `text/`、`vl/` 的 2/4/8/32B `w8a16.toml` | Text7 + **head W8**；**Vision W8**；embedding/norm/bias FP16；A16 | 对应官方 BF16 父 | Text/head 离线，[python/rpu_backend/quant/convert_qwen3_vl.py](../../../python/rpu_backend/quant/convert_qwen3_vl.py) `--bits 8`；Vision 加载时转换；[完整命令](#runtime-w8官方浮点父--text7head-w8) |
| `text/`、`vl/` 的 2/4/8B `w4a16.toml` | Text7 **保留 AWQ W4/G32**；head RTN W4/G32；**Vision W8**；embedding/norm/bias FP16；A16 | 对应 cyankiwi AWQ **整包** | 离线保留/重排 AWQ + 量化 head；同脚本 `--bits 4`；[完整命令](#runtime-w4g32248b-awq--runtime-w4) |
| `text/32b/w4a16.toml`、`vl/32b/w4a16.toml` | Text7 + head **RTN W4/G32**；Vision W8；embedding/norm/bias FP16；A16 | 官方32B BF16父 | 离线，同脚本 `--bits 4`；[完整命令](#runtime-w4g3232b-官方浮点父--runtime-w4) |
| 2B `text/`、`vl/` 的 `w8a16_text.toml` | **仅 Text7 W8**；head/Vision/embedding/norm FP16；A16 | 官方2B BF16父 | 离线，[python/rpu_backend/quant/convert_qwen3.py](../../../python/rpu_backend/quant/convert_qwen3.py)，`--skip-modules model.visual lm_head`，不加量化 head/embedding 选项；[完整命令](#text-only-w82b只量化text7headvision保留fp16) |
| 2/4B `text/`、`vl/` 的 `w4a16_text_awq.toml` | **仅 Text7 AWQ W4/G32**；head/Vision/embedding/norm FP16；A16 | 对应 cyankiwi AWQ 整包 | 直接下载；[python/rpu_backend/quant/load_qwen3_vl_awq.py](../../../python/rpu_backend/quant/load_qwen3_vl_awq.py) 加载/重排；[完整命令](#text-only-awq24b直接下载不再转换) |
| `vl/32b/w8a16_legacy.toml` | 仅 Text7 W8；head/Vision/embedding/norm FP16；A16 | 历史同包；或官方32B父生成的新同格式包 | 离线 [python/rpu_backend/quant/convert_qwen3.py](../../../python/rpu_backend/quant/convert_qwen3.py)；[完整命令与历史限制](#32b-legacy-text-w8原包复现--新的同格式转换) |

运行时 W8/W4 的离线文件里，**Vision 仍存 FP16**；安装时由 [python/rpu_backend/adapters/qwen3_vl/vision.py](../../../python/rpu_backend/adapters/qwen3_vl/vision.py) 的 `install_qwen3_vl_vision_for_rpu` 转换 patch embedding、各 block 的 Q/K/V/O/MLP、末端 merger 和 DeepStack merger 为 W8。norm、bias、位置参数保留 FP16。Text-only 配方不启用这一步。

W4 运行时派生包以 UINT8 packed 两个4bit值及 FP16 scale 保存；公开 AWQ 原包是 compressed-tensors 的 I32 packed 格式，必须由对应 loader/转换器处理，不能靠重命名目录互换。

<a id="quantization"></a>
## 下载与转换

**离线转换可在 CPU 机器完成；各命令块最后一行 `python examples/run.py ...` 是模型推理，需在可用的 设备 板卡上执行。若当前是 CPU 转换机，请只执行下载和转换部分，再将产物按相同缓存路径放到板卡。**

以下 **Bash 命令在仓库根目录、已安装项目依赖的 Python 环境中执行**。先替换缓存根目录：

```bash
export RPU_MODEL_CACHE=/absolute/path/to/model_cache
```

`SRC` 为完整输入 checkpoint，`DST` 为尚不存在的输出目录。已下载时可修改 `SRC` 并跳过下载；保持下面 `DST`，对应模板的 `example.registry_alias` 会自动读取它。此族不支持 `--model` 或 `input.checkpoint`。命令先用 `text/` 模板运行；需图文时用同大小同精度 `vl/` 模板并填写 `input.images`，仍读同一 `DST`。

<a id="cmd-fp16"></a>
### FP16：官方 BF16 下载后直接运行

```bash
SIZE=4  # 可选：2、4、8
hf download "Qwen/Qwen3-VL-${SIZE}B-Instruct" \
  --local-dir "$RPU_MODEL_CACHE/Qwen3-VL-${SIZE}B-Instruct"
python examples/run.py --config "examples/configs/qwen3_vl/text/${SIZE}b/fp16.toml"
```

图文/纯视觉改用相同大小的 `vl/${SIZE}b/fp16.toml` 或 `vision/${SIZE}b/fp16.toml`，先填写该模板 `input.images`，无需另一份权重。

<a id="runtime-w8官方浮点父--text7head-w8"></a>
### 运行时 W8：官方浮点父 → Text7/输出头 W8

只改 `SIZE` 选择2/4/8/32B。转换器必须使用 `python -m` 调用；Vision权重在加载时转W8，不需要第二个离线目录。

```bash
SIZE=4  # 可选：2、4、8、32
SRC="$RPU_MODEL_CACHE/Qwen3-VL-${SIZE}B-Instruct"
DST="$RPU_MODEL_CACHE/Qwen3-VL-${SIZE}B-Instruct-runtime-w8"
hf download "Qwen/Qwen3-VL-${SIZE}B-Instruct" --local-dir "$SRC"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=python python -m rpu_backend.quant.convert_qwen3_vl \
  --src "$SRC" --dst "$DST" --bits 8
python examples/run.py --config "examples/configs/qwen3_vl/text/${SIZE}b/w8a16.toml"
```

例如4B输出供 [text/4b/w8a16.toml](text/4b/w8a16.toml) / [vl/4b/w8a16.toml](vl/4b/w8a16.toml) 使用。

<a id="runtime-w4g32248b-awq--runtime-w4"></a>
### 运行时 W4/G32：2/4/8B AWQ → 运行时 W4

**输入必须是下列 cyankiwi AWQ 完整包，不是官方浮点目录。** 保留该包Text7 AWQ及浮点 companion，转换器将head转RTN W4；Vision加载时转W8。只改 `SIZE`：

```bash
SIZE=4  # 可选：2、4、8
SRC="$RPU_MODEL_CACHE/Qwen3-VL-${SIZE}B-Instruct-AWQ"
DST="$RPU_MODEL_CACHE/Qwen3-VL-${SIZE}B-Instruct-runtime-w4-awq"
hf download "cyankiwi/Qwen3-VL-${SIZE}B-Instruct-AWQ-4bit" --local-dir "$SRC"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=python python -m rpu_backend.quant.convert_qwen3_vl \
  --src "$SRC" --dst "$DST" --bits 4
python examples/run.py --config "examples/configs/qwen3_vl/text/${SIZE}b/w4a16.toml"
```

例如4B输出供 [text/4b/w4a16.toml](text/4b/w4a16.toml) / [vl/4b/w4a16.toml](vl/4b/w4a16.toml) 使用。

<a id="runtime-w4g3232b-官方浮点父--runtime-w4"></a>
### 运行时 W4/G32：32B 官方浮点父 → 运行时 W4

32B使用浮点父，Text7/head均为RTN W4；不要套用上面的AWQ源命令。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3-VL-32B-Instruct"
DST="$RPU_MODEL_CACHE/Qwen3-VL-32B-Instruct-runtime-w4"
hf download Qwen/Qwen3-VL-32B-Instruct --local-dir "$SRC"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=python python -m rpu_backend.quant.convert_qwen3_vl \
  --src "$SRC" --dst "$DST" --bits 4
python examples/run.py --config examples/configs/qwen3_vl/text/32b/w4a16.toml
```

输出也供 [vl/32b/w4a16.toml](vl/32b/w4a16.toml) 使用。以上三类离线入口源码为 [python/rpu_backend/quant/convert_qwen3_vl.py](../../../python/rpu_backend/quant/convert_qwen3_vl.py)。

<a id="text-only-w82b只量化text7headvision保留fp16"></a>
### 仅文本量化 W8：2B只量化Text7，输出头/视觉保留FP16

这里的“Text-only”指**量化范围**；产物同时供 text / VL 的 `w8a16_text.toml` 使用，与 runtime W8 目录不同。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3-VL-2B-Instruct"
DST="$RPU_MODEL_CACHE/Qwen3-VL-2B-Instruct-w8a16-text"
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir "$SRC"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3.py \
  --src "$SRC" --dst "$DST" --skip-modules model.visual lm_head
python examples/run.py --config examples/configs/qwen3_vl/text/2b/w8a16_text.toml
```

入口为 [python/rpu_backend/quant/convert_qwen3.py](../../../python/rpu_backend/quant/convert_qwen3.py)；不加 `--quant-lm-head`。

<a id="text-only-awq24b直接下载不再转换"></a>
### 仅文本量化 AWQ：2/4B直接下载，不再转换

head/Vision保留FP16；以下目录直接供 `w4a16_text_awq.toml`，**不要**指向上述 runtime W4 输出。只改 `SIZE`：

```bash
SIZE=2  # 可选：2、4；8B无独立Text-AWQ模板
DST="$RPU_MODEL_CACHE/Qwen3-VL-${SIZE}B-Instruct-AWQ"
hf download "cyankiwi/Qwen3-VL-${SIZE}B-Instruct-AWQ-4bit" --local-dir "$DST"
python examples/run.py --config "examples/configs/qwen3_vl/text/${SIZE}b/w4a16_text_awq.toml"
```

<a id="32b-legacy-text-w8原包复现--新的同格式转换"></a>
### 32B 旧版文本 W8：原包复现 / 新的同格式转换

原包目录为 `Qwen3-VL-32B-Instruct-w8a16-full-20260816`。下面是从官方浮点父生成**新同格式资产**的完整调用，不证明新输出与历史包字节相同。`DST` 必须是新目录；历史原包已在此处时不要重新转换。它只用于 [vl/32b/w8a16_legacy.toml](vl/32b/w8a16_legacy.toml)，图像路径填写该模板 `input.images` 后运行。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3-VL-32B-Instruct"
DST="$RPU_MODEL_CACHE/Qwen3-VL-32B-Instruct-w8a16-full-20260816"
hf download Qwen/Qwen3-VL-32B-Instruct --local-dir "$SRC"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3.py \
  --src "$SRC" --dst "$DST" --skip-modules lm_head
python examples/run.py --config examples/configs/qwen3_vl/vl/32b/w8a16_legacy.toml
```

FP16模板无需量化，直接读取[权重来源](#weights)的官方浮点目录。

## 选择配置

<a id="runtime-w8a16--w4a16"></a>
| 模型 | FP16 | runtime W8 | runtime W4/G32 |
|---|---|---|---|
| 2B | [text](text/2b/fp16.toml) / [VL](vl/2b/fp16.toml) / [vision](vision/2b/fp16.toml) | [text](text/2b/w8a16.toml) / [VL](vl/2b/w8a16.toml) | [text](text/2b/w4a16.toml) / [VL](vl/2b/w4a16.toml) |
| 4B | [text](text/4b/fp16.toml) / [VL](vl/4b/fp16.toml) / [vision](vision/4b/fp16.toml) | [text](text/4b/w8a16.toml) / [VL](vl/4b/w8a16.toml) | [text](text/4b/w4a16.toml) / [VL](vl/4b/w4a16.toml) |
| 8B | [text](text/8b/fp16.toml) / [VL](vl/8b/fp16.toml) / [vision](vision/8b/fp16.toml) | [text](text/8b/w8a16.toml) / [VL](vl/8b/w8a16.toml) | [text](text/8b/w4a16.toml) / [VL](vl/8b/w4a16.toml) |
| 32B | 不支持 | [text](text/32b/w8a16.toml) / [VL](vl/32b/w8a16.toml) | [text](text/32b/w4a16.toml) / [VL](vl/32b/w4a16.toml) |

runtime W8/W4 均量化 Text7 与 lm_head，**Vision 均为 W8**，embedding/norm/激活为 FP16。W4 配方：2/4/8B 为 AWQ Text7+RTN head，32B 为 RTN Text7+head；这些 runtime 配置为受控范围。

| 其他配方 | Text / VL | 区别 |
|---|---|---|
| 2B Text-W8 | [text](text/2b/w8a16_text.toml) / [VL](vl/2b/w8a16_text.toml) | 只量化七类decoder投影，head/Vision FP16 |
| 2B Text-AWQ | [text](text/2b/w4a16_text_awq.toml) / [VL](vl/2b/w4a16_text_awq.toml) | Text7 AWQ/G32，head/Vision FP16 |
| 4B Text-AWQ | [text](text/4b/w4a16_text_awq.toml) / [VL](vl/4b/w4a16_text_awq.toml) | 同上，独立量化资产与证据 |
| 32B legacy Text-W8 | [VL](vl/32b/w8a16_legacy.toml) | Text7 INT8，head/Vision FP16；物理prefill≤336、chunk≤64 |

`*_p192_d192`（text）、`*_p256_d192`（VL）及`*_p4096_d4096`为 FP16 固定负载复现；具体登记范围见 [模型矩阵](../../../docs/model_support.md)。

## 使用

先通过 `RPU_MODEL_CACHE` 准备对应registry权重；VL/vision填写真实图片路径，再修改选中的TOML：

```bash
python examples/run.py --config examples/configs/qwen3_vl/text/4b/w8a16.toml --check-config
python examples/run.py --config /path/to/my_qwen3vl.toml
```

runtime量化资产先转换；2/4/8B W4使用受支持AWQ源，32B W4使用对应官方Dense源。

```bash
python -m rpu_backend.quant.convert_qwen3_vl --bits 8 --src /path/to/Qwen3-VL-4B-Instruct --dst "$RPU_MODEL_CACHE/Qwen3-VL-4B-Instruct-runtime-w8"
python -m rpu_backend.quant.convert_qwen3_vl --bits 4 --src /path/to/supported-Qwen3-VL-4B-AWQ --dst "$RPU_MODEL_CACHE/Qwen3-VL-4B-Instruct-runtime-w4-awq"
```

### 可修改参数

| 字段 | 作用 / 范围 |
|---|---|
| `input.prompt/images` | 文字与图片；VL的实际P包含图像及chat模板 |
| `input.prefill_tokens/decode_steps` | 目标P与decode上限；输出含prefill首token，实际输出≤decode上限+1 |
| `rpu_execution.prefill/vision.linear_acc32` | false为ACC16，true为ACC32；prefill同时决定text decode累加 |
| `rpu_execution.prefill.fast_replay` | 模板显式true；API省略为false，无vision.fast_replay |
| `rpu_execution.model.num_cores` | 依精确profile，runtime量化TP8 |
| `cpu.threads`、`run.warmup/runs/inference_mode` | CPU 线程与执行模式 |

长度、ACC直接修改同一配置文件。冷配置变化后用新进程；入口核对模型尺寸与量化配方，`--check-config`只检查配置。普通runtime模板W2/R5；2B FP16 text基础模板默认正式运行3轮。采用 W2/R5 采样时设置 `run.warmup=2`、`run.runs=5`，输入、CPU线程、核数与ACC按对应模板填写。固定长测模板采用 W1/预热D3/R1/no_grad。

### 重放保存输入

VL可以重放带完整SHA256的预处理safetensors。替换整个 `[input]`，不保留 `prompt/images/prefill_tokens` 等其他输入字段：

```toml
[input]
component = "multimodal"
prepared_input_safetensors = "/path/to/input.safetensors"
prepared_input_sha256 = "填写该文件的64位SHA256"
decode_steps = 31
stop_on_eos = false
```

文件含INT64的 `input_ids/attention_mask/image_grid_thw/position_ids` 和FP32的 `pixel_values`，仅支持B1、单图单帧、全1 mask。入口在模型安装前校验文件、五张量和正常M-RoPE位置；不添加chat或补齐，正常decode delta保持。相对路径按配置文件目录解析，输出的 `generation.prepared_input` 记录全部哈希。

## 输入与边界

VL超出P时失败，不能截断图像或assistant结构。2/4/8B text现有prefill执行行上限4176，chunk上限320/256/128。P4096/D4096模板申请8192位置cache，实际运行仍受native资源准入限制。

32B legacy与runtime W8的资产格式不同；Dense-source 8B RTN W4不在支持范围。更多说明：[量化合同](../../../docs/quantization.md)、[main / DDR两份REF](../../../docs/runtime_assets.md)。
