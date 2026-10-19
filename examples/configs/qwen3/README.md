# Qwen3 配置

[配置索引](../CONFIGURATIONS.md) · [配置与权重](#weight-recipes) · [运行命令](#quantization)

本目录运行纯文本 Qwen3。先选模型大小，再选精度；首次运行可用 [0.6B FP16](text/0_6b/fp16.toml)。图文模型见 [Qwen3-VL](../qwen3_vl/README.md)，混合注意力模型见 [Qwen3.5](../qwen3_5/README.md)。

## 配置说明

<a id="选择配置"></a>

<a id="weight-recipes"></a>

<a id="配置与权重对应"></a>

<a id="基准与配置增量"></a>

### 配置、精度与权重

0.6/1.7/4/8B 以同尺寸的 `fp16.toml` 为基准：官方 BF16 权重加载为 FP16，文本投影、词嵌入、归一化和输出头均使用浮点权重。14B/32B 没有 FP16 运行模板，使用同尺寸 W8 作为比较起点，浮点父权重用于转换和参考计算。

**Text7** 指每层 Attention 的 `q_proj/k_proj/v_proj/o_proj` 和 MLP 的 `gate_proj/up_proj/down_proj`。下表只列相对 FP16 改变的模块；embedding、norm 和激活始终为 FP16。W8 是逐输出通道对称 INT8；W4 是沿输入维度每 32 个值一组的对称 INT4（G32）。A16 指激活精度，ACC 指独立的累加精度。

| 配方与配置 | 相对基准的改动 | 权重获取步骤 |
|---|---|---|
| FP16：[0.6B](text/0_6b/fp16.toml) / [1.7B](text/1_7b/fp16.toml) / [4B](text/4b/fp16.toml) / [8B](text/8b/fp16.toml) | 基准；直接加载对应官方浮点包 | [下载官方权重](#cmd-fp16) |
| Instruct FP16：[0.6B](text/0_6b/fp16_instruct.toml) / [4B](text/4b/fp16_instruct.toml) | 只更换 `registry_alias/profile_id` 和 Instruct 权重，计算精度不变 | [4B 官方下载、0.6B 交付包](#cmd-instruct) |
| Text7 + head W8：[0.6B](text/0_6b/w8a16_text_head.toml) / [1.7B](text/1_7b/w8a16_text_head.toml) / [4B](text/4b/w8a16_text_head.toml) / [8B](text/8b/w8a16_text_head.toml) / [14B](text/14b/w8a16_text_head.toml) / [32B](text/32b/w8a16_text_head.toml) | Text7 **和输出头**改为 W8；`dtype/registry_alias/profile_id` 一起切换到 W8 配方 | [从官方浮点包离线转换](#cmd-w8) |
| Text7 W4、head FP16：[0.6B](text/0_6b/w4a16_text_rtn_g32.toml) / [1.7B](text/1_7b/w4a16_text_rtn_g32.toml) / [4B](text/4b/w4a16_text_rtn_g32.toml) / [8B](text/8b/w4a16_text_rtn_g32.toml) / [14B](text/14b/w4a16_text_rtn_g32.toml) / [32B](text/32b/w4a16_text_rtn_g32.toml) | Text7 做 RTN W4/G32，输出头保持 FP16；`dtype="w4a16"`，使用 W4 profile，仍读取浮点 alias | [0.6–14B 加载时转换](#w4g32加载时转换head-保留-fp16) / [32B 逐层转换](#cmd-32b-w4) |
| Text7 + head W4：[0.6B](text/0_6b/w4a16_text_head_rtn_g32.toml) / [1.7B](text/1_7b/w4a16_text_head_rtn_g32.toml) / [4B](text/4b/w4a16_text_head_rtn_g32.toml) / [8B](text/8b/w4a16_text_head_rtn_g32.toml) / [14B](text/14b/w4a16_text_head_rtn_g32.toml) / [32B](text/32b/w4a16_text_head_rtn_g32.toml) | 相对上一行再量化输出头；只切换带 `-lm-head` 的 `profile_id`，沿用同一浮点 alias | [0.6–14B 加载时转换](#w4g32加载时转换-text7--head) / [32B 逐层转换](#cmd-32b-w4) |

这些是从相应浮点源分别准备的配方。W8 生成独立 checkpoint；两种 W4 都直接读取浮点目录，由 `profile_id` 决定安装时是否量化输出头，不生成 W4 目录。RTN 指直接舍入量化；设备上保存 packed INT4 与 FP16 scale。源 embedding 与 head 共享权重时，W8 转换会建立独立 INT8 head，embedding 仍为浮点。

### 输入与执行变体

基准模板运行分块 prefill 和带 KV cache 的逐 token decode，使用自动 chunk、默认 TP8/ACC16、自然提示词长度、D4、EOS 停止、W2/R3。P 是输入 token 数，D 是 prefill 后的额外 decode 次数；prefill 已产生首个输出，因此固定 D 时共输出 D+1 个 token。W/R 是预热和正式测量次数。

| 配置组 | 相对 FP16 基础模板的输入、执行变化 |
|---|---|
| 0.6–14B W8 | 保留自然长度、D4、W2/R3 |
| 所有 W4、32B W8 | P128/D7、`stop_on_eos=false`、W2/R5；显式 TP8、ACC16 |
| Instruct FP16 | 沿用基础输入和执行设置，换用对应 Instruct tokenizer 与权重 |

<a id="输入与执行参数"></a>

## 准备与运行

<a id="准备权重并运行"></a>

<a id="quantization"></a>

下载和离线转换可在 CPU 机器完成，`examples/run.py` 在设备板卡执行。命令从仓库根目录运行，使用已安装项目依赖的 Python 环境。

先设置缓存根目录：

```bash
export RPU_MODEL_CACHE=/absolute/path/to/model_cache
```

离线 W8 转换时，`SRC` 填完整浮点 checkpoint 的实际路径，`DST` 按下面命令写入尚不存在的登记目录。运行时只按 `example.registry_alias` 查找缓存目录，不读取 Shell 的 `SRC/DST`；此族不支持 `--model`。

已有权重可跳过对应下载，但需绑定到下表的缓存目录。例如，已有 4B 原始权重时，在目标路径尚不存在的情况下建立链接：

```bash
mkdir -p "$RPU_MODEL_CACHE"
ln -s /absolute/path/to/existing-Qwen3-4B "$RPU_MODEL_CACHE/Qwen3-4B"
```

4B FP16 和两种加载时 W4 配方直接读取这个目录。其他尺寸、Instruct 或已有 W8 产物也绑定到各自在下表中的目录；离线转换的 `SRC` 可另选源路径，`DST` 仍须匹配 W8 登记目录。

<a id="weights"></a>

<a id="weight-sources"></a>

### 权重来源

官方权重以 BF16 保存，FP16 指加载后的计算精度。目录均相对于 `$RPU_MODEL_CACHE`；下载完整仓库，包含 config、tokenizer 和全部权重分片。

| 权重 | 磁盘存储精度 | 发布者与下载来源 | 缓存目录 |
|---|---|---|---|
| Qwen3-0.6B | BF16 | Qwen 官方 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-0.6B) | `Qwen3-0.6B` |
| Qwen3-1.7B | BF16 | Qwen 官方 [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-1.7B) | `Qwen3-1.7B` |
| Qwen3-4B | BF16 | Qwen 官方 [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-4B) | `Qwen3-4B` |
| Qwen3-8B | BF16 | Qwen 官方 [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-8B) | `Qwen3-8B` |
| Qwen3-14B | BF16 | Qwen 官方 [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-14B) | `Qwen3-14B` |
| Qwen3-32B | BF16 | Qwen 官方 [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-32B) | `Qwen3-32B` |
| 4B Instruct | BF16 | Qwen 官方 [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | `qwen3-4b-instruct` |
| 0.6B Instruct | 交付包存储精度未记录；公开包为 BF16 | 登记下载来源为第三方 [rd211/Qwen3-0.6B-Instruct](https://huggingface.co/rd211/Qwen3-0.6B-Instruct)；现有登记使用配套的 `instruct-local` 包 | `qwen3-0.6b-instruct`，使用交付包 |
| 本仓库 W8 产物 | Text7/head 为 INT8 + FP16 scale；其余 FP16 | 本仓库从相应浮点父转换；没有可替代此格式的官方量化下载地址 | 0.6–14B：`qwen3_lmhead_int8_ckpts/qwen3-{SIZE}b-w8a16-lmhead-int8`；32B：`Qwen3-32B-w8a16` |

14B/32B 只提供量化运行模板，官方 BF16 权重用于转换。

<a id="cmd-fp16"></a>

### 官方浮点源：FP16 运行与量化准备

首次 0.6B FP16 使用与[入门指南](../../../docs/getting_started.zh.md)相同的固定 revision：

```bash
SIZE=0.6
REVISION=c1899de289a04d12100db370d81485cdf75e47ca
hf download "Qwen/Qwen3-${SIZE}B" --revision "$REVISION" \
  --local-dir "$RPU_MODEL_CACHE/Qwen3-${SIZE}B"
```

其他尺寸可选 1.7、4、8、14、32；修改 `SIZE` 时，必须同时将 `REVISION` 改为该模型
上游仓库对应的完整 commit 并记录来源，不能跨尺寸复用 0.6B revision。
14/32B 浮点源仅用于量化和参考。下载完整 config/tokenizer/权重，不需要先将官方
BF16 包离线转换为 FP16。

首次运行按入门指南创建本地配置、设置仓库外的 `run.output_dir`、检查 SDK/设备权限
并取得平台锁，再执行 `--check-config` 和正式运行。原始 0.6B 模板为裸提示词 P7、
D4、W2/R3，最多输出 5 token，没有应用 chat template。结果的 `PASS` 只表示
执行完成，`correctness=not_run` / `formal_certification=false` 不代表参考验证或认证。

<a id="cmd-instruct"></a>

### Instruct FP16：使用独立权重

4B 使用已登记的官方 revision：

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --local-dir "$RPU_MODEL_CACHE/qwen3-4b-instruct"
```

0.6B 将交付的完整包放到 `$RPU_MODEL_CACHE/qwen3-0.6b-instruct`。

<a id="cmd-w8"></a>

### W8：离线转换 Text7 + 输出头

先按[官方权重步骤](#cmd-fp16)下载对应大小，再转换。只改 `SIZE`，输入和输出路径会一起匹配；`--quant-lm-head` 不可省略。

```bash
SIZE=4  # 可选：0.6、1.7、4、8、14、32
SRC="$RPU_MODEL_CACHE/Qwen3-${SIZE}B"
DST="$RPU_MODEL_CACHE/qwen3_lmhead_int8_ckpts/qwen3-${SIZE}b-w8a16-lmhead-int8"
if [ "$SIZE" = 32 ]; then DST="$RPU_MODEL_CACHE/Qwen3-32B-w8a16"; fi
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 python/rpu_backend/quant/convert_qwen3.py \
  --src "$SRC" --dst "$DST" --quant-lm-head
```

脚本：[python/rpu_backend/quant/convert_qwen3.py](../../../python/rpu_backend/quant/convert_qwen3.py)。以 `SIZE=4` 运行时，对应 [text/4b/w8a16_text_head.toml](text/4b/w8a16_text_head.toml)。

<a id="w4g32加载时转换head-保留-fp16"></a>

<a id="w4g32加载时转换-text7--head"></a>

<a id="cmd-32b-w4"></a>

### W4/G32：加载时量化，选择输出头精度

两种 W4 共用[已下载的官方浮点包](#cmd-fp16)，不生成离线 W4 checkpoint。配置名 `w4a16_text_rtn_g32.toml` 只量化 Text7、输出头保持 FP16；`w4a16_text_head_rtn_g32.toml` 同时量化输出头。在下方公共运行步骤选对应配置即可。

0.6–14B 由安装入口量化；32B 由同一公共入口逐层量化上传，不先加载完整 CPU FP16 模型。实现见[加载时量化](../../../python/rpu_backend/quant/convert_qwen3_w4a16.py)和[32B 分层加载](../../../python/rpu_backend/quant/load_qwen3_quantized.py)。

### 关键配置项

公共字段见[参数说明](../PARAMETERS.md)，输出字段见[日志与结果](../../RESULTS.md#日志设置)。本模型需要一起检查以下几组值：

| 层次 / 字段 | 模板值与省略后的有效值 | 范围与关联 |
|---|---|---|
| 配方：`example.dtype/registry_alias/profile_id` | 由上表所选模板固定，三项必填 | 一起选择精度、权重目录和执行入口；W4 是否量化 head 由 profile 决定 |
| 输入：`input.prompt/prefill_tokens` | 基础模板自然长度；W4、32B W8 为 P128；省略 P 时保留原 token 长度 | P 为正整数；指定 P 时重复或截取提示词 token。14B 量化 P≤192 |
| 生成：`input.decode_steps/stop_on_eos` | 基础 D4/`true`；固定负载 D7/`false`；省略值分别为 4/`true` | D≥0；固定输出长度时关闭 EOS。32B 量化缓存容量 P+D≤4096 |
| 拓扑：`rpu_execution.model.num_cores` | 基础省略，量化固定负载显式 8；省略为 8 | 已登记 FP16 B1 配方可用 4/6/8；量化模板 TP8；与 profile 配对 |
| 累加：`rpu_execution.prefill.linear_acc32` | 省略或 `false` 为 ACC16，`true` 为 ACC32 | 同时影响文本 prefill/decode Linear；不改变权重目录 |
| 分块：`rpu_execution.prefill.chunk_size` | 模板及省略值均为 `auto` | 可指定 16 的正整数倍；14B 量化≤32，32B≤64；还须符合当前模型布局 |
| 测量：`cpu.threads`、`run.*` | CPU 线程省略时沿用启动环境；模板 W2/R3 或 W2/R5；省略 W/R 为 2/1，`inference_mode` 为 `true` | `warmup≥0`、`runs≥1`；对比时统一线程、轮次和执行上下文 |

32B W4 由公共入口逐层量化上传，W8 从量化文件 mmap 加载。尺寸与执行范围见[模型矩阵](../../../docs/model_support.md)。改变精度、核数或 ACC 后用新进程运行。

<a id="使用"></a>

### 运行自己的输入

准备权重后复制 TOML，修改输入和 `run.output_dir`：

```bash
CONFIG=examples/configs/qwen3/text/4b/w4a16_text_head_rtn_g32.toml
python3 examples/run.py --config "$CONFIG" --check-config
python3 examples/run.py --config "$CONFIG"
```

`--check-config` 检查字段和登记关系；正式运行时读取权重，并写出 `report.json` 与生成结果。精度、核数、ACC 等安装参数改变后，用新进程运行。

转换实现：[W8](../../../python/rpu_backend/quant/convert_qwen3.py)、[W4 加载时量化](../../../python/rpu_backend/quant/convert_qwen3_w4a16.py)、[32B 分层加载](../../../python/rpu_backend/quant/load_qwen3_quantized.py)。运行库见[运行资产](../../../docs/runtime_assets.md)。
