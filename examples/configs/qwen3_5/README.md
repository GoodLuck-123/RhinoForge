# Qwen3.5 配置
[全部配置](../CONFIGURATIONS.md)

选择 `text/` 纯文本或 `vl/` 图文生成，再选模型与精度。Dense、MoE、官方 27B 与 legacy 27B 使用各自的权重和输入范围。

<a id="weights"></a>
<a id="weight-sources"></a>
## 权重来源与存储类型

官方原父的 **BF16 主体 + 少量 FP32 参数**已通过官方 config 与 HF safetensors 元数据核对（2026-10-10），不是纯 FP16 包；`mamba_ssm_dtype=float32` 也不是整个 checkpoint 的存储精度。下列目录相对于 `$RPU_MODEL_CACHE`，必须包括完整 config、tokenizer、processor 和全部权重分片。

VL 还需 `preprocessor_config.json` 和 `video_preprocessor_config.json`，与权重从同一 revision 下载。量化目录也需保留这两个文件；缺失时从对应浮点父的固定 revision 补齐。

| 权重 / 用途 | 磁盘存储精度 | 发布者与下载来源 | 下载到的目录 |
|---|---|---|---|
| Dense 0.8B：浮点父 | BF16 + 少量 FP32 | Qwen 官方 [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B) | `qwen3_5-0.8b` |
| Dense 2B：浮点父 | BF16 + 少量 FP32 | Qwen 官方 [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3.5-2B) | `Qwen3.5-2B` |
| Dense 4B：浮点父 | BF16 + 少量 FP32 | Qwen 官方 [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3.5-4B) | `Qwen3.5-4B` |
| Dense 9B：浮点父 | BF16 + 少量 FP32 | Qwen 官方 [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3.5-9B) | `Qwen3.5-9B` |
| Dense 27B：浮点父 | BF16 + 少量 FP32 | Qwen 官方 [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3.5-27B) | `Qwen3.5-27B` |
| Dense 本仓库 W8/W4 产物 | 文本投影 INT8 容器（值为8bit或4bit）+ FP16 scale；其余 FP16 | 本仓库从对应官方父转换，无可直接替代的同名官方量化下载包 | `qwen3_5_quant_ckpts/qwen3_5-{SIZE}b-w8a16` 或 `-w4a16` |
| MoE 35B-A3B：RTN 转换父 | BF16 + 少量 FP32 | Qwen 官方 [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3.5-35B-A3B) | `Qwen3.5-35B-A3B` |
| MoE RTN W4/G128：本仓库产物 | 路由专家 INT4（I32 packed），FP16 scale；非路由参数保留原 BF16/FP32 | 本仓库 routed-only RTN 转换 | `Qwen3.5-35B-A3B-RTN-W4A16-G128` |
| MoE GPTQ W4/G128：公开量化包 | 路由专家 INT4（I32 packed）；未量化参数/scale 含 BF16、FP32 | **Qwen 官方量化** [Qwen/Qwen3.5-35B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4) · [量化声明](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4/blob/main/config.json) | `Qwen3.5-35B-A3B-GPTQ-Int4` |
| MoE mixed E4M3：历史已供派生包 | 必需 E4M3 原始 UINT8 + FP16 scale；其余磁盘 dtype 未公开核实，加载为 FP16 | 完整生产器和原浮点父亲缘未公开证明；没有已核实的 HF/GitHub 下载地址，须原同包资产 | `Qwen3.5-35B-A3B-rpu-fp8-mixed` |
| legacy27：历史已供父 / W8 包 | 原父记录为 BF16；产物 W8 主干 + FP16 GDN/其余参数 | 已供父→W8 亲缘已证明；公开 HF 来源及与官方27B的关系未核实 | 父目录由交付提供；产物 `Qwen3.8-27B-w8a16` |

官方 27B 固定父 revision 为 `fc05daec18b0a78c049392ed2e771dde82bdf654`。27B父用于量化/CPU参考，不开放FP16模板；MoE官方浮点父也不是本目录可直接运行的配置。固定使用所选 revision 和配套元数据。

<a id="weight-recipes"></a>
## 模块精度与配方

Dense“**全部文本投影**”指：每层 MLP `gate_proj/up_proj/down_proj`；full-attention 层的 `q_proj/k_proj/v_proj/o_proj`；GDN 层的 `in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj`。下表 A16 为 FP16 主路径激活；ACC16/ACC32 是累加选择，不是另一种权重格式。长度、核数与 ACC 文件变体沿用同一行的配方和资产。

| config 配方 / 范围 | 运行时模块权重 / 激活 | 输入来源 | 获取方式、源码与完整命令 |
|---|---|---|---|
| Dense 0.8/2/4/9B `text/`、`vl/` 的 `fp16*.toml` | 文本投影、head、embedding、norm、conv、Vision FP16；A16 | 对应官方 BF16/FP32 父 | 不量化；[examples/runners/qwen3_5/runner.py](../../runners/qwen3_5/runner.py) 加载转 FP16；[完整下载/运行](#cmd-dense-fp16) |
| Dense 0.8/2/4/9B `w8a16*.toml`（Text/VL） | 全部文本投影 **W8**（逐输出通道）；head/embedding/norm/conv/**Vision FP16**；A16 | 对应官方父 | 离线 [python/rpu_backend/quant/convert_qwen3_5.py](../../../python/rpu_backend/quant/convert_qwen3_5.py) `--method w8a16`；[完整命令](#普通dense-w808249b) |
| Dense 0.8/2/4/9B `w4a16*.toml`（9B仅Text，其余Text/VL） | 全部文本投影 **W4/G32**；head/embedding/norm/conv/**Vision FP16**；A16 | 对应官方父 | 同脚本 `--method w4a16_pgrp --group-size 32`；[完整命令](#普通dense-w4g3208249b) |
| 官方27B `w8a16*.toml`（TP4/6/8 Text/VL） | 全部496个文本投影 W8；head/embedding/norm/conv/**Vision FP16**；A16 | 固定官方27B父 | 同一 Dense 离线脚本；[完整 W8 命令](#cmd-official27-w8) |
| 官方27B `w4a16*.toml`（TP8 Text/VL） | 全部496个文本投影 W4/G32；上述其余模块 FP16；A16 | 固定官方27B父 | 同一 Dense 离线脚本；[完整 W4 命令](#cmd-official27-w4) |
| MoE 35B `rtn_w4a16*.toml`（Text/VL） | **仅 routed experts gate/up/down W4/G128**；Attention/GDN、router、shared expert、head、embedding、norm、conv、Vision 均 FP16；A16 | 官方35B-A3B BF16/FP32父 | 离线 [python/rpu_backend/quant/convert_qwen3_5_moe_rtn.py](../../../python/rpu_backend/quant/convert_qwen3_5_moe_rtn.py)；[完整 RTN 命令](#cmd-moe-rtn) |
| MoE 35B `gptq_w4a16*.toml`（Text/VL） | 同样仅 routed experts W4/G128；其余 FP16；A16；**算法及零点编码与 RTN 分开** | 官方 GPTQ 完整包 | 直接下载；[python/rpu_backend/adapters/qwen3_5_moe/gptq_checkpoint.py](../../../python/rpu_backend/adapters/qwen3_5_moe/gptq_checkpoint.py) 读取并按运行布局安装；[下载/运行](#cmd-moe-gptq) |
| MoE 35B `mixed_e4m3*.toml`（Text/VL、TP4/6/8） | routed experts + full-attention Q/K/V/O + GDN QKV/Z/out 为 **E4M3**；GDN a/b/conv、router、shared expert、head、embedding、norm、Vision FP16；A16 | 原 mixed 同包资产 | 直接使用原包；[python/rpu_backend/adapters/qwen3_5_moe/loader.py](../../../python/rpu_backend/adapters/qwen3_5_moe/loader.py)；**无公开完整转换 CLI**；[实际运行命令](#cmd-moe-mixed) |
| `legacy/27b/w8a16*.toml`（仅Text） | MLP + full-attention W8；**所有 GDN 投影 FP16**；head/embedding/norm/conv FP16；A16 | 原 exact legacy BF16父或历史派生包 | Dense 脚本的独立 `--profile w8_main_fp16_gdn_legacy`；[完整命令](#历史legacy27-w8只能使用原exact-legacy浮点父) |

Dense 转换器同时排除 Vision、MTP、embedding、lm_head、norm、GDN `A_log/dt_bias` 与 conv1d 的量化；未量化浮点文件按既有 BF16→FP16 路径保存。W4值在文件中仍占 INT8 容器，加载时才打包，不能据 W4 名称预期磁盘大小减半。MoE RTN 转换器相反保留非路由原张量 dtype/值（包括 MTP），由 loader 在运行时转 FP16；它不量化 shared expert，也不做校准或 GPTQ 误差反馈。

<a id="text-w4-recipes"></a>
### 纯文本 W4 配方

新增文本模板包括 2B MLP-only G32，以及 2B、4B、9B 的 QuantTrio AWQ G128。MLP-only 使用官方浮点父和 `w4_mlp_only` 转换配置；AWQ 包直接下载，loader 保留其量化元数据，不做再次量化。四个 alias 已登记在 `profiles.json` 与模型注册表中，使用对应 `text/*/*.toml` 模板运行。

<a id="cmd-mlp-only-g32"></a>
#### MLP-only G32

```bash
SRC="$RPU_MODEL_CACHE/Qwen3.5-2B"
DST="$RPU_MODEL_CACHE/qwen3_5-2b-w4a16-g32-mlp-only"
hf download Qwen/Qwen3.5-2B --local-dir "$SRC"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3_5.py \
  --src "$SRC" --dst "$DST" --profile w4_mlp_only
python examples/run.py --config examples/configs/qwen3_5/text/2b/w4_mlp_only_g32.toml
```

<a id="cmd-quanttrio-awq"></a>
#### QuantTrio AWQ

```bash
SIZE=2  # 2、4 或 9
DST="$RPU_MODEL_CACHE/Qwen3.5-${SIZE}B-AWQ"
hf download "QuantTrio/Qwen3.5-${SIZE}B-AWQ" --local-dir "$DST"
python examples/run.py --config "examples/configs/qwen3_5/text/${SIZE}b/quanttrio_awq_g128.toml"
```


<a id="quantization"></a>
## 下载与转换

**离线转换可在 CPU 机器完成；各命令块最后一行 `python examples/run.py ...` 是模型推理，需在可用的 设备 板卡上执行。若当前是 CPU 转换机，请只执行下载和转换部分，再将产物按相同缓存路径放到板卡。**

以下 **Bash 命令在仓库根目录、已安装项目依赖的 Python 环境中执行**。先将缓存占位路径改为实际位置：

```bash
export RPU_MODEL_CACHE=/absolute/path/to/model_cache
```

`SRC` 是完整浮点父 checkpoint，`DST` 是尚不存在的量化输出目录。已下载时可修改 `SRC` 并跳过下载；Dense/legacy请保持下面 `DST`，对应模板的 `example.registry_alias` 会自动读取它，不能用 `--model` 改路径。MoE支持下文显式 `--model "$DST"`。每个块末尾给出转换后的实际运行命令；VL选择同大小同精度的 `vl/` 模板并填写 `input.images`，仍读取同一资产。

<a id="cmd-dense-fp16"></a>
### 普通 Dense FP16：下载官方浮点父后直接运行

```bash
SIZE=4  # 可选：0.8、2、4、9
SRC="$RPU_MODEL_CACHE/Qwen3.5-${SIZE}B"
if [ "$SIZE" = 0.8 ]; then SRC="$RPU_MODEL_CACHE/qwen3_5-0.8b"; fi
CONFIG_SIZE=${SIZE/./_}
hf download "Qwen/Qwen3.5-${SIZE}B" --local-dir "$SRC"
python examples/run.py --config "examples/configs/qwen3_5/text/${CONFIG_SIZE}b/fp16.toml"
```

VL使用同尺寸 `vl/${CONFIG_SIZE}b/fp16.toml`，先填写 `input.images`；不另存 FP16 checkpoint。

<a id="普通dense-w808249b"></a>
### 普通稠密模型 W8：0.8/2/4/9B

只改 `SIZE` 选择模型大小，其他路径随之匹配。此处W8为全部文本投影的统一量化，不选混合精度profile。

```bash
SIZE=9  # 可选：0.8、2、4、9；官方27B用下方独立命令
SRC="$RPU_MODEL_CACHE/Qwen3.5-${SIZE}B"
if [ "$SIZE" = 0.8 ]; then SRC="$RPU_MODEL_CACHE/qwen3_5-0.8b"; fi
DST="$RPU_MODEL_CACHE/qwen3_5_quant_ckpts/qwen3_5-${SIZE}b-w8a16"
CONFIG_SIZE=${SIZE/./_}
hf download "Qwen/Qwen3.5-${SIZE}B" --local-dir "$SRC"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3_5.py \
  --src "$SRC" --dst "$DST" --method w8a16
python examples/run.py --config "examples/configs/qwen3_5/text/${CONFIG_SIZE}b/w8a16.toml"
```

例如9B输出供 [text/9b/w8a16.toml](text/9b/w8a16.toml) / [vl/9b/w8a16.toml](vl/9b/w8a16.toml) 使用。

<a id="普通dense-w4g3208249b"></a>
### 普通稠密模型 W4/G32：0.8/2/4/9B

```bash
SIZE=9  # 可选：0.8、2、4、9
SRC="$RPU_MODEL_CACHE/Qwen3.5-${SIZE}B"
if [ "$SIZE" = 0.8 ]; then SRC="$RPU_MODEL_CACHE/qwen3_5-0.8b"; fi
DST="$RPU_MODEL_CACHE/qwen3_5_quant_ckpts/qwen3_5-${SIZE}b-w4a16"
CONFIG_SIZE=${SIZE/./_}
hf download "Qwen/Qwen3.5-${SIZE}B" --local-dir "$SRC"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3_5.py \
  --src "$SRC" --dst "$DST" --method w4a16_pgrp --group-size 32
python examples/run.py --config "examples/configs/qwen3_5/text/${CONFIG_SIZE}b/w4a16.toml"
```

例如9B输出供 [text/9b/w4a16.toml](text/9b/w4a16.toml) 使用；9B无公开W4 VL模板。0.8/2/4B有对应W4 Text/VL模板。

<a id="cmd-official27-w8"></a>
### 官方27B W8：固定官方父 → W8

下列revision为已测官方父。不能用历史 `Qwen3.8-27B` 父替代。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3.5-27B"
DST="$RPU_MODEL_CACHE/qwen3_5_quant_ckpts/qwen3_5-27b-w8a16"
hf download Qwen/Qwen3.5-27B --revision fc05daec18b0a78c049392ed2e771dde82bdf654 \
  --local-dir "$SRC"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3_5.py \
  --src "$SRC" --dst "$DST" --method w8a16
python examples/run.py --config examples/configs/qwen3_5/text/27b/w8a16.toml
```

同一输出供27B W8的TP4/6/8 Text/VL模板使用，例如 [vl/27b/w8a16.toml](vl/27b/w8a16.toml)。

<a id="cmd-official27-w4"></a>
### 官方27B W4/G32：固定官方父 → W4

```bash
SRC="$RPU_MODEL_CACHE/Qwen3.5-27B"
DST="$RPU_MODEL_CACHE/qwen3_5_quant_ckpts/qwen3_5-27b-w4a16"
hf download Qwen/Qwen3.5-27B --revision fc05daec18b0a78c049392ed2e771dde82bdf654 \
  --local-dir "$SRC"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3_5.py \
  --src "$SRC" --dst "$DST" --method w4a16_pgrp --group-size 32
python examples/run.py --config examples/configs/qwen3_5/text/27b/w4a16.toml
```

输出供27B TP8 [text/27b/w4a16.toml](text/27b/w4a16.toml) / [vl/27b/w4a16.toml](vl/27b/w4a16.toml) 使用。上述Dense离线入口源码为 [python/rpu_backend/quant/convert_qwen3_5.py](../../../python/rpu_backend/quant/convert_qwen3_5.py)。Vision、embedding、head、norm、conv等未量化部分保留FP16；转换能力不扩展模板准入。

<a id="cmd-moe-rtn"></a>
### 35B-A3B MoE RTN W4/G128：独立脚本转换

RTN输入为官方浮点父；不能用Dense脚本代替。可把 `DST` 改成其他新目录，运行时 `--model "$DST"` 明确选择该产物。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3.5-35B-A3B"
DST="$RPU_MODEL_CACHE/Qwen3.5-35B-A3B-RTN-W4A16-G128"
hf download Qwen/Qwen3.5-35B-A3B --local-dir "$SRC"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3_5_moe_rtn.py \
  --src "$SRC" --dst "$DST" --threads 8
python examples/run.py --config examples/configs/qwen3_5/text/35b_a3b/rtn_w4a16.toml --model "$DST"
```

入口：[python/rpu_backend/quant/convert_qwen3_5_moe_rtn.py](../../../python/rpu_backend/quant/convert_qwen3_5_moe_rtn.py)。同一输出也供 [vl/35b_a3b/rtn_w4a16.toml](vl/35b_a3b/rtn_w4a16.toml) 使用。

<a id="cmd-moe-gptq"></a>
### 35B-A3B MoE GPTQ：直接下载，无需转换

```bash
DST="$RPU_MODEL_CACHE/Qwen3.5-35B-A3B-GPTQ-Int4"
hf download Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --local-dir "$DST"
python examples/run.py --config examples/configs/qwen3_5/text/35b_a3b/gptq_w4a16.toml --model "$DST"
```

同一包也供 [vl/35b_a3b/gptq_w4a16.toml](vl/35b_a3b/gptq_w4a16.toml) 使用。MoE mixed E4M3须取得原同包资产，目前没有可替代原包的公开转换命令，详见[权重来源](#weights)。

<a id="cmd-moe-mixed"></a>
### MoE mixed E4M3：仅使用原已供整包

该配方使用既有整包。将 `SRC` 换成已取得的原 mixed E4M3 checkpoint，或先将其放到登记缓存目录 `$RPU_MODEL_CACHE/Qwen3.5-35B-A3B-rpu-fp8-mixed`；不能用官方 FP8/GPTQ 包直接替换。

```bash
SRC=/absolute/path/to/original-Qwen3.5-35B-A3B-rpu-fp8-mixed
python examples/run.py --config examples/configs/qwen3_5/text/35b_a3b/mixed_e4m3.toml --model "$SRC"
```

VL改用 `vl/35b_a3b/mixed_e4m3.toml` 并填写图像路径，同样传 `--model "$SRC"`。

<a id="历史legacy27-w8只能使用原exact-legacy浮点父"></a>
### 历史 legacy27 W8：只能使用原精确匹配的旧版浮点父

已供BF16父→历史W8的亲缘已证明，公开HF来源及与官方27B的关系未核实。将 `SRC` 的占位路径改为原exact legacy BF16父目录；`DST` 必须尚不存在，不覆盖历史包。

```bash
SRC=/path/to/original-exact-legacy27-bf16
DST="$RPU_MODEL_CACHE/Qwen3.8-27B-w8a16"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3_5.py \
  --src "$SRC" --dst "$DST" --profile w8_main_fp16_gdn_legacy
python examples/run.py --config examples/configs/qwen3_5/legacy/27b/w8a16.toml
```

`--profile` 自动生成W8主干/FP16 GDN声明；源配置不得预先带 `quant_config`。不能输入官方27B父，新输出不声称与历史包字节/SHA相同。[来源及精确配置要求](#历史legacy27-w8只能使用原exact-legacy浮点父)见后文。

0.8/2/4/9B FP16模板无需量化，直接读取[权重来源](#weights)的原浮点目录；官方27B没有公开FP16运行模板。

## 选择配置

| 模型 | Text | VL |
|---|---|---|
| 0.8B | [FP16](text/0_8b/fp16.toml) / [W8](text/0_8b/w8a16.toml) / [W4](text/0_8b/w4a16.toml) | [FP16](vl/0_8b/fp16.toml) / [W8](vl/0_8b/w8a16.toml) / [W4](vl/0_8b/w4a16.toml) |
| 2B | [FP16](text/2b/fp16.toml) / [W8](text/2b/w8a16.toml) / [W4](text/2b/w4a16.toml) | [FP16](vl/2b/fp16.toml) / [W8](vl/2b/w8a16.toml) / [W4](vl/2b/w4a16.toml) |
| 4B | [FP16](text/4b/fp16.toml) / [W8](text/4b/w8a16.toml) / [W4](text/4b/w4a16.toml) | [FP16](vl/4b/fp16.toml) / [W8](vl/4b/w8a16.toml) / [W4](vl/4b/w4a16.toml) |
| 9B | [FP16](text/9b/fp16.toml) / [W8](text/9b/w8a16.toml) / [W4](text/9b/w4a16.toml) | [FP16](vl/9b/fp16.toml) / [W8](vl/9b/w8a16.toml) |
| 官方 27B | [W8](text/27b/w8a16.toml) / [W8 TP4](text/27b/w8a16_tp4.toml) / [W8 TP6](text/27b/w8a16_tp6.toml) / [W4](text/27b/w4a16.toml) | [W8](vl/27b/w8a16.toml) / [W8 TP4](vl/27b/w8a16_tp4.toml) / [W8 TP6](vl/27b/w8a16_tp6.toml) / [W4](vl/27b/w4a16.toml) |
| 35B-A3B MoE | [mixed E4M3](text/35b_a3b/mixed_e4m3.toml) / [RTN W4](text/35b_a3b/rtn_w4a16.toml) / [GPTQ W4](text/35b_a3b/gptq_w4a16.toml) | [mixed E4M3](vl/35b_a3b/mixed_e4m3.toml) / [RTN W4](vl/35b_a3b/rtn_w4a16.toml) / [GPTQ W4](vl/35b_a3b/gptq_w4a16.toml) |

Dense W8/W4 量化文本主干，VL 视觉部分保留 FP16；量化 VL 用于受控评估。MoE 是集成入口，mixed、RTN、GPTQ 权重互不通用。准入与 opt-in 见[模型矩阵](../../../docs/model_support.md)。

| 文件变体 | 区别 |
|---|---|
| `fp16/w8a16/w4a16.toml` | 基础提示词与生成，选择匹配权重 |
| `*_tp4/6.toml` | 精确核数 profile，不可套用任意模型 |
| `*_p64_d4096`、`*_p4096_d4096` | 固定 P/D 固定负载运行，EOS 不提前停止 |
| Dense VL `fp16_p512_d192` | 固定历史图像与长度，独立输入条件 |
| Dense text `fp16_p4096_d4`、2B `*_c640` | 长 prefill / 短 decode；C640 显式 chunk profile |
| [legacy/27b](legacy/27b/README.md) | 原 Qwen3.8-27B 已供权重，W8 核数/长度变体 |

## 官方 Qwen3.5-27B

W8 支持 Text/VL 的 4、6、8 核配置；W4 使用 group32、8 核。TP6 中 MLP 使用 6 核，Attention、GDN、lm_head 和 Vision 使用 4 核。FP16 运行配置暂不提供。

官方 27B 量化全部 496 个文本投影，Vision、embedding、lm_head、norm 和 conv 保留 FP16。安装时先选择共享缓存分配器和四个 Graph arena，再将 CPU 权重逐层转换并上传；prefill 使用有界单次 Graph，decode 独立保留 Graph。

Text 需要 `QWEN3_5_QUANT_ALLOW_UNCERTIFIED=1`；VL 另需 `QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1`。模板已声明对应 opt-in。加载器检查官方 27B 身份和精度，拒绝 legacy、其他尺寸及混合配方。

`*_p64_d4096`、`*_p4096_d4096` 等模板设置固定 P/D；`*_acc32` 分别选择文本、Vision 或两者的 ACC32。基础配置使用 CPU 8 线程，长 Text 使用 24 线程，长 VL 使用 12 线程；预热/运行默认为 2/5。`*_mascot.toml` 和 VL `*_tp4/6.toml` 需填写自备图片路径。修改冷配置后启动新进程；视频和 W4 降核不支持。

## 使用

在 `RPU_MODEL_CACHE` 准备对应权重；复制需要的 TOML，修改输入和输出目录。VL 的 `input.images` 必须指向真实图片。

```bash
python examples/run.py --config examples/configs/qwen3_5/text/0_8b/fp16.toml --check-config
python examples/run.py --config /path/to/my_qwen35.toml
```

MoE 可用 `--model /path/to/checkpoint` 指定精度匹配的 checkpoint。相对路径按启动工作目录解析；`--check-config` 只检查配置。真实运行结果写入配置的输出目录。

基础 Dense/MoE 模板默认运行 3 轮；官方 27B 模板预热 2 轮、运行 5 轮。

## 可修改参数

| 字段 | 用途 |
|---|---|
| `input.prompt/images` | 提示词与图片，修改后形成新的测试输入 |
| `input.prefill_tokens/decode_steps` | 固定长度与 decode 上限，固定负载运行保留原 P/D |
| `rpu_execution.model.num_cores` | 已登记的 TP4/6/8 profile |
| `rpu_execution.prefill/vision.linear_acc32` | false=ACC16，true=ACC32；依各profile准入，MoE W4文本Prefill仅支持false，Vision独立 |
| `rpu_execution.prefill.chunk_size` | auto 或模板的显式 chunk |
| `cpu.threads`、`run.warmup/runs/output_dir` | CPU 线程、预热、正式轮次与输出目录 |

<a id="输入预热与计时"></a>
固定 D 产生 D+1 个输出 token，逻辑 KV=P+D；普通配置保留 EOS 条件，实际次数读 `generation_runs`。VL 的 P 含视觉及模板 token，不能任意截断到长度桶。MoE 正文模板读取 [prefill_realtext.txt](text/35b_a3b/prefill_realtext.txt)，正文不足目标长度时失败。冷配置变化使用新进程。算子库使用[正式 main/DDR 两份 REF](../../../docs/runtime_assets.md)。
