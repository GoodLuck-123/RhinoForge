# Qwen3.5 配置

[全部配置](../CONFIGURATIONS.md) · [配置与权重](#weight-recipes) · [运行命令](#quantization)

先选 `text/` 纯文本或 `vl/` 图文生成，再选模型大小和精度。0.8/2/4/9B 有 FP16 基准；官方 27B 提供 W8/W4；35B-A3B MoE 提供 RTN、GPTQ 和 mixed E4M3 三种配方。配方决定权重内容，P/D、核数和 ACC 变体共用对应配方的权重。

## 配置说明

<a id="选择配置"></a>

<a id="weight-recipes"></a>

<a id="配置与权重对应"></a>

<a id="基准与配置增量"></a>

### Dense：以 FP16 为基准选配方

0.8/2/4/9B 以同尺寸、同输入类型的 `fp16.toml` 为基准。文本主干包含普通 Attention、Gated DeltaNet（GDN）和 MLP；图文输入先经过 Vision encoder。基准将权重加载为 FP16，prefill 自动分块，decode 使用 KV cache 和 GDN 状态缓存，每轮生成前重置缓存。

量化范围按模块读：**全部文本投影**包括各层 MLP 的 `gate_proj/up_proj/down_proj`、full-attention 层的 `q_proj/k_proj/v_proj/o_proj`、GDN 层的 `in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj`。以下 Dense 配方中，**输出头、embedding、norm、conv、Vision 均保持 FP16**，GDN 的 `A_log/dt_bias` 和 MTP 也不量化。A16 是主路径激活精度，ACC16/ACC32 是独立的累加选项。

| 配方与配置 | 相对 FP16 量化了什么 | 来源与权重获取步骤 |
|---|---|---|
| FP16：0.8B [Text](text/0_8b/fp16.toml) / [VL](vl/0_8b/fp16.toml)；2B [Text](text/2b/fp16.toml) / [VL](vl/2b/fp16.toml)；4B [Text](text/4b/fp16.toml) / [VL](vl/4b/fp16.toml)；9B [Text](text/9b/fp16.toml) / [VL](vl/9b/fp16.toml) | 基准，无量化 | [下载各尺寸官方 BF16/FP32 包](#cmd-dense-fp16)，加载为 FP16 |
| 统一 W8：0.8B [Text](text/0_8b/w8a16_text.toml) / [VL](vl/0_8b/w8a16_text.toml)；2B [Text](text/2b/w8a16_text.toml) / [VL](vl/2b/w8a16_text.toml)；4B [Text](text/4b/w8a16_text.toml) / [VL](vl/4b/w8a16_text.toml)；9B [Text](text/9b/w8a16_text.toml) / [VL](vl/9b/w8a16_text.toml) | 全部文本投影，逐输出通道对称 INT8 | [从对应官方浮点包离线转换](#普通dense-w808249b) |
| 统一 W4/G32：0.8B [Text](text/0_8b/w4a16_text_rtn_g32.toml) / [VL](vl/0_8b/w4a16_text_rtn_g32.toml)；2B [Text](text/2b/w4a16_text_rtn_g32.toml) / [VL](vl/2b/w4a16_text_rtn_g32.toml)；4B [Text](text/4b/w4a16_text_rtn_g32.toml) / [VL](vl/4b/w4a16_text_rtn_g32.toml)；9B [Text](text/9b/w4a16_text_rtn_g32.toml) | 与 W8 范围相同；RTN 对称 INT4，每 32 个输入值一组 | [从对应官方浮点包离线转换](#普通dense-w4g3208249b)；9B 无 W4 VL 模板 |
| 2B 仅 MLP W4/G32：[Text](text/2b/w4a16_mlp_rtn_g32.toml) | 全部 24 层的 72 个 MLP gate/up/down，包括第 0 层；114 个 Attention/GDN 投影保持 FP16 | [官方 2B 浮点包 → RTN MLP-only](#cmd-mlp-only-g32) |
| 2B MLP W4/G32 + Attention/GDN W8：[Text](text/2b/w4a16_mlp_rtn_g32_attn_gdn_w8a16.toml) | 同样72个MLP signed RTN W4/G32；24个Full Attention及90个GDN投影（含in_proj_b/a）改为对称逐输出通道W8；其余FP16 | [官方2B浮点包 → 混合W4/W8](#cmd-mlp-w4-attn-gdn-w8) |
| 2B QuantTrio AWQ/G128：[Text](text/2b/w4a16_mlp_awq_g128.toml) | 第 1 层起的 69 个 MLP 投影为非对称 UINT4/G128；第 0 层 MLP、全部 Attention/GDN 保持 FP16 | [直接下载 QuantTrio AWQ 原包](#cmd-quanttrio-awq) |
| 4B QuantTrio AWQ/G128：[Text](text/4b/w4a16_mlp_awq_g128.toml) | 同上，量化 93 个 MLP 投影 | [直接下载 QuantTrio AWQ 原包](#cmd-quanttrio-awq) |
| 9B QuantTrio AWQ/G128：[Text](text/9b/w4a16_mlp_awq_g128.toml) | 同上，量化 93 个 MLP 投影 | [直接下载 QuantTrio AWQ 原包](#cmd-quanttrio-awq) |

统一 W4 与 MLP-only 均从官方浮点包独立转换：RTN 直接按分组 scale 舍入为低比特整数，不读取校准数据。[AWQ](https://arxiv.org/abs/2306.00978) 先观察激活数据，再调整通道缩放，减少重要权重的量化误差。这里直接使用 QuantTrio 已完成的量化结果，本仓库保留 UINT4、直接零点和 FP16 scale，仅调整加载布局。它与 MLP-only 同时在**第 0 层是否量化、分组大小、对称性和量化方法**上不同。

选择量化模板会同时改变 `example.dtype/registry_alias/profile_id`，Dense 量化入口还要求模板中的 `QWEN3_5_QUANT_ALLOW_UNCERTIFIED=1`。这些字段绑定权重格式和加载入口。上述各行是独立配方，比较时以 FP16 看改动；准备时分别从对应源包开始。

Dense 转换器的 W8/W4 方法和混合配方把量化输入及其余浮点参数从实际存储 dtype 直接转 FP16：FP16 保持原值，FP32 不经 BF16 中间舍入。旧产物保持不变；使用新转换规则时，从原始浮点权重生成到新目录。W4 整数在文件中仍用 INT8 容器，安装到设备时才打包，因此磁盘占用不能按 4 bit 直接估算。AWQ 的 FP32 GDN 参数直接转 FP16，不经过 BF16 中转。

<a id="text-w4-recipes"></a>

#### MLP-only / 混合 W4-W8 / AWQ 的长输入配置

以上五种配方都使用 B1、TP8 的纯文本入口；短模板与以下长输入模板共用权重。

| 配方 | P64/D4096 | P4096/D4096 |
|---|---|---|
| 2B MLP-only | [配置](text/2b/w4a16_mlp_rtn_g32_p64_d4096.toml) | [配置](text/2b/w4a16_mlp_rtn_g32_p4096_d4096.toml) |
| 2B MLP W4 + Attention/GDN W8 | [配置](text/2b/w4a16_mlp_rtn_g32_attn_gdn_w8a16_p64_d4096.toml) | [配置](text/2b/w4a16_mlp_rtn_g32_attn_gdn_w8a16_p4096_d4096.toml) |
| 2B QuantTrio AWQ | [配置](text/2b/w4a16_mlp_awq_g128_p64_d4096.toml) | [配置](text/2b/w4a16_mlp_awq_g128_p4096_d4096.toml) |
| 4B QuantTrio AWQ | [配置](text/4b/w4a16_mlp_awq_g128_p64_d4096.toml) | [配置](text/4b/w4a16_mlp_awq_g128_p4096_d4096.toml) |
| 9B QuantTrio AWQ | [配置](text/9b/w4a16_mlp_awq_g128_p64_d4096.toml) | [配置](text/9b/w4a16_mlp_awq_g128_p4096_d4096.toml) |

MLP-only 的 alias 为 `qwen3_5-2b-w4-mlp-only-g32`，profile 为 `qwen3_5-2b.w4-mlp-only-g32.text-controlled`；AWQ 对应 `qwen3_5-{2,4,9}b-quanttrio-awq-g128` 和 `qwen3_5-{2,4,9}b.quanttrio-awq-g128.text-controlled`。加载器按模型尺寸检查逐层范围和分组，格式定义见[量化范围实现](../../../python/rpu_backend/adapters/qwen3_5/quant_scope.py)。

MLP W4 + Attention/GDN W8 使用现有 mixed-v2 `quant_config`：W8 default 加三个 MLP W4/G32 role override。入口 `dtype="w4a16"` 是加载声明，实际逐投影精度由 checkpoint 决定；新 alias 的入口会校验 2B 几何、层类型和每个投影，拒绝 MLP-only、统一 W4/W8 或误标的 checkpoint。

`w4a16_mlp_rtn_g32_attn_gdn_w8a16` 文件名中的 `G32` 仅属于 MLP W4；Attention/GDN W8 每输出通道一个 FP16 scale，没有分组或 zero-point。它仍使用 `--profile w4_mlp_w8_attn_gdn`、alias `qwen3_5-2b-w4-mlp-w8-attn-gdn-g32` 和 profile ID `qwen3_5-2b.w4-mlp-w8-attn-gdn-g32.text-controlled`。

<a id="官方-qwen35-27b"></a>

### 官方 Qwen3.5-27B：以 W8 为比较起点

27B 没有 FP16 运行模板，浮点权重用于转换与 CPU 参考。

| 配方与配置 | 比较起点与改动 | 权重获取步骤 |
|---|---|---|
| W8：Text [TP8](text/27b/w8a16_text.toml) / [TP4](text/27b/w8a16_text_tp4.toml) / [TP6](text/27b/w8a16_text_tp6.toml)；VL [TP8](vl/27b/w8a16_text.toml) / [TP4 mascot/D31](vl/27b/w8a16_text_tp4_image_mascot_d31.toml) / [TP6 mascot/D31](vl/27b/w8a16_text_tp6_image_mascot_d31.toml) | 比较起点；全部 496 个文本投影 W8，其余模块 FP16 | [固定官方浮点父 → W8](#cmd-official27-w8) |
| W4/G32：[Text](text/27b/w4a16_text_rtn_g32.toml) / [VL](vl/27b/w4a16_text_rtn_g32.toml) | 与 W8 相同范围，改为 RTN W4/G32，换用 W4 alias/profile/目录；TP8 | [同一固定官方浮点父 → W4](#cmd-official27-w4) |
| [Qwen3.8-27B 配套配置](legacy/27b/README.md) | 独立模型配置：MLP + full-attention W8，所有 GDN 投影 FP16 | [配套浮点父或 W8 包](legacy/27b/README.md#weights) |

官方 27B 的 Text prefill 使用单次构图执行，decode 和 Vision 按各自 Graph 缓存重放。官方 27B 模板均为 W2/R5、`inference_mode=true`、`timing_mode="per_call"`。短模板 CPU8，长 Text CPU24，长 VL CPU12。TP6 的 MLP 用 6 核，Attention、GDN、输出头和 Vision 用 4 核。

<a id="35b-a3b-moe"></a>

### 35B-A3B MoE：区分专家量化与混合精度

MoE 由 router 选择路由专家参与计算，shared expert 是独立的共享分支。本目录没有 FP16 运行模板，以 RTN W4 为配方比较起点；三种配方的获取方式各自独立。

| 配方与配置 | 量化范围及相对 RTN 的区别 | 来源与权重获取步骤 |
|---|---|---|
| RTN W4/G128：[Text](text/35b_a3b/w4a16_experts_rtn_g128.toml) / [VL](vl/35b_a3b/w4a16_experts_rtn_g128.toml) | **仅路由专家** gate/up/down 为 W4/G128；Attention/GDN、router、shared expert、head、embedding、norm、conv、Vision 全部 FP16 | [官方 35B-A3B 浮点包 → 本仓库 RTN](#cmd-moe-rtn) |
| GPTQ W4/G128：[Text](text/35b_a3b/w4a16_experts_gptq_g128.toml) / [VL](vl/35b_a3b/w4a16_experts_gptq_g128.toml) | 量化范围相同，改用 Qwen 官方 GPTQ 结果；方法和零点编码不同，使用 GPTQ alias/profile | [直接下载 Qwen 官方 GPTQ](#cmd-moe-gptq) |
| mixed E4M3：[Text](text/35b_a3b/fp8_e4m3_experts_attn_gdn.toml) / [VL](vl/35b_a3b/fp8_e4m3_experts_attn_gdn.toml) | **路由专家 + full-attention Q/K/V/O + GDN QKV/Z/out** 为逐输出通道缩放 E4M3；GDN a/b/conv、router、shared expert、head、embedding、norm、Vision FP16；`dtype="mixed-e4m3"` | [使用配套 mixed E4M3 包](#cmd-moe-mixed) |

RTN 在本仓库直接舍入。[GPTQ](https://arxiv.org/abs/2210.17323) 利用校准数据，在逐步量化权重时补偿产生的误差。这里直接读取上游 GPTQ 成品包；[加载器](../../../python/rpu_backend/adapters/qwen3_5_moe/gptq_checkpoint.py)识别原包零点编码并按设备布局安装。mixed E4M3 同时改变专家精度与 Attention/GDN 的量化范围，使用配套派生包，格式不是官方 block-FP8。

RTN/GPTQ 使用 B1、TP8、文本 ACC16；短模板 CPU24、no_grad、W2/R3，Text D31、VL D4。mixed 有 TP4/6/8 模板；短模板省略 CPU 线程与 `inference_mode`，分别沿用环境和默认 `true`。RTN 转换保留非路由张量（包括 MTP）的原 dtype/值，加载时转为 FP16。

<a id="输入预热与计时"></a>

<a id="输入核数与-acc-变体"></a>

### 输入、核数与 ACC 变体

W2/R3 表示预热 2 次、正式测量 3 次，TP 表示配置的根核数。P 表示输入 token 数，D 表示 prefill 后的 decode 次数；prefill 产生首个输出，所以固定 D 时输出为 D+1，逻辑缓存长度为 P+D。VL 的 P 包含图像和 chat 模板 token。

| 文件或参数 | 相对相同配方基础模板的变化 |
|---|---|
| `text/*/fp16_tp4.toml`、`fp16_tp6.toml` | 更换核数 profile，设置 `num_cores=4/6`，同时把 D4 改为 D32 |
| 27B Text `w8a16_text_tp4.toml` / `w8a16_text_tp6.toml` | 更换核数 profile，同时把 D4 改为 D32；若只比较核数，使用下述 P29/D31 组 |
| 27B Text `w8a16_text*_p29_d31.toml` | 使用文件中的 29 个 token、D31、关闭 EOS；`_tp4` / `_tp6` 只改核数，`_acc32_text` 只改文本累加 |
| Dense Text `*_p4096_d4.toml` | 固定 P4096、保留 D4；2B 的 `_chunk640` 额外指定 `prefill.chunk_size=640` |
| Dense Text `*_p64_d4096`、`*_p4096_d4096` | 固定 token 前缀，以 `prefill_fill_token_id=248046` 补到 P，D4096、关闭 EOS、CPU24。0.8–9B 用 no_grad，正式 R1（0.8B W4 P64 为 R3）；27B 用 inference mode、R5 |
| MLP-only/混合 W4-W8/AWQ 的 `*_p64_d4096`、`*_p4096_d4096` | 使用完整固定 token 列表，P64 是 P4096 的前缀，不加填充；CPU4、TP8、ACC16、no_grad、W2/R5 |
| Dense VL `*_p512_d192`、`*_p4096_d4096` | 仓库 [`assets/test_frame.png`](../../../assets/test_frame.png)、`image_size=448`，按 `prefill_filler_prompt` 补文本，关闭 EOS、CPU12。0.8–9B FP16 使用 `legacy_dense_vl`/R3；27B W8 使用 `per_call`/R5 |
| 27B VL `w8a16_text_d4.toml` | 从 D32 改为 D4；省略 `stop_on_eos`，VL 入口仍按 `false` 执行 |
| 27B VL `w8a16_text_image_mascot_d31.toml`、`w8a16_text_tp4_image_mascot_d31.toml`、`w8a16_text_tp6_image_mascot_d31.toml` | 同一 mascot 图片和提示词、D31，分别使用 8/4/6 根核数；需填写配套 `r1_mascot.jpeg` 路径 |
| 27B Text 的 `*_acc32_text*.toml`、VL 的 `w4a16_text_rtn_g32_acc32_text_vision.toml` | Text 文件只开启 `prefill.linear_acc32`；VL W4 文件同时开启文本和 Vision ACC32 |
| 27B VL `w8a16_text_acc32_text.toml`、`w8a16_text_acc32_vision.toml` | 分别只开启文本或 Vision 的 `linear_acc32`；另一项保持 ACC16 |
| MoE `fp8_e4m3_experts_attn_gdn_tp4.toml` / `fp8_e4m3_experts_attn_gdn_tp6.toml` | 只切换核数及对应 profile，共用 mixed 权重 |
| MoE Text `*_p64_d4096`、`*_p4096_d4096` | 读取 [prefill_realtext.txt](text/35b_a3b/prefill_realtext.txt) 的正文 token，校验文件和 token 哈希；TP8、CPU24、ACC16、`padding_rows=0`、no_grad、W2/R3，计时为 `forward_argmax` |
| MoE VL `fp8_e4m3_experts_attn_gdn_p4096_d4096.toml` | 单独固定图文 P4096/D4096，关闭 EOS、no_grad、W2/R1 |
| [legacy/27b](legacy/27b/README.md) | 使用交付的 Qwen3.8-27B 配方，核数和长度在该页单列 |

Dense 与 MoE 的根核数为 6 时，MLP 使用 6 核，Attention/GDN、输出头和 Vision 使用 4 核；4 核与 8 核配置分别使用相应核数。

`linear_acc32` 省略或 `false` 为 ACC16，`true` 为 ACC32；文本项也决定 decode Linear 累加。MoE W4 文本路径仅接受 ACC16，Vision 单独配置。核数、ACC、chunk 与输入长度不要求重新转换权重；安装参数变化后启动新进程。

纯文本 `prefill_tokens` 根据输入模式重复、截取或补齐 token。VL 保留图像和对话结构，实际输入超过目标 P 时会报错；改变图片或提示词后应读取运行结果中的实际 P。

## 准备与运行

<a id="准备权重并运行"></a>

<a id="quantization"></a>

下载和离线转换可在 CPU 机器完成，`examples/run.py` 在设备板卡执行。以下命令从仓库根目录运行，使用已安装项目依赖的 Python 环境。

先设置缓存根目录：

```bash
export RPU_MODEL_CACHE=/absolute/path/to/model_cache
```

离线转换时，`SRC` 填完整浮点父 checkpoint 的实际路径，`DST` 为尚不存在的量化输出目录。Dense/legacy 运行时按 `example.registry_alias` 查找缓存目录，不读取 Shell 的 `SRC/DST`；转换输出须保持下面的登记路径，不能用 `--model` 覆盖。MoE 可用下文的 `--model /实际路径` 指定权重。

已有权重可跳过对应下载，但 Dense/legacy 仍需绑定到下表的缓存目录。例如，已有官方 2B 浮点权重时，在目标路径尚不存在的情况下建立链接：

```bash
mkdir -p "$RPU_MODEL_CACHE"
ln -s /absolute/path/to/existing-Qwen3.5-2B "$RPU_MODEL_CACHE/Qwen3.5-2B"
```

2B FP16 直接读取这个目录；直接加载 AWQ 或已有量化产物时，改用下表中该配方的缓存目录。离线转换的 `SRC` 可另选源路径。同尺寸、同配方的 Text/VL 共用权重，VL 另填 `input.images`。

<a id="weights"></a>

<a id="weight-sources"></a>

### 权重来源

官方权重以 BF16 为主，含少量 FP32 参数；加载后的 FP16 计算精度与磁盘精度分开看。`mamba_ssm_dtype=float32` 只描述相应参数。下列目录相对于 `$RPU_MODEL_CACHE`，必须包括完整 config、tokenizer、processor 和全部权重分片。

| 权重 | 磁盘存储精度 | 发布者与下载来源 | 缓存目录 |
|---|---|---|---|
| Dense 0.8B | BF16 + 少量 FP32 | Qwen 官方 [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B) | `qwen3_5-0.8b` |
| Dense 2B | BF16 + 少量 FP32 | Qwen 官方 [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3.5-2B) | `Qwen3.5-2B` |
| Dense 4B | BF16 + 少量 FP32 | Qwen 官方 [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3.5-4B) | `Qwen3.5-4B` |
| Dense 9B | BF16 + 少量 FP32 | Qwen 官方 [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3.5-9B) | `Qwen3.5-9B` |
| Dense 27B | BF16 + 少量 FP32 | Qwen 官方 [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3.5-27B) | `Qwen3.5-27B` |
| Dense 本仓库 W8/W4 产物 | 文本投影 INT8 容器（值为8bit或4bit）+ FP16 scale；其余 FP16 | 本仓库从对应官方父转换，无可直接替代的同名官方量化下载包 | `qwen3_5_quant_ckpts/qwen3_5-{SIZE}b-w8a16` 或 `-w4a16` |
| Dense 2B 仅 MLP RTN G32 | INT8 容器保存 signed INT4 值，scale 与其余参数为 FP16 | 本仓库从 Qwen 官方 2B 父离线转换 | `qwen3_5-2b-w4a16-g32-mlp-only` |
| Dense 2B MLP RTN G32 + Attention/GDN W8 | MLP的INT8容器保存signed INT4值；其余114个投影为INT8；scale和未量化参数FP16 | 本仓库从Qwen官方2B父离线转换，源dtype直接转FP16 | `qwen3_5-2b-w4a16-g32-mlp-w8-attn-gdn` |
| Dense 2B/4B/9B QuantTrio AWQ G128 | MLP 的 qweight/qzeros 为 I32 packed UINT4，scales FP16；未量化参数按原包浮点 dtype 保存 | **QuantTrio 社区量化** [2B](https://huggingface.co/QuantTrio/Qwen3.5-2B-AWQ) / [4B](https://huggingface.co/QuantTrio/Qwen3.5-4B-AWQ) / [9B](https://huggingface.co/QuantTrio/Qwen3.5-9B-AWQ)，非 Qwen 官方量化发布 | `Qwen3.5-{2,4,9}B-AWQ` |
| MoE 35B-A3B | BF16 + 少量 FP32 | Qwen 官方 [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3.5-35B-A3B) | `Qwen3.5-35B-A3B` |
| MoE RTN W4/G128：本仓库产物 | 路由专家 INT4（I32 packed），FP16 scale；非路由参数保留原 BF16/FP32 | 本仓库 routed-only RTN 转换 | `Qwen3.5-35B-A3B-RTN-W4A16-G128` |
| MoE GPTQ W4/G128：公开量化包 | 路由专家 INT4（I32 packed）；未量化参数/scale 含 BF16、FP32 | **Qwen 官方量化** [Qwen/Qwen3.5-35B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4) · [量化声明](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4/blob/main/config.json) | `Qwen3.5-35B-A3B-GPTQ-Int4` |
| MoE mixed E4M3：交付的派生包 | E4M3 原始 UINT8 + FP16 scale；其余参数加载为 FP16 | 使用交付的完整包；公开下载地址和完整转换脚本未提供 | `Qwen3.5-35B-A3B-rpu-fp8-mixed` |
| legacy27：交付的浮点父 / W8 包 | 原父记录为 BF16；产物为 INT8 + FP16 scale，其余 FP16 | 使用交付的 legacy27 浮点父 | 父目录由交付提供；产物 `Qwen3.8-27B-w8a16` |

官方 27B 使用固定 revision `fc05daec18b0a78c049392ed2e771dde82bdf654`。其他命令按来源仓库下载。所有下载目录都要保留配套 tokenizer、processor、config 和完整分片。

<a id="cmd-dense-fp16"></a>

### 普通 Dense 浮点源：FP16 运行与量化准备

```bash
SIZE=4  # 可选：0.8、2、4、9
SRC="$RPU_MODEL_CACHE/Qwen3.5-${SIZE}B"
if [ "$SIZE" = 0.8 ]; then SRC="$RPU_MODEL_CACHE/qwen3_5-0.8b"; fi
hf download "Qwen/Qwen3.5-${SIZE}B" --local-dir "$SRC"
```

VL 使用同尺寸的 `vl/` FP16 配置并填写 `input.images`；不另存 FP16 checkpoint。

<a id="普通dense-w808249b"></a>

### 普通稠密模型 W8：0.8/2/4/9B

先按[普通 Dense 官方下载](#cmd-dense-fp16)准备相应浮点目录，再执行本步骤。

只改 `SIZE` 选择模型大小，其他路径随之匹配。此处W8为全部文本投影的统一量化，不选混合精度profile。

```bash
SIZE=9  # 可选：0.8、2、4、9；官方27B用下方独立命令
SRC="$RPU_MODEL_CACHE/Qwen3.5-${SIZE}B"
if [ "$SIZE" = 0.8 ]; then SRC="$RPU_MODEL_CACHE/qwen3_5-0.8b"; fi
DST="$RPU_MODEL_CACHE/qwen3_5_quant_ckpts/qwen3_5-${SIZE}b-w8a16"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 python/rpu_backend/quant/convert_qwen3_5.py \
  --src "$SRC" --dst "$DST" --method w8a16
```

例如9B输出供 [text/9b/w8a16_text.toml](text/9b/w8a16_text.toml) / [vl/9b/w8a16_text.toml](vl/9b/w8a16_text.toml) 使用。

<a id="普通dense-w4g3208249b"></a>

### 普通稠密模型 W4/G32：0.8/2/4/9B

先按[普通 Dense 官方下载](#cmd-dense-fp16)准备相应浮点目录，再执行本步骤。

```bash
SIZE=9  # 可选：0.8、2、4、9
SRC="$RPU_MODEL_CACHE/Qwen3.5-${SIZE}B"
if [ "$SIZE" = 0.8 ]; then SRC="$RPU_MODEL_CACHE/qwen3_5-0.8b"; fi
DST="$RPU_MODEL_CACHE/qwen3_5_quant_ckpts/qwen3_5-${SIZE}b-w4a16"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 python/rpu_backend/quant/convert_qwen3_5.py \
  --src "$SRC" --dst "$DST" --method w4a16_pgrp --group-size 32
```

例如9B输出供 [text/9b/w4a16_text_rtn_g32.toml](text/9b/w4a16_text_rtn_g32.toml) 使用；9B无公开W4 VL模板。0.8/2/4B有对应W4 Text/VL模板。

<a id="cmd-mlp-only-g32"></a>

### 2B 仅 MLP RTN G32：从官方父转换

先按[普通 Dense 官方下载](#cmd-dense-fp16)准备相应浮点目录，再执行本步骤。

以下命令沿用上文的 `$RPU_MODEL_CACHE`，输出目录应尚不存在。所有层的 MLP 都量化，Attention/GDN 保留 FP16；不要把普通 `--method w4a16_pgrp` 的全投影输出放进此目录。离线转换在 CPU 执行。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3.5-2B"
DST="$RPU_MODEL_CACHE/qwen3_5-2b-w4a16-g32-mlp-only"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 python/rpu_backend/quant/convert_qwen3_5.py \
  --src "$SRC" --dst "$DST" --profile w4_mlp_only
```

转换器写入仅 MLP 量化声明；Attention/GDN 保持 FP16，不能使用普通全投影 W4 产物代替。

<a id="cmd-mlp-w4-attn-gdn-w8"></a>

### 2B MLP RTN W4/G32 + Attention/GDN W8

先按[普通 Dense 官方下载](#cmd-dense-fp16)准备2B原始浮点父，再转换到独立缓存目录。`--profile` 固定 MLP G32 和 Attention/GDN 逐输出通道 W8，无需再传 `--method` 或校准数据。保留此 `DST` 名称以匹配 examples alias；不要用已有量化包再次量化。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3.5-2B"
DST="$RPU_MODEL_CACHE/qwen3_5-2b-w4a16-g32-mlp-w8-attn-gdn"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3_5.py \
  --src "$SRC" --dst "$DST" --profile w4_mlp_w8_attn_gdn
```

转换完成后，短模板和两档长输入模板分别运行如下；长模板均为 CPU4、TP8、语言 ACC16，输入 IDs 与对应 MLP-only 模板相同：

```bash
python examples/run.py --config examples/configs/qwen3_5/text/2b/w4a16_mlp_rtn_g32_attn_gdn_w8a16.toml
python examples/run.py --config examples/configs/qwen3_5/text/2b/w4a16_mlp_rtn_g32_attn_gdn_w8a16_p64_d4096.toml
python examples/run.py --config examples/configs/qwen3_5/text/2b/w4a16_mlp_rtn_g32_attn_gdn_w8a16_p4096_d4096.toml
```

加 `--check-config` 可在主机上检查模板而不启动模型；实际运行还会检查 checkpoint 配方。

<a id="cmd-quanttrio-awq"></a>

### 2B/4B/9B QuantTrio AWQ G128：直接使用社区原包

只改 `SIZE`，保留对应 alias 的目录。不要调用 RTN 转换器、手改 `quantization_config` 或把 GPTQ 的 zero-minus-one 当作 AWQ 零点。

```bash
SIZE=2  # 可选：2、4、9
hf download "QuantTrio/Qwen3.5-${SIZE}B-AWQ" \
  --local-dir "$RPU_MODEL_CACHE/Qwen3.5-${SIZE}B-AWQ"
```

加载器保留原始 UINT4 整数码、直接零点和 FP16 scale，仅重排为运行布局。未量化参数直接转 FP16，其中 FP32 GDN 参数不经 BF16 中转。

<a id="cmd-official27-source"></a>

### 官方 27B 源权重：固定版本下载一次

W8 和 W4 都从这个浮点目录独立转换。使用下列固定 revision。

```bash
hf download Qwen/Qwen3.5-27B --revision fc05daec18b0a78c049392ed2e771dde82bdf654 \
  --local-dir "$RPU_MODEL_CACHE/Qwen3.5-27B"
```

<a id="cmd-official27-w8"></a>

### 官方 27B W8：固定官方父 → W8

先按[官方 27B 固定版本下载](#cmd-official27-source)准备相应浮点目录，再执行本步骤。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3.5-27B"
DST="$RPU_MODEL_CACHE/qwen3_5_quant_ckpts/qwen3_5-27b-w8a16"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 python/rpu_backend/quant/convert_qwen3_5.py \
  --src "$SRC" --dst "$DST" --method w8a16
```

同一输出供27B W8的TP4/6/8 Text/VL模板使用，例如 [vl/27b/w8a16_text.toml](vl/27b/w8a16_text.toml)。

<a id="cmd-official27-w4"></a>

### 官方 27B W4/G32：固定官方父 → W4

先按[官方 27B 固定版本下载](#cmd-official27-source)准备相应浮点目录，再执行本步骤。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3.5-27B"
DST="$RPU_MODEL_CACHE/qwen3_5_quant_ckpts/qwen3_5-27b-w4a16"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 python/rpu_backend/quant/convert_qwen3_5.py \
  --src "$SRC" --dst "$DST" --method w4a16_pgrp --group-size 32
```

输出供27B TP8 [text/27b/w4a16_text_rtn_g32.toml](text/27b/w4a16_text_rtn_g32.toml) / [vl/27b/w4a16_text_rtn_g32.toml](vl/27b/w4a16_text_rtn_g32.toml) 使用。Dense 转换实现见 [convert_qwen3_5.py](../../../python/rpu_backend/quant/convert_qwen3_5.py)。

<a id="cmd-moe-rtn"></a>

### 35B-A3B MoE RTN W4/G128：独立脚本转换

RTN输入为官方浮点父；不能用Dense脚本代替。可把 `DST` 改成其他新目录，运行时 `--model "$DST"` 明确选择该产物。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3.5-35B-A3B"
DST="$RPU_MODEL_CACHE/Qwen3.5-35B-A3B-RTN-W4A16-G128"
hf download Qwen/Qwen3.5-35B-A3B --local-dir "$SRC"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 python/rpu_backend/quant/convert_qwen3_5_moe_rtn.py \
  --src "$SRC" --dst "$DST" --threads 8
```

入口：[python/rpu_backend/quant/convert_qwen3_5_moe_rtn.py](../../../python/rpu_backend/quant/convert_qwen3_5_moe_rtn.py)。同一输出也供 [vl/35b_a3b/w4a16_experts_rtn_g128.toml](vl/35b_a3b/w4a16_experts_rtn_g128.toml) 使用。

<a id="cmd-moe-gptq"></a>

### 35B-A3B MoE GPTQ：直接下载，无需转换

```bash
DST="$RPU_MODEL_CACHE/Qwen3.5-35B-A3B-GPTQ-Int4"
hf download Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --local-dir "$DST"
```

同一包也供 [vl/35b_a3b/w4a16_experts_gptq_g128.toml](vl/35b_a3b/w4a16_experts_gptq_g128.toml) 使用。

<a id="cmd-moe-mixed"></a>

### MoE mixed E4M3：使用交付包

公共运行步骤中的 `MODEL` 使用交付的 mixed E4M3 checkpoint。也可放到登记目录 `$RPU_MODEL_CACHE/Qwen3.5-35B-A3B-rpu-fp8-mixed`，省略 `--model`。此配方使用逐输出通道 scale，与官方 block-FP8 包格式不同。

VL改用 `vl/35b_a3b/fp8_e4m3_experts_attn_gdn.toml` 并填写图像路径，同样用 `--model` 指定该包。

<a id="历史-legacy27"></a>

<a id="历史legacy27-w8只能使用原exact-legacy浮点父"></a>

### Qwen3.8-27B 配套权重

使用独立的 legacy27 浮点父或 W8 包，按[该模型的准备步骤](legacy/27b/README.md#weights)运行。其 GDN 保留 FP16，与官方 27B 的统一量化权重分别准备。

### 关键配置项

字段通用语义见[参数说明](../PARAMETERS.md)。表中“省略值”指运行入口实际采用的值，和模板显式配置分开列。

| 层次 / 字段 | 模板值与有效省略值 | 范围与关联 |
|---|---|---|
| 配方：`example.dtype/registry_alias/profile_id` | 必填；按前面的配方表成套选择 | alias 决定权重目录，profile 约束精度、组件、模型大小和核数；只改 dtype 不能切换配方 |
| 模型：`example.target/architecture/opt_in` | Dense 为 `Qwen3_5ForConditionalGeneration`，MoE 为 `Qwen3_5MoeForConditionalGeneration`；Text target 为 `qwen3_5`，VL 为 `qwen3_5_vision` | Dense 量化及 Vision 所需 opt-in 已写入模板，必须与 profile 一致 |
| 文本：`input.prompt/prompt_token_ids/prompt_file` | 普通模板用 prompt；Dense 长输入用固定 token；MoE 长输入用正文文件及两个 SHA256 | 每种模板保留对应输入模式；MoE 长输入逐项校验文件哈希和 tokenizer 输出哈希，不重复或填充正文 |
| 输入长度：`input.prefill_tokens` | 基础省略，保留自然 P；变体 P29/64/4096，VL 还有 P512 | P≥1；Text 按模式补齐/截取；VL 的 P 包含图像和 chat token，目标 P 不能短于已有结构 |
| 补齐：`input.prefill_fill_token_id/prefill_filler_prompt` | 普通 Dense 长 Text 填充 248046；长 VL 用模板中的 filler 文本；MLP-only/AWQ 和 MoE 不补齐 | 与 `prefill_tokens` 配合；token 填充与 VL 文本填充是不同入口 |
| 图像：`input.images/image_size` | VL 填写图像路径；长 Dense VL 为 `test_frame.png`、448；省略 image_size 时保留原图尺寸交给 processor | image_size 为正整数，先缩放为该边长的正方形；图像尺寸和数量改变视觉 token 数 |
| 生成：`input.decode_steps/stop_on_eos` | D4/31/32/192/4096 见变体表；省略 D 为 4；**Text 省略 EOS 为 true，VL 为 false** | D≥0；固定长度模板显式 false。P+D 是逻辑缓存需求，输出 token 数为 D+1（未提前 EOS 时） |
| 拓扑：`rpu_execution.model.num_cores` | 默认 8；TP4/TP6 显式 4/6 | 与 profile 成对；MLP-only/AWQ、MoE W4、27B W4 为 TP8。TP6 的 MLP6，其余 Attention/GDN/head/Vision4 |
| 累加：`rpu_execution.prefill/vision.linear_acc32` | 基础省略或 false，为 ACC16；ACC32 变体显式 true | 文本项同时控制 prefill/decode；Vision 独立。MoE W4 文本仅接受 ACC16；不改变权重 |
| 分块：`rpu_execution.prefill/vision.chunk_size` | 默认和模板为 auto；2B `_chunk640` 显式文本 chunk 640 | 显式值为 16 的正整数倍，并受当前模型/组件布局约束；改变 P/图片后重新选块 |
| 行补齐：`rpu_execution.prefill.padding_rows` | MoE 长 Text 显式 0；其余省略为 auto | 0 为不增加执行行；非负整数固定补齐量，不改变逻辑 P |
| 计时：`run.timing_mode` | 省略为 per_call；长 Dense VL 有 legacy_dense_vl；MoE 长 Text 为 forward_argmax | 按变体表保留匹配模式，三种计时窗口不同；不直接混算速度 |
| 测量：`cpu.threads`、`run.warmup/runs/inference_mode` | 线程为省略或 4/8/12/24，轮次见变体；省略线程沿用环境，W/R 为 2/1，inference_mode 为 true | warmup≥0、runs≥1；false 使用 no_grad。比较配方时统一 CPU 线程、预热/测量次数与上下文 |

基础 Dense Text 为自然 P/D4、EOS 停止、W2/R3；基础 FP16 VL 为自然 P/D4、EOS 不停止、W2/R3。统一 W8/W4 的 Text 保持基础负载，VL 改成 D32 并显式关闭 EOS。MLP-only/AWQ 短模板为 CPU4、TP8、ACC16、inference mode、W2/R5。

<a id="使用"></a>

### 运行自己的输入

Dense 和 legacy27 通过 `example.registry_alias` 读取 [model_registry.py](../../../python/rpu_backend/model_registry.py) 中的相对目录；完整路径为 `$RPU_MODEL_CACHE/<登记目录>`。MoE 还可用 `--model /path/to/checkpoint` 覆盖目录。Text/VL 共用同一配方权重，VL 要另外填写 `input.images`。

```bash
CONFIG=examples/configs/qwen3_5/text/0_8b/fp16.toml
python3 examples/run.py --config "$CONFIG" --check-config
python3 examples/run.py --config "$CONFIG"
```

`--check-config` 检查 TOML 和登记关系；实际加载时检查权重配方及模型形状。输入、输出目录、CPU 线程和轮次可在配置副本修改。Dense 量化、视觉入口所需的 opt-in 已写在相应模板中。

MoE 长文本的 `input.prompt_file` 相对 **TOML 所在目录**解析。复制 P64/D4096 或 P4096/D4096 模板时，同时复制配套正文，保持正文内容及 `prompt_sha256/input_ids_sha256` 不变：

```bash
QWEN35_RUN_DIR=$(mktemp -d /tmp/qwen35-moe.XXXXXX)
cp examples/configs/qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128_p64_d4096.toml "$QWEN35_RUN_DIR/config.toml"
cp examples/configs/qwen3_5/text/35b_a3b/prefill_realtext.txt "$QWEN35_RUN_DIR/"
# 按需修改 run.output_dir、CPU 线程和轮次。
vi "$QWEN35_RUN_DIR/config.toml"
python3 examples/run.py --config "$QWEN35_RUN_DIR/config.toml" --check-config
python3 examples/run.py --config "$QWEN35_RUN_DIR/config.toml"
```

也可只复制 TOML，将 `input.prompt_file` 改为仓库内 `prefill_realtext.txt` 的实际绝对路径，仍保留两项哈希。六份 MoE 长文本模板共用这份正文，各自保留模板中的 token 哈希。

MoE 权重放在其他目录时，用同一入口显式指定；Text/VL 都支持 `--model`：

```bash
CONFIG=examples/configs/qwen3_5/text/35b_a3b/w4a16_experts_rtn_g128.toml
MODEL=/absolute/path/to/Qwen3.5-35B-A3B-RTN-W4A16-G128
python3 examples/run.py --config "$CONFIG" --model "$MODEL"
```
