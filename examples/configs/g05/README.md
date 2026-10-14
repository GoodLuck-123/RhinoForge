# G0.5 K1 配置

本目录使用客户 `vla_rtc` K1 交付，包含两图、三图的 FP16、W8A16 和动作 NVFP4 配方。六份模板均为 batch1、TP8、每相机一帧、十步 flow matching，输出 `[1,32,27]`。两图使用 `head_rgb`、`left_wrist_rgb`，三图增加 `right_wrist_rgb`。

<a id="选择配置"></a>

<a id="配置与权重对应"></a>

<a id="weight-recipes"></a>

## 配置说明

| 配方 | 两图 | 三图 | 默认累加 | 权重获取 |
|---|---|---|---|---|
| FP16 基准 | [配置](2cam/fp16.toml) | [配置](3cam/fp16.toml) | ACC32 | [原始 K1 交付](#weight-sources) |
| W8A16 | [配置](2cam/w8a16.toml) | [配置](3cam/w8a16.toml) | ACC16 | [同一原始 K1，加载时量化](#weight-sources) |
| W8 + 动作 NVFP4 | [配置](2cam/w8a16_action_nvfp4.toml) | [配置](3cam/w8a16_action_nvfp4.toml) | ACC16 | [同一原始 K1，加载时量化](#weight-sources) |

<a id="推荐配置与路径迁移"></a>

固定预处理 batch 的 `[example]` 入口见 [legacy 指南](legacy/README.md)，同样使用原始 K1 交付。

<a id="fp16-基准与配置增量"></a>

### FP16 基准

以同相机数的 `fp16.toml` 为基准：`g05.precision="fp16"`、`linear_accumulation="acc32"`、`math_precision="base"`。Vision、VLM 和 Action 已使用设备融合实现和各自的 Graph replay；Vision 逐相机执行，VLM 前缀 KV 供 Action 复用。位置及相同 token 的 embedding 缓存也属于基准。固定时间步的时间条件和 AdaLN 调制参数在准备阶段计算并缓存，十步去噪复用这些参数。

默认 `vision_camera_batch="auto"` 和预填充 `chunk_size="auto"` 按相机数、精度、ACC 与文本长度从 [selected_profiles.json](selected_profiles.json) 选择执行参数；FP16 的相机 batch 固定为 1。

### W8A16：主体投影量化，同时默认改为 ACC16

相对 FP16，`g05.precision` 改为 `"w8a16"`，`linear_accumulation` 从 `"acc32"` 改为 `"acc16"`，权重路径不变。加载阶段量化以下投影：

- Vision：Q/K/V/O、FC1/FC2。
- Language：MLP；全注意力层 Q/K/V/O；Gated DeltaNet（GDN）层 QKV/Z/Out。
- Action：Q/K/V/O、Gate/Up/Down，以及 attention gate。

这些投影使用逐输出通道 INT8 权重、FP16 scale 和激活。Vision patch/merger、Language norm/bias、GDN conv、标量及 `in_proj_a/b` 保持 FP16；embedding 不量化，输出转为 FP16 送设备；动作 IO 为 FP16，CPU AdaLN/时间条件为 FP32。

W8 还允许把两路或三路相机合批，AUTO 会按表选择。FP16 与 W8 的默认累加和调度也不同。

### 动作 NVFP4：在 W8 基础上改动作五类投影

`g05.precision` 改为 `"w8a16_nvfp4"`，仍读取原始 K1。加载时从原始 CPU FP16 权重直接转换 Action Q/O/Gate/Up/Down，使用 E2M1 四位浮点、K 维 block16 的 E4M3FN scale、FP32 tensor scale 和 `striped_v2` 布局。Action K/V/attention gate 以及 Vision/Language 继续使用 W8；激活仍为 FP16。这里不会先将五类投影变成 W8 再二次量化为 NVFP4。

### HIGH 和 ACC 是独立选项

`g05.math_precision="high"` 选择高精度非线性与 Newton RMSNorm，仍使用 FP16 激活和输出。`linear_accumulation="acc32"` 改变 Linear 累加；三种权重配方均可选择。HIGH 沿用 BASE 的相机分组与 chunk，权重目录和量化范围不变。

执行及加载实现见 [runner](../../runners/g05/optimized.py)、[runtime](../../../python/rpu_backend/adapters/g05/runtime.py)与[量化模块](../../../python/rpu_backend/adapters/g05/quant.py)。

<a id="准备权重并运行"></a>

<a id="quantization"></a>

<a id="weight-sources"></a>

<a id="weights"></a>

## 准备与运行

### 获取交付资产

六份模板读取同一原始 K1 checkpoint，量化在加载时完成，不生成独立目录。权重、release 源码、processor 和 golden 输入从同一客户 `vla_rtc` 交付获取。

| 资产 | 配置字段 | 用途 |
|---|---|---|
| 原始 K1 checkpoint | `input.checkpoint` | FP16 直接安装；W8/NVFP4 在 CPU 上转换后安装 |
| `vla_rtc` release 源码及 processor | `input.release_root` | 构造策略和处理真实 K1 输入 |
| 同交付 golden 文件 | `input.source_golden` | 读取图像、状态、RTC 前缀；`source_step` 选择其中一步 |
| 已保存的预处理 batch | `input.batch`，用于 `source="batch"` | 重放已有 token、相机及状态输入 |

原交付文件的存储 dtype 由文件确定；registry 别名 `g05-base` 对应这份交付资产。

<a id="run-fp16"></a>

<a id="run-w8"></a>

<a id="w8--action-nvfp4"></a>

<a id="使用"></a>

### 绑定路径并运行

从第一章选择模板。三种精度只需换 TOML，不改原始 checkpoint；以下以两图 W8 为例：

```bash
G05_CONFIG=examples/configs/g05/2cam/w8a16.toml
cp "$G05_CONFIG" /tmp/g05.toml
# input.checkpoint = 原始 K1；input.release_root = 同交付 release 源码。
# input.source_golden = 配对 golden 文件；run.output_dir = 本次结果目录。
# [libraries] 的 native/main/expansion 填实际路径，或删除整个表使用当前环境。
vi /tmp/g05.toml
python examples/run.py --config /tmp/g05.toml
```

三图改选 `3cam/` 同精度文件，并提供三图输入。`[libraries]` 三项一起配置，见[算子库说明](../../../docs/runtime_assets.md)。

<a id="输入与输出"></a>

### 输入格式

`source="synthetic"` 从 golden 中读取真实 K1 图像、状态和 RTC 前缀，文本由 release processor 编码。默认构造指定长度的测试指令；自定义 `instruction` 时只选择一个与编码结果一致的实际长度。

重放已有输入时设 `source="batch"`、填写 `batch`，移除 `source_golden/source_step/instruction/text_lengths`。此时 token 长度和相机顺序来自保存文件。

`T` 为实际 TEXT/COT/PRED_TEXT 数量，包含模板与图像边界文本，排除 IMAGE/STATE；两图前缀长度 `P=T+129`，三图 `P=T+193`。

<a id="输入与执行变体"></a>

### 常调参数

| 字段 | 模板值 | 可选范围 | 修改时的关联 |
|---|---|---|---|
| `g05.linear_accumulation` | FP16 为 acc32，量化为 acc16 | acc16 / acc32 | 与权重精度独立；安装前选择 |
| `g05.math_precision` | base | base / high | HIGH 作用见配置说明，沿用 BASE 调度 |
| `input.text_lengths` | `[32,52,64,128,192,256]` | 32–256 整数列表 | 单长度可改 `text_tokens`；两字段互斥；自定义指令选择匹配长度 |
| `input.source_step` | 1 | 0 / 1 | 选择 RTC/no-RTC 实际输入 |
| `g05.vision_camera_batch` | auto | FP16 为 1；W8/NVFP4 为 1 或相机数 | AUTO 按调度表选择合批和 chunk |
| `rpu_execution.components.language_model.prefill.chunk_size` | auto | 自动或有效整数 | 显式值通过运行时容量检查，优先于调度表 |
| `cpu.threads` | 8 | 正整数 | CPU 线程 |
| `run.warmup/runs/seed` | 2 / 6 / 7 | 预热≥0、采样≥1、seed≥0 | CPU 比较使用保存的实际 FM 噪声 |

AUTO 未匹配到调度表时使用逐相机执行和自动 chunk。公共字段见[参数说明](../PARAMETERS.md)。多长度分别启动进程运行。

## 输出

输出 `report.json`、`actions.pt` 和 `input_batch.pt`。多长度分别保存到 `text_N/`，包含日志与退出码。
