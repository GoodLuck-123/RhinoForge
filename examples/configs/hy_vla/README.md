# HyVLA 配置

本目录使用腾讯 Hy-Embodied-0.5-VLA-UMI，提供两图和三图的三种精度配置。输入为 batch1、TP8、十步动作去噪，输出归一化动作 `[50,20]`。两图使用 `top_head`、`hand_left`，三图增加 `hand_right`。

<a id="选择配置"></a>

<a id="配置与权重对应"></a>

<a id="weight-recipes"></a>

## 配置说明

| 配方 | 两图 | 三图 | 权重获取 |
|---|---|---|---|
| FP16 | [配置](2cam/fp16.toml) | [配置](3cam/fp16.toml) | [原始 UMI 目录](#download-parent) |
| W8A16 | [配置](2cam/w8a16.toml) | [配置](3cam/w8a16.toml) | [同一原始目录，加载时量化](#download-parent) |
| W8 + 动作 NVFP4 | [配置](2cam/w8a16_action_nvfp4.toml) | [配置](3cam/w8a16_action_nvfp4.toml) | [同一原始目录，加载时量化](#download-parent) |

本组可构造确定性测试输入，也可重放保存的 batch；UMI 固定输入入口见 [hy_embodied](../hy_embodied/README.md)。

<a id="fp16-基准与配置增量"></a>

### FP16 基准

以同相机数的 `fp16.toml` 为基准。除了浮点权重，默认已启用 attention TP8、Action MLP 多核、专家 RMSNorm 对齐到 16 行、Prefill 只生成所需 KV、Vision KV 保留在片上，以及各阶段 Graph replay。Action 的前 6 层 Gate/Up 权重驻留片上，减少十步去噪中的重复搬运。

三种模板都设置 `resident_fp8_scale_layers=0`、`rmsnorm_vwarp="auto"`；前者表示 NVFP4 scale 不额外按层驻留，后者自动选择 RMSNorm 的执行路径。本入口固定 TP8，不提供独立 ACC 配置字段。

### W8A16：主投影量化，增加驻留层数

`hy_vla.precision` 从 `"fp16"` 改为 `"w8a16"`，源目录不变。加载时量化 ViT 的 Q/K/V/O、FC1/FC2，VLM text/vision 两分支的 Q/K/V/O、Gate/Up/Down，以及实际动作 `_v` 分支的相同七类投影。W8 使用逐输出通道 INT8 权重、FP16 scale 和激活。

运行器同时把 Action Gate/Up 驻留层数从 6 改为 12。ViT patch/merger、norm/bias/final norm 保持 FP16；host token embedding、state/时间预处理为 FP32，图内动作 IO 为 FP16。

### 动作 NVFP4：七类动作投影全部量化为四位浮点

在 W8 基础上将 `hy_vla.precision` 改为 `"w8a16_nvfp4"`。Action `_v` 分支 Q/K/V/O/Gate/Up/Down 全部改用 NVFP4，包括 K/V。加载时从原始权重的 FP16 舍入结果生成 E2M1 四位权重、K 维 block16 的 E4M3FN scale 和 FP32 tensor scale，激活保持 FP16。

Vision/VLM 继续为 W8，Gate/Up 驻留层数进一步由 12 改为 24。三种精度的 O 投影都未启用额外驻留。

选择与加载实现见 [runner](../../runners/hy_vla/optimized.py)、[权重加载](../../../python/rpu_backend/adapters/hy_vla/weights.py)和 [NVFP4 转换](../../../python/rpu_backend/adapters/hy_vla/nvfp4.py)。

<a id="准备权重并运行"></a>

<a id="quantization"></a>

<a id="weight-sources"></a>

<a id="weights"></a>

## 准备与运行

<a id="download-parent"></a>

### 下载原始权重

六份配置均将 `input.checkpoint` 指向同一个完整原始目录，不需要离线量化包。源权重为腾讯 [Hy-Embodied-0.5-VLA-UMI](https://huggingface.co/tencent/Hy-Embodied-0.5-VLA-UMI/tree/3f53d1f8d2bc587c523cfdc9f1041ceee42c2524)，发布权重为 BF16。保留 `model.safetensors`、tokenizer 和归一化文件。

```bash
HYVLA_SRC=/absolute/path/Hy-Embodied-0.5-VLA-UMI
hf download tencent/Hy-Embodied-0.5-VLA-UMI \
  --revision 3f53d1f8d2bc587c523cfdc9f1041ceee42c2524 \
  --local-dir "$HYVLA_SRC"
```

<a id="run-fp16"></a>

<a id="run-w8"></a>

<a id="w8--action-nvfp4"></a>

<a id="使用"></a>

### 绑定路径并运行

从第一章选择 TOML，以下以两图 FP16 为例：

```bash
HYVLA_CONFIG=examples/configs/hy_vla/2cam/fp16.toml
cp "$HYVLA_CONFIG" /tmp/hyvla.toml
# input.checkpoint = HYVLA_SRC 的实际绝对路径；run.output_dir = 本次结果目录。
# [libraries] 的 native/main/expansion 填实际路径，或删除整个表使用当前环境。
vi /tmp/hyvla.toml
python examples/run.py --config /tmp/hyvla.toml
```

三种精度共用 `HYVLA_SRC`；三图改选 `3cam/` 文件。保留模板别名 `hy-embodied-0.5-vla-umi` 时，由 `RPU_MODEL_CACHE/Hy-Embodied-0.5-VLA-UMI` 定位，目录名区分大小写。运行库见[算子库说明](../../../docs/runtime_assets.md)。

<a id="输入与输出"></a>

### 输入格式

默认 `source="synthetic"` 使用 checkpoint tokenizer 和 `run.seed` 生成测试 RGB、state、noise。指令至少编码出 T 个 token，入口截取前 T 个，再补至 64 槽；运行摘要记录实际 token 和截断情况。重放已有输入时设 `source="batch"`、填写 `batch` 并移除 `instruction`，相机数、token、mask 和位置与文件一致。

### 从同一观测准备文本长度变体

`prepare_hyvla_text_batches.py` 保留三图、state 和 noise，为 T16/32/48/64 分别选择可完整编码回读的自然语言前缀，并重新生成 mask 与位置。

```bash
python scripts/prepare_hyvla_text_batches.py \
  --source-observation /path/to/input_batch.pt --checkpoint "$HYVLA_SRC" \
  --output /path/to/hyvla-text-inputs
```

输出 `hyvla_3cam_t16_batch.pt`、`hyvla_3cam_t32_batch.pt`、`hyvla_3cam_t48_batch.pt`、`hyvla_3cam_t64_batch.pt` 和 `inputs.json`。在三图配置中设置 `source="batch"`、相应的 `text_tokens` 与 `batch` 路径，删除 `instruction`。也可用 `--source-assets` 读取包含 `synth_images.pt`、`state.pt`、`initial_noise.pt` 的目录；`--instruction-file` 提供自定义指令。

<a id="输入与执行变体"></a>

### 常调参数

| 字段 | 模板值或入口默认 | 可选范围 | 修改时的关联 |
|---|---|---|---|
| `input.cameras` | 2 或 3 | 2 / 3 | 使用对应相机输入；权重目录不变 |
| `input.text_tokens` | 32 | 1–64 | 实际有效 token 数；指令至少编码出 T 个 token |
| `input.prefix_len` | auto | 合法 16 对齐行数 | 自动为 `ceil16(2+相机数×58+T)`；显式值介于自动值和完整 64 语言槽容量之间 |
| `input.source` | synthetic | synthetic / batch | batch 模式填写文件，移除 instruction |
| `input.instruction` | 入口默认提示词 | 自定义文本 | 截取前 T 个 token，再补至 64 槽 |
| `hy_vla.resident_fp8_scale_layers` | 0 | 0 / 4 / 8 | 4/8 仅用于动作 NVFP4；不改变权重目录 |
| `hy_vla.rmsnorm_vwarp` | auto | auto / "16" | 自动选择或固定为 16 |
| `cpu.threads` | 模板省略，入口默认 12 | 正整数 | CPU 线程 |
| `run.warmup/runs/seed` | 3 / 6 / 0 | 预热至少 3，采样至少 1，seed 非负 | 影响输入生成和测量次数 |
| `input.reference` | 省略 | 同输入的已保存设备reference | 记录动作输出的比较 |

T16/32/48/64 时，两图 P144/160/176/192，三图 P192/208/224/240。公共字段见[参数说明](../PARAMETERS.md)。更换冷配置使用新进程和独立输出目录。

## 输出

输出 `actions.pt`、`reference.pt` 和 `report.json`；synthetic 模式另存 `input_batch.pt`。
