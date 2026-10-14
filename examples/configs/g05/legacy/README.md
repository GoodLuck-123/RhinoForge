<a id="g05-legacy-配置"></a>

# G0.5 固定 batch 配置

本目录提供 `[example]` 格式的 [fp16.toml](fp16.toml)，用于 K1 RTC/FM 客户交付的预处理 batch。两/三图、多长度与 W8/NVFP4 选择见[模型入口](../README.md)。

<a id="选择配置"></a>

<a id="配置与权重对应"></a>

<a id="weight-recipes"></a>

<a id="fp16-基准"></a>

## 配置说明

本目录仅有 [fp16.toml](fp16.toml)，使用[原始 K1 交付资产](#weight-sources)，无需离线量化。

Vision、Language 和 Action 主投影及设备 IO 使用 FP16，CPU AdaLN、时间条件使用 FP32。入口经 `patch_g05_policy_for_rpu()` 安装融合执行路径；Vision、Language 预填充和 Action 的 chunk 均为 auto，使用相应 Graph 缓存。`example.dtype="fp16"`，本目录只有这一种精度配方。

每次调用复制原 batch 再执行 `predict_action`，保持相同观测及调用方提供的 RTC 前缀。实现见[固定 batch runner](../../../runners/g05/runner.py)。

<a id="准备权重并运行"></a>

<a id="quantization"></a>

<a id="weight-sources"></a>

<a id="weights"></a>

<a id="准备并运行"></a>

## 准备与运行

### 获取资产

| 资产 | 配置字段 | 要求 |
|---|---|---|
| 原始 K1 checkpoint | `input.checkpoint` | 客户 `vla_rtc` 交付目录 |
| 配对 release 源码及 processor | `input.release_root` | 与 checkpoint 同一交付 |
| 预处理输入 | `input.batch` | 由该 release processor 生成，含 `samples`、`pixel_values` |

本配置直接加载浮点权重，不需要离线量化目录。registry 别名 `g05-base` 标识这份客户资产。

<a id="run-fp16"></a>

<a id="使用"></a>

### 绑定路径并运行

以下在仓库根目录执行，编辑副本后运行：

```bash
cp examples/configs/g05/legacy/fp16.toml /tmp/g05_legacy_fp16.toml
# input.checkpoint = 原始 K1 checkpoint；input.release_root = 同交付源码。
# input.batch = 同交付预处理 batch；在 [run] 添加本次 output_dir。
vi /tmp/g05_legacy_fp16.toml
python examples/run.py --config /tmp/g05_legacy_fp16.toml
```

### 常调参数

| 字段 | 模板值 | 可选范围 | 修改时的关联 |
|---|---|---|---|
| `input.batch` | 占位路径 | 同交付预处理 batch | 包含 `samples` 和 `pixel_values`，由 release processor 生成 |
| 三个组件的 `chunk_size` | auto | 自动或有效整数 | 分别位于 vision_encoder.vision、language_model.prefill、action_expert.action |
| `run.warmup/runs/seed` | 2 / 3 / 7 | 预热≥0、采样≥1、seed≥0 | 正式运行次数可调整 |
| `cpu.threads` | 未指定 | 正整数 | 省略时继承运行环境 |

组件路径位于 `rpu_execution.components` 下。公共字段见[参数说明](../../PARAMETERS.md)，运行库见[算子库说明](../../../../docs/runtime_assets.md)。
