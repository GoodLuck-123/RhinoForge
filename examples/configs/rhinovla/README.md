# RhinoVLA 配置

RhinoVLA 分为公开 v1 和 v3。两者的模型权重、动作维度和输入各自配套，先按手中的资产选择入口。

<a id="路径迁移"></a>
<a id="weight-sources"></a>
<a id="weight-recipes"></a>
<a id="weights"></a>

## 配置说明

| 模型 | 配置 | 权重与配套资产 | 详细说明 |
|---|---|---|---|
| v1，72D 动作 | [官方 FP16](v1/fp16.toml) | 官方 `HuixiAI/RhinoVLA` checkpoint、固定源码和 LIBERO 输入；加载为 FP16 | [下载、输入准备与运行](v1/README.md) |
| v3，96D 动作 | [FP16](v3/fp16.toml)、[专家 W8](v3/w8a16_action.toml)、[完整专家 W8](v3/w8a16_action_adarms.toml) | 同一套公开浮点 checkpoint、固定模型源码和准备工具生成的请求；W8 在加载时转换 | [基准与配置增量](v3/README.md#baseline) |
| v3，RTC | [FP16](v3/rtc_fp16.toml)、[专家 W8](v3/rtc_w8a16_action.toml)、[完整专家 W8](v3/rtc_w8a16_action_adarms.toml)、[全流水线 W8](v3/rtc_w8a16_pipeline.toml)、[HIGH/vector QK](v3/rtc_w8a16_pipeline_high_fused_vector_qk.toml) | 与对应 non-RTC 配方共用固定源码和原始 checkpoint；请求增加已确定的动作前缀与 `delay=0..6` | [输入契约、运行与验证范围](v3/README.md#rtc) |
| v3，全流水线量化 | [全流水线 W8](v3/w8a16_pipeline.toml)、[HIGH/vector QK](v3/w8a16_pipeline_high_fused_vector_qk.toml) | 与上述 v3 配置共用原始 checkpoint；量化范围扩至 Vision、Text、Action IO 和 AdaRMS | [量化范围与计算精度](v3/README.md#full-pipeline) |

W8A16 表示所选线性投影使用 INT8 权重、FP16 激活和逐输出通道 FP16 scale。v3 的 non-RTC 和 RTC factory 配置共用一份原始权重，转换结果保存在运行进程中。

<a id="quantization"></a>
<a id="使用"></a>

## 准备与运行

- **v1：** 按 [v1 说明](v1/README.md#quantization)下载固定版本、准备输入，再运行 `examples/run.py`。
- **v3：** 按 [v3 说明](v3/README.md#quantization)下载固定源码与 checkpoint，生成输入和 factory JSON，再运行 `examples/run.py`。

[九份独立配置](v3/controlled/README.md)提供输入缓存、HIGH、vector K/QK 与 partial RoPE 的单独选择，通过 `scripts/validate_rhinovla_v3.py` 运行；增加 `--rtc --delay 6` 可在原配方上启用 RTC，delay 可选 `0..6`。
