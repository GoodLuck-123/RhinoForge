# RhinoVLA v3 独立运行配置

[模型与权重准备](../README.md#quantization) · [统一入口配置](../README.md) · [完整索引](../../../CONFIGURATIONS.md)

本目录供 `scripts/validate_rhinovla_v3.py run` 使用。配置为 `[env]` 与平铺的 `[rpu_execution]`，不直接交给 `examples/run.py`。

<a id="weight-recipes"></a>

## 配置说明

FP16 基准已开启多视角与 prefix 准备缓存、设备 patch/merger、算子融合、AdaRMS 预计算及驻留、十步去噪展开、共享缓存、对齐 KV 和图重放。各项含义见 [v3 基准说明](../README.md#baseline)。

| 配置 | 比较基准 | 具体增量 |
|---|---|---|
| [fp16.toml](fp16.toml) | 浮点基准 | Vision/Text/Action 使用 FP16 投影 |
| [w8a16_action.toml](w8a16_action.toml) | FP16 | `EXPERT_W8A16=1`，量化动作专家 Q/K/V/O/Gate/Up/Down |
| [w8a16_pipeline.toml](w8a16_pipeline.toml) | 专家 W8 | 再启用 `FULL_W8A16=1`，覆盖 Vision、Text、动作 IO 和全部 AdaRMS 条件投影 |
| [w8a16_pipeline_input_cache.toml](w8a16_pipeline_input_cache.toml) | 全流水线 W8 | 开启 gate tanh 预计算和 Vision 输入缓存，prefill 增加 `padding_rows=10` |
| [w8a16_pipeline_high.toml](w8a16_pipeline_high.toml) | [全流水线 W8 + 输入缓存](w8a16_pipeline_input_cache.toml) | 开启 `HIGH_PRECISION`，使用高精度非线性和归一化 |
| [w8a16_pipeline_high_fused.toml](w8a16_pipeline_high_fused.toml) | HIGH | 开启 `HIGH_PRECISION_FUSIONS`，融合高精度 norm+shift 和 SiLU×up |
| [w8a16_pipeline_high_fused_vector_k.toml](w8a16_pipeline_high_fused_vector_k.toml) | HIGH 融合 | 开启 `VECTOR_K_NORM`，K norm 使用已有对齐缓冲上的 Newton V32 |
| [w8a16_pipeline_high_fused_vector_qk.toml](w8a16_pipeline_high_fused_vector_qk.toml) | vector K | 再开启 `VECTOR_Q_NORM`，Q norm 补齐物理行并使用 Newton V32 |
| [w8a16_pipeline_high_fused_vector_qk_partial_rope.toml](w8a16_pipeline_high_fused_vector_qk_partial_rope.toml) | vector QK | 再开启 `PARTIAL_ROPE`，使用独立的 RoPE 调度路径 |

大写开关均带 `RPU_RHINOVLA_` 前缀，取值在文件中为字符串 `"1"`。[w8a16_pipeline.toml](w8a16_pipeline.toml) 表示全流水线量化；factory 的 [w8a16_action_adarms.toml](../w8a16_action_adarms.toml) 在专家 W8 基础上增加 AdaRMS 条件投影量化。

`PARTIAL_ROPE` 路径仍旋转完整的 128 维 head，处理 31 个逻辑 token，并保留对齐填充行。其余 HIGH/vector 开关的算术和缓冲细节见[上级说明](../README.md#全流水线-w8--highvector-qk)。这些选择不改变线性权重的 W8A16 配方。

<a id="weight-sources"></a>
<a id="weights"></a>
<a id="quantization"></a>

## 准备与运行

按[上级说明](../README.md#quantization)准备固定源码与 checkpoint。源码还需包含三个相机的示例视频 `datasets/example_lerobot_v3/videos/observation.images.{head_color,hand_left,hand_right}/chunk-000/file-000.mp4`；CPU 环境安装模型依赖和 PyAV。随后执行：

```bash
python scripts/validate_rhinovla_v3.py prepare \
  --source "$RHINOVLA_V3_SRC" --checkpoint "$RHINOVLA_V3_CKPT" \
  --fixture /tmp/rhinovla_v3_inputs.pt
python scripts/validate_rhinovla_v3.py run \
  --source "$RHINOVLA_V3_SRC" --checkpoint "$RHINOVLA_V3_CKPT" \
  --fixture /tmp/rhinovla_v3_inputs.pt \
  --config examples/configs/rhinovla/v3/controlled/fp16.toml \
  --output /tmp/rhinovla_v3_actions.pt
```

`prepare` 使用公开示例视频生成两组图像输入，state/noise 使用固定种子。`run` 固定十步去噪，检查输出形状、有限值、未激活动作槽、不同请求的独立输出与图重放，保存动作 tensor。`--runs` 控制重放次数；线程默认 12。修改环境开关或组件配置后使用新进程。

### RTC：沿用同一精度与执行配置

上表九份配置均可通过 `run --rtc` 选择 RTC，不需要另外准备权重、fixture 或修改 TOML。`--delay` 接受 `0,1,2,3,4,5,6` 中的一个整数；启用 RTC 且省略该参数时默认为 `6`。例如，在保留全流水线 W8、HIGH、融合、vector Q/K 与 partial RoPE 设置的情况下运行：

```bash
python scripts/validate_rhinovla_v3.py run \
  --source "$RHINOVLA_V3_SRC" --checkpoint "$RHINOVLA_V3_CKPT" \
  --fixture /tmp/rhinovla_v3_inputs.pt \
  --config examples/configs/rhinovla/v3/controlled/w8a16_pipeline_high_fused_vector_qk_partial_rope.toml \
  --rtc --delay 6 --runs 3 \
  --output /tmp/rhinovla_v3_rtc_actions.pt
```

替换 `--config` 即可选择其余配方；将 `--delay` 改为 `0` 可检查无已确定动作的 RTC 请求。`--rtc` 只用于 `run`，`prepare` 输出由两种模式共用；不带 `--rtc` 时不能单独使用 `--delay`。

脚本为每组 fixture 生成固定种子的测试动作前缀，除原有检查外，还检查输出保留掩码处理后的前缀，输出 metadata 记录 `rtc` 和 `delay`。这段测试前缀不代表机器人已执行的动作；应用集成须按照 [RTC 输入契约](../README.md#rtc)提供同一归一化 96D 空间内真实已确定的动作。每次命令只验证所选 delay，不能将单次运行当作所有 delay 或所有精度的验证结果。

### 参数与输出

`--source`、`--checkpoint` 和 `--fixture` 必须来自同一套配对资产；`--config` 选择上表完整模板，`--output` 保存动作文件。`--runs` 默认 `3`，`--threads` 默认 `12`；去噪固定为十步。`--rtc` 默认为关闭，开启后 `--delay` 默认为 `6`、范围为 `0..6`。fixture 与统一入口的 `request.pt` 分别使用。

实现见[运行脚本](../../../../../scripts/validate_rhinovla_v3.py)、[权重与表准备](../../../../../python/rpu_backend/adapters/rhinovla/convert.py)、[流水线](../../../../../python/rpu_backend/adapters/rhinovla/pipeline.py)。
