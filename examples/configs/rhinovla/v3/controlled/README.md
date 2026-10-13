# RhinoVLA v3 独立运行配置

[模型与权重准备](../README.md#quantization) · [统一入口配置](../README.md) · [完整索引](../../../CONFIGURATIONS.md)

本目录供 `scripts/validate_rhinovla_v3.py run` 使用。配置为 `[env]` 与平铺的 `[rpu_execution]`，不直接交给 `examples/run.py`。

## 配置选择

| 配置 | 功能 |
|---|---|
| [FP16](fp16.toml) | 浮点流水线，包含缓存、融合和图重放 |
| [Action W8](w8a16_action.toml) | 动作专家 Q/K/V/O、Gate/Up/Down 使用 W8A16，条件投影保持浮点 |
| [Pipeline W8](w8a16_pipeline.toml) | 已执行的 Vision/Text/Action 线性投影使用 W8A16 |
| [输入缓存](w8a16_pipeline_input_cache.toml) | Pipeline W8，增加图像输入缓冲复用 |
| [HIGH](w8a16_pipeline_high.toml) | 输入缓存与 HIGH 非线性、归一化计算 |
| [HIGH 融合](w8a16_pipeline_high_fused.toml) | HIGH，增加归一化与激活融合、gate tanh 预计算 |
| [vector K](w8a16_pipeline_high_fused_vector_k.toml) | HIGH 融合，增加向量 K 归一化 |
| [vector QK](w8a16_pipeline_high_fused_vector_qk.toml) | 增加向量 Q 归一化 |
| [partial RoPE](w8a16_pipeline_high_fused_vector_qk_partial_rope.toml) | vector QK，增加部分旋转位置编码 |

W8 使用 INT8 权重、FP16 scale 和激活；norm、bias、embedding 与位置张量保留浮点。HIGH 控制非线性与归一化，Linear 累加方式由各组件单独设置。

## 运行

按[上级说明](../README.md#quantization)准备固定源码与 checkpoint，随后执行：

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
