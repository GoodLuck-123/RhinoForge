# RhinoVLA v3 受控验证片段

[上级选型](../README.md) · [完整配置索引](../../../CONFIGURATIONS.md)

独立验证片段采用 `[env]` 与平铺 `[rpu_execution]`，与公共 factory example schema 不同。该验证器及其专用输入不随本仓库发布；公共运行请选择 [factory 模板](../README.md)，不要把独立片段直接传给 `examples/run.py`。

## 配方与对应入口

- 浮点：使用 [fp16.toml](../fp16.toml)。
- Expert W8：使用 [expert_w8a16.toml](../w8a16_action.toml)，仅量化指定专家范围。
- 完整专家 W8：使用 [full_expert_w8a16.toml](../w8a16_action_adarms.toml)。
- 全流水线 W8 与传输优化：使用 [full_pipeline_w8a16.toml](../w8a16_pipeline.toml)。
- 全流水线 W8、HIGH、融合与 vector QK：使用 [full_pipeline_w8a16_high_precision_vector_qk.toml](../w8a16_pipeline_high_fused_vector_qk.toml)。

## 参数与限制

冷环境和组件参数修改后使用新进程与匹配的输入、资产。HIGH 控制非线性精度，与 Linear 累加模式不同；部分 RoPE、vector K/QK 与融合选项需要 factory 实际实现并绑定相应冷配置。保持量化安装范围、输入状态刷新、缓存与 Graph 生命周期一致。

<a id="weights"></a>
## 权重来源

本归档/受控入口沿用对应客户交付及配对转换资产；来源限制和现有转换实现见[上级权重说明](../README.md#weights)。保留原 checkpoint 和配套输入。

<a id="weight-recipes"></a>
## 量化范围

| 配方 | 权重与激活 |
|---|---|
| expert W8 | Action Q/K/V/O/Gate/Up/Down 为 W8/A16，其余组件浮点 |
| full expert W8 | 加入 AdaRMS 条件投影 W8/A16，最终 norm 预计算及 IO 保持浮点 |
| full pipeline W8 | Vision/Text/Action 已执行的线性投影 W8/A16；norm、bias、embedding 和位置张量浮点 |

HIGH 与向量 QK 改变非线性/归一化计算。新集成使用[对应 factory 模板与完整运行命令](../README.md#quantization)；本目录的旧环境片段不能直接交给统一运行入口。
