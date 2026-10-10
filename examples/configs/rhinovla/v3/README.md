# RhinoVLA v3 factory 配置
[上级选型](../README.md) · [使用](#使用)

本目录供调用者集成v3 pipeline，入口为 `examples/run.py --config <TOML>`。需要独立的模型源码、checkpoint、真实输入、factory及factory JSON，仓库不附带这些私有资产。

<a id="weights"></a>
<a id="weight-sources"></a>
## 来源与原始存储精度

| 资产 | 下载 / 源码来源 | 原始存储精度与边界 |
|---|---|---|
| v3 / 96D checkpoint、源码和 factory | 调用者/客户配对交付；由资产提供方交付 | 存储 dtype 按交付清单确认；安装器接收相应浮点模块后量化 |
| 可公开下载的 v1 / 72D | [HuixiAI/RhinoVLA](https://huggingface.co/HuixiAI/RhinoVLA)；[官方源码](https://github.com/HUIXI-AI/RhinoVLA) | 不同模型，不能作为 v3 量化父权重。官方下载命令只在 [v1 页](../v1/README.md#weight-sources)适用 |

因此没有可诚实提供的 v3 通用 `hf download` 命令。先向交付方取得实际 checkpoint、模型源码、可信 factory/JSON 和 request，再执行下方加载命令；公开 v1 下载不能补齐这组资产。

<a id="weight-recipes"></a>
## 模块精度与配方

以下全部由同一份匹配的客户 v3 浮点父权重加载，W8 表示逐输出通道 INT8 + FP16 scale，激活 A16。

| 配置 | 模块实际 W/A 精度 | 转换实现 / 完整命令 |
|---|---|---|
| `fp16.toml` | Vision/Text/Action 的 设备 投影 W16/A16 | 无量化；[FP16 加载命令](#fp16-loading) |
| `expert_w8a16.toml` | Action expert Q/K/V/O/Gate/Up/Down 为 W8/A16；其余 Vision/Text、AdaRMS 条件投影和 IO 为 FP16 | `convert_expert_for_rpu()`；[下方专家命令](#quantization) |
| `full_expert_w8a16.toml` | 再增加 18 组 AdaRMS 条件投影 W8/A16；最终 norm 预计算、IO 与 Vision/Text 仍浮点 | 再调用 `_prepare_full_w8_cond_weights()` / `_prepare_full_w8_adarms_tables()`；[下方完整专家命令](#quantization) |
| `full_pipeline_w8a16.toml` | Vision block/patch/merger、Text 七类投影、Action expert/IO/AdaRMS 线性投影 W8/A16；norm/bias/embedding/位置仍浮点 | 再覆盖 `quantize_text_prefix()`、Vision 安装和 `_prepare_denoise_loop_weights()`；[下方完整流水线命令](#quantization) |
| `full_pipeline_w8a16_high_precision_vector_qk.toml` | 同上一行 W8/A16；HIGH 与向量 Q/K norm 改变计算精度，不是另一种权重格式 | 同一转换范围；[HIGH 命令](#全流水线-w8--highvector-qk) |

实现源码和入口调用链列在下方。它们是运行时安装函数，**没有离线 checkpoint 转换 CLI，也不导出量化目录**；配置必须由真实 factory 安装并通过首次推理前的范围核对。

<a id="quantization"></a>
## 加载时量化

<a id="fp16-loading"></a>
### FP16 加载

```bash
cp examples/configs/rhinovla/v3/fp16.toml /tmp/rhinovla_fp16.toml
# input.checkpoint/request/factory_config = 同一客户交付的实际路径。
# input.runtime_factory = 已安装可信 module:callable；run.output_dir = 本次独立目录。
vi /tmp/rhinovla_fp16.toml
python examples/run.py --config /tmp/rhinovla_fp16.toml
```

此块只安装浮点模型，不执行 W8 转换；以下各块才选择不同 W8 范围。

以下命令在仓库根目录执行。量化在最后一行加载模型时发生，不生成独立量化目录。先替换实际路径，再按注释用 `vi` 编辑副本并保存；`--check-config` 只校验配置，不能代替最后的运行命令。

### 仅动作专家七类投影 W8

```bash
cp examples/configs/rhinovla/v3/expert_w8a16.toml /tmp/rhinovla_expert.toml
# input.checkpoint = 客户原始 v3 checkpoint；input.request = 配对请求文件。
# input.runtime_factory = 已安装且可信的真实 module:callable。
# input.factory_config = 该 factory 的真实 JSON 配置。
# 在 [run] 中添加 output_dir；保留模板 expert_w8a16/full_w8a16、trust_model_code 和 opt_in。
vi /tmp/rhinovla_expert.toml
python examples/run.py --config /tmp/rhinovla_expert.toml
```

### 完整专家 W8（加条件投影）

```bash
cp examples/configs/rhinovla/v3/full_expert_w8a16.toml /tmp/rhinovla_full_expert.toml
# input.checkpoint = 客户原始 v3 checkpoint；input.request = 配对请求文件。
# input.runtime_factory = 已安装且可信的真实 module:callable。
# input.factory_config = 该 factory 的真实 JSON 配置。
# 在 [run] 中添加 output_dir；保留模板 expert_w8a16/full_w8a16、trust_model_code 和 opt_in。
vi /tmp/rhinovla_full_expert.toml
python examples/run.py --config /tmp/rhinovla_full_expert.toml
```

### 全流水线 W8

```bash
cp examples/configs/rhinovla/v3/full_pipeline_w8a16.toml /tmp/rhinovla_pipeline.toml
# input.checkpoint = 客户原始 v3 checkpoint；input.request = 配对请求文件。
# input.runtime_factory = 已安装且可信的真实 module:callable。
# input.factory_config = 该 factory 的真实 JSON 配置。
# 在 [run] 中添加 output_dir；保留模板 expert_w8a16/full_w8a16、trust_model_code 和 opt_in。
vi /tmp/rhinovla_pipeline.toml
python examples/run.py --config /tmp/rhinovla_pipeline.toml
```

<a id="全流水线-w8--highvector-qk"></a>
### 全流水线 W8 + HIGH/向量 QK

```bash
cp examples/configs/rhinovla/v3/full_pipeline_w8a16_high_precision_vector_qk.toml /tmp/rhinovla_high.toml
# input.checkpoint = 客户原始 v3 checkpoint；input.request = 配对请求文件。
# input.runtime_factory = 已安装且可信的真实 module:callable。
# input.factory_config = 该 factory 的真实 JSON 配置。
# 在 [run] 中添加 output_dir；保留模板 expert_w8a16/full_w8a16、trust_model_code 和 opt_in。
vi /tmp/rhinovla_high.toml
python examples/run.py --config /tmp/rhinovla_high.toml
```

factory 模块须能被当前 Python 环境导入，并实际安装所选范围；仅改精度标签不会转换。绝对 checkpoint 路径不依赖 `RPU_MODEL_CACHE`。**v1/v3 FP16 均无需量化**；v1 官方 72D 权重不能代替 v3 客户 96D 父权重。

**实现位置**：[python/rpu_backend/adapters/rhinovla/convert.py](../../../../python/rpu_backend/adapters/rhinovla/convert.py)；[python/rpu_backend/adapters/rhinovla/pipeline.py](../../../../python/rpu_backend/adapters/rhinovla/pipeline.py)；[python/rpu_backend/adapters/rhinovla/precision.py](../../../../python/rpu_backend/adapters/rhinovla/precision.py)。可信 factory 创建 `RhinoVLAOnRPU`；专家 W8 调用 `convert_expert_for_rpu()`，完整专家增加 `_prepare_full_w8_cond_weights()` / `_prepare_full_w8_adarms_tables()`，全流水线再含 `quantize_text_prefix()`、Vision 安装和 `_prepare_denoise_loop_weights()`。上述模块由 factory 调用，不导出 checkpoint。

## 选择配置

| 配置 | 量化范围 / 区别 | 状态 |
|---|---|---|
| [fp16.toml](fp16.toml) | 浮点，保留原factory参数 | 调用者factory模板 |
| [expert_w8a16.toml](expert_w8a16.toml) | 仅expert Q/K/V/O/Gate/Up/Down七类投影W8 | 调用者factory模板 |
| [full_expert_w8a16.toml](full_expert_w8a16.toml) | 上述投影加18组AdaRMS条件投影W8；最终norm预计算与IO浮点 | 调用者factory模板，非全流水线W8 |
| [full_pipeline_w8a16.toml](full_pipeline_w8a16.toml) | text/vision/expert/action IO/AdaRMS全流水线W8；HIGH关闭 | 实验配方 |
| [full_pipeline_w8a16_high_precision_vector_qk.toml](full_pipeline_w8a16_high_precision_vector_qk.toml) | 同全W8范围，启用HIGH与vector Q/K norm | 实验配方 |

`full_w8a16` 只控制 expert，完整流水线范围由模板指定。旧 `w8a16.toml` 请改用 `expert_w8a16.toml`。

## 使用

采样设置示例为 `run.warmup=2`、`run.runs=5`；前三份模板设 `cpu.threads=8`，两份 full-pipeline 设 `cpu.threads=12`。所有模板默认2次预热、3次正式运行。

复制模板，填写checkpoint、`runtime_factory="your_package.module:create_runtime"`、request、factory_config和输出目录。factory必须接收配置的冷选择并返回 `RhinoVLAPolicy.from_factory` 接受的runtime：

```bash
python examples/run.py --config examples/configs/rhinovla/v3/expert_w8a16.toml --check-config
python examples/run.py --config /path/to/my_rhinovla_v3.toml
```

factory 必须实际安装模板请求的量化范围与冷配置，runner 会在首次推理前核对。冷配置变化后使用新进程，板卡运行持既有设备锁；`--check-config` 仅检查配置。

## 可修改参数

| 字段 | 默认 / 选择 | 条件 |
|---|---|---|
| `input.runtime_factory/request/factory_config` | 必填实际集成/输入 | `trust_model_code=true`明确采用调用者代码 |
| `input.expert_w8a16/full_w8a16` | W8模板严格bool | JSON中的同名值必须同类型同值；不能绕过模板范围 |
| `input.num_steps` | 10 | AdaRMS表必须匹配实际schedule |
| 组件 `prefill/vision.linear_acc32` | full-pipeline默认false=ACC16 | true选ACC32；不改变action/patch/AdaRMS，不可放进action表 |
| `cpu.threads` | full-pipeline12，其余按运行环境 | full-pipeline默认no_grad、W2/R3 |
| `rpu_execution.components.*` | 模板已选冷缓存/融合 | factory需声明和绑定实际实现，不具备即拒绝 |

## 输入与限制

expert-only 不量化其他组件，也不支持 packed QKV；full-pipeline 的 norm/bias/embedding/位置张量仍为浮点。HIGH 控制非线性精度，与 Linear ACC32 分开。

factory 须负责缓存、量化安装和每次调用的输入状态刷新。库依赖为 [正式 main/DDR 两份 REF](../../../../docs/runtime_assets.md)；独立验证器的 [controlled 片段](controlled/README.md) 不能混入 factory schema。
