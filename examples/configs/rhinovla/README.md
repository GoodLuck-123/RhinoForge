# RhinoVLA 配置
[配置索引](../CONFIGURATIONS.md) · [使用](#使用)

入口：`python examples/run.py --config <file>`；推荐 [v1/official_fp16.toml](v1/official_fp16.toml)。v1/72D与v3/96D是不同模型合同，权重和输入分别准备。

<a id="weights"></a>
<a id="weight-sources"></a>
## 官方来源与原始存储精度

| 模型 / 资产 | 官方下载 / 源码 | 原始存储精度与边界 |
|---|---|---|
| 官方 v1 / 72D | [HuixiAI/RhinoVLA](https://huggingface.co/HuixiAI/RhinoVLA) 的 `rhinovla_pretrain.ckpt`；[官方源码](https://github.com/HUIXI-AI/RhinoVLA) | 固定权重版本 `7f3708650e302f395e7c63607b48b56fb96b8076`、源码 `a3edf0c1addf4bf849192642bff3ad15559db3f4`。HF 未提供该 `.ckpt` 的张量 dtype 清单，**原始存储类型待确认**；FP16 是运行选择，不是下载精度证明 |
| 客户 v3 / 96D | 调用者交付 checkpoint、源码、factory 与配对输入 | 由资产提供方交付；**原始存储精度按实际交付记录确认**，不能用 v1 72D 权重替代 |

v1 的官方下载和固定源码准备见 [v1 完整命令](v1/README.md#weight-sources)。v3 无通用官方下载命令，需先取得客户资产和可信 factory；下方运行命令不提供这些资产。

<a id="weight-recipes"></a>
## 模块精度与配方

W/A 表示权重/激活精度；W8 为逐输出通道 INT8 权重、FP16 scale 与 A16。下表区分专家量化和完整流水线量化。

| 配置组 | 模块实际 W/A 精度 | 输入父权重 → 转换方式与命令 |
|---|---|---|
| `v1/fp16.toml` | 原 caller-factory 浮点路径 | 调用者提供可信 factory、checkpoint 与配对请求；[旧入口说明](v1/README.md#旧-caller-factory-入口) |
| `v1/official_fp16.toml` | 官方非 RTC Vision/Text/Action 浮点路径，设备投影 W16/A16 | 固定官方 v1 父权重 → 加载时转 FP16；[资产/fixture 准备及运行](v1/README.md#quantization)，无需量化 |
| `v3/fp16.toml` | v3 Vision/Text/Action 浮点路径，设备投影 W16/A16 | 客户 v3 父权重 → 可信 factory 加载；[FP16 命令](v3/README.md#fp16-loading) |
| `v3/expert_w8a16.toml` | 仅 Action expert Q/K/V/O/Gate/Up/Down 七类投影 W8/A16；Vision、Text、AdaRMS 条件投影与 Action IO 保持 FP16 | 客户 v3 父权重 → `convert_expert_for_rpu()`；[加载时命令](#quantization) |
| `v3/full_expert_w8a16.toml` | 上述七类投影 + 18 组 AdaRMS 条件投影 W8/A16；最终 norm 预计算、Action IO、Vision/Text 仍浮点 | 同一父权重 → 再用 `_prepare_full_w8_cond_weights()` / `_prepare_full_w8_adarms_tables()`；[加载时命令](#quantization) |
| `v3/full_pipeline_w8a16*.toml` | Vision block/patch/merger、Text 七类投影、Action expert/IO/AdaRMS 已执行线性投影 W8/A16；norm/bias/embedding/位置张量仍浮点。HIGH/vector QK 变体改变非线性/归一化计算，不改变这份 W/A 配方 | 同一父权重 → 专家转换、`quantize_text_prefix()`、Vision 安装和 `_prepare_denoise_loop_weights()`；[两套完整命令](#quantization) |
| `v3/controlled/*.toml` | 旧验证器的 FP16、专家 W8、完整流水线 W8 及 HIGH/传输优化变体；不能仅由文件名 `full` 推断与 factory 的 full-expert 相同 | 同一客户模型的历史配对资产 → [受控片段逐配方表与命令](v3/controlled/README.md#weight-recipes) |

v3 量化均在安装时执行，不生成独立量化目录；本仓库没有该客户模型的离线 checkpoint 导出 CLI，也没有已核实的公开设备量化下载。`--check-config` 只校验配置。

<a id="quantization"></a>
## 加载时量化

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

**实现位置**：[python/rpu_backend/adapters/rhinovla/convert.py](../../../python/rpu_backend/adapters/rhinovla/convert.py)；[python/rpu_backend/adapters/rhinovla/pipeline.py](../../../python/rpu_backend/adapters/rhinovla/pipeline.py)；[python/rpu_backend/adapters/rhinovla/precision.py](../../../python/rpu_backend/adapters/rhinovla/precision.py)。可信 factory 创建 `RhinoVLAOnRPU`；专家 W8 调用 `convert_expert_for_rpu()`，完整专家增加 `_prepare_full_w8_cond_weights()` / `_prepare_full_w8_adarms_tables()`，全流水线再含 `quantize_text_prefix()`、Vision 安装和 `_prepare_denoise_loop_weights()`。上述模块由 factory 调用，不导出 checkpoint。

## 选择配置

| 组别 | 覆盖 / 区别 | 入口与状态 |
|---|---|---|
| [v1 FP16](v1/README.md) | 官方非RTC，完整72D/有效64D | 公共 example，使用 `official_fp16.toml` 与固定完整官方资产 |
| [v3 factory](v3/README.md) | FP16、仅expert W8、full-expert W8、full-pipeline W8/HIGH | 公共example，需要调用者实现可信factory；不新增登记支持 |
| [v3 controlled](v3/controlled/README.md) | 原验证器的9份env/冷配置片段 | 独立validate_v3入口，不可传给公共example |

## 使用

v1按[官方完整资产说明](v1/README.md)准备固定 source/weights/fixture；v3按 [factory说明](v3/README.md)准备实际request与模型集成：

```bash
python examples/run.py --config /path/to/rhinovla.toml --check-config
python examples/run.py --config /path/to/rhinovla.toml
```

运行使用空闲板卡、既有设备锁和匹配native。`--check-config`只检查结构；support由已登记profile派生，不能用配置路径替代认证。

## 可修改参数与限制

版本和精度范围由模板决定，输入、factory、组件冷配置按各版本README修改；运行中不能改变冷选项。性能/误差来自对应输入和构建，固定误差阈值只作诊断。全族只使用正式main/DDR两份REF，见 [算子库规则](../../../docs/runtime_assets.md)。<a id="路径迁移"></a>
