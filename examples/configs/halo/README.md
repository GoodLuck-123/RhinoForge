# HALO Action Expert 配置

[配置索引](../CONFIGURATIONS.md) · [使用](#使用)

本配置运行 FP16 Action Expert 的一个去噪 step。输入为已准备好的前缀 K/V 和单步 CPU 张量。

<a id="weight-recipes"></a>
<a id="quantization"></a>
## 配置说明

| 配置 | 组件 / 精度 | 登记范围 |
|---|---|---|
| [action_expert/fp16.toml](action_expert/fp16.toml) | Action Expert 单步，投影 W16/A16 | `SUPPORTED_LIMITED` |

加载器将原 EMA 转为 FP16；无需离线量化。runner 先通过 `insert_prefix()` 安装逐层前缀 K/V，再重复调用 `step()`。单步 query 为 18 行：第 0、17 行为文本，1–16 行为动作；动作维度 14，前缀长度 4706。

动作专家保留 Graph 重放，Attention 使用 2 核，MLP 使用 8 核。`chunk_size="auto"` 按这一单步形状规划。图像编码、语言前缀生成和完整多步策略在调用前准备。实现见 [runner](../../runners/halo/runner.py)、[权重定义](../../../python/rpu_backend/adapters/halo/weights.py)和[运行时](../../../python/rpu_backend/adapters/halo/runtime.py)。

<a id="weights"></a>
<a id="weight-sources"></a>
## 准备与运行

### 权重来源

| 资产 | 来源与存储精度 |
|---|---|
| 官方预训练 EMA | [qshou-coder/halo_pt_weight](https://huggingface.co/qshou-coder/halo_pt_weight)，revision `2bd3a209c6c1536afa43da6540c817196dde96e0`；`ema.safetensors` 的 1624 个张量均为 BF16 |
| 官方无 EM-CoT 微调 EMA | [qshou-coder/halo_ft_weight](https://huggingface.co/qshou-coder/halo_ft_weight)，revision `d7a149753d76247d44ee5e73ca010ee43686cfe1`；`no_cot/ema.safetensors` 的 1615 个张量均为 BF16 |
| 已登记动作专家 EMA | `halo_action_expert` → `$RPU_MODEL_CACHE/HALO/action_expert.safetensors`；原交付 CoT EMA，BF16，需提供来源清单和配对单步 fixture |

官方源码见 [HALO](https://github.com/qshou-coder/HALO)。登记 EMA 与上述公开权重的对应关系尚未核实；公开下载文件应单独存放：

```bash
hf download qshou-coder/halo_pt_weight ema.safetensors --revision 2bd3a209c6c1536afa43da6540c817196dde96e0 --local-dir /path/to/official-halo-pretrain
hf download qshou-coder/halo_ft_weight no_cot/ema.safetensors --revision d7a149753d76247d44ee5e73ca010ee43686cfe1 --local-dir /path/to/official-halo-no-cot
```

<a id="使用"></a>

### 输入与运行

将匹配的 EMA 文件或符号链接放在 `$RPU_MODEL_CACHE/HALO/action_expert.safetensors`。未设置 `RPU_MODEL_CACHE` 时，缓存根目录为 `~/.cache/rhinoforge/models`。该别名指向文件，加载器不追加子目录。

复制模板，将 `input.fixture` 改为配对输入文件：

```bash
python examples/run.py --config /path/to/halo.toml --check-config
python examples/run.py --config /path/to/halo.toml
```

`--check-config` 检查 TOML；运行入口在加载权重前检查固定位置、文本/动作行索引、前缀长度、K/V 容器和 checkpoint 文件。

### 输入与参数

fixture 为可用 `torch.load(..., weights_only=True)` 读取的 CPU 张量字典，可直接包含以下字段，也可放在 `kwargs` 中：

| 字段 | 结构 |
|---|---|
| `x_t` / `timestep` | 浮点 `[16,14]` / `[16]`；数值有限，timestep 各行相同 |
| `packed_action_position_ids` / `packed_text_ids` | 整数 `[16]` / `[2]` |
| `packed_position_ids` | 整数 `[18]`，所有位置相同 |
| `packed_text_indexes` / `packed_action_token_indexes` | `[0,17]` / `[1,...,16]` |
| `key_values_lens` | `[4706]` |
| `past_key_values` | `key_cache` 和 `value_cache` 各 28 层，每层浮点 `[4706,2,128]`；层容器可为整数键字典、序列或堆叠张量 |

索引必须非负。保留原权重与 fixture 的配对关系，输出为单步动作张量。

| 可修改字段 | 默认 / 条件 |
|---|---|
| `input.fixture` | 配对的单步 CPU 张量文件 |
| `rpu_execution.components.action_expert.action.chunk_size` | `auto`，遵循单步准入 |
| `cpu.threads` | 可选正整数，省略时继承环境 |
| `run.warmup/runs/seed` | 2 / 3 / 0 |

完整模型、多步策略和 CPU 质量参考不在此入口范围内。环境准备见[示例指南](../../README.zh.md)和[运行资产](../../../docs/runtime_assets.md)。

## 输出

保存 `report.json` 和动作专家单步输出快照；保留独立输出后再提交下一次输入。
