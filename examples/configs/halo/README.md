# HALO Action Expert 配置

[配置索引](../CONFIGURATIONS.md) · [使用](#使用)

入口：`examples/run.py --config <file>`。登记范围为 FP16 Action Expert 单步，设备测试状态见[示例索引](../../README.zh.md)。

<a id="weights"></a>
<a id="weight-sources"></a>
## 权重来源

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

<a id="weight-recipes"></a>
<a id="quantization"></a>
## 配置与精度

| 配置 | 组件 / 精度 | 登记范围 |
|---|---|---|
| [action_expert/fp16.toml](action_expert/fp16.toml) | Action Expert 单步，投影 W16/A16 | `SUPPORTED_LIMITED` |

加载器将原 EMA 转为 FP16；无需离线量化，没有 W8/W4/NVFP4 配方。实现见 [weights.py](../../../python/rpu_backend/adapters/halo/weights.py)。

## 使用

将匹配的 EMA 文件或符号链接放在 `$RPU_MODEL_CACHE/HALO/action_expert.safetensors`。未设置 `RPU_MODEL_CACHE` 时，缓存根目录为 `~/.cache/rhinoforge/models`。该别名指向文件，加载器不追加子目录。

复制模板，将 `input.fixture` 改为配对输入文件：

```bash
python examples/run.py --config /path/to/halo.toml --check-config
python examples/run.py --config /path/to/halo.toml
```

`--check-config` 检查 TOML；运行入口在加载权重前检查 checkpoint 文件和 fixture 结构。

## 输入与参数

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
