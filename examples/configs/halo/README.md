# HALO Action Expert 配置
[配置索引](../CONFIGURATIONS.md) · [使用](#使用)

入口：`examples/run.py --config <file>`；本页只整理已有接口。

<a id="weights"></a>
<a id="weight-sources"></a>
## 官方来源与原始存储精度

| 资产 | 官方下载 / 源码 | 原始存储精度与可复现边界 |
|---|---|---|
| 官方预训练 EMA | [qshou-coder/halo_pt_weight](https://huggingface.co/qshou-coder/halo_pt_weight)；[HALO 官方源码](https://github.com/qshou-coder/HALO) | `2bd3a209c6c1536afa43da6540c817196dde96e0` 的 `ema.safetensors` 文件头中 1624 个张量均 **BF16** |
| 官方无 EM-CoT 微调 EMA | [qshou-coder/halo_ft_weight](https://huggingface.co/qshou-coder/halo_ft_weight) | `d7a149753d76247d44ee5e73ca010ee43686cfe1` 的 `no_cot/ema.safetensors` 文件头中 1615 个张量均 **BF16**；官方说明这是无 EM-CoT 版本 |
| 本仓库已登记动作专家 EMA | `halo_action_expert` → `$RPU_MODEL_CACHE/HALO/pytorch_weights_current` | 交付文件为 `<checkpoint>/ema.safetensors`；历史加载器记为 BF16，需按提供方清单确认与公开 EMA 的对应关系；还需原交付来源清单与配对单步 fixture |

下列命令只下载已公开的官方文件，**不能据此运行本页已登记动作专家，不能将它们改名覆盖登记 EMA**。本轮只核对公开元数据与文件头，不加载或测试延期的 HALO 模型。

```bash
hf download qshou-coder/halo_pt_weight ema.safetensors --revision 2bd3a209c6c1536afa43da6540c817196dde96e0 --local-dir /path/to/official-halo-pretrain
hf download qshou-coder/halo_ft_weight no_cot/ema.safetensors --revision d7a149753d76247d44ee5e73ca010ee43686cfe1 --local-dir /path/to/official-halo-no-cot
```

<a id="weight-recipes"></a>
## 模块精度与配方

| 配置 | 模块实际 W/A 精度 | 输入父权重 → 转换方式与命令 |
|---|---|---|
| `action_expert/fp16.toml` | **仅动作专家单步**的已安装投影为 W16/A16；仅包含此组件 | 原交付 EMA + 匹配单步 fixture → 加载时转换到 FP16；[无需量化](#quantization)，没有 W8/W4/NVFP4 配方或离线量化 CLI。原资产齐备后的[既有运行方式](#使用)保留，模型测试仍延期 |

实际 FP16 上传见 [adapters/halo/weights.py](../../../python/rpu_backend/adapters/halo/weights.py)，由加载器调用。运行使用原配对资产。

<a id="quantization"></a>
## 下载与运行

**无需量化转换。** 本目录只有 `action_expert/fp16.toml`，读取已有 EMA 浮点资产，没有 W8/W4/NVFP4 转换脚本或量化输出目录。HALO 模型加载、测试仍延期；EMA 与公开权重的亲缘未核实，见下方来源说明。

## 选择配置

| 配置 | 范围 / 精度 | 既有登记 |
|---|---|---|
| [action_expert/fp16.toml](action_expert/fp16.toml) | HALO单个Action Expert step，FP16 | SUPPORTED_LIMITED，精确单步profile |

这是单组件一步，不是HALO全模型或多步策略。HALO的继续优化/测试按既有决定延期，测试延期。

## 使用

准备 `halo_action_expert` registry资产及对应CPU tensor fixture，填 `input.fixture`：

```bash
python examples/run.py --config /path/to/halo.toml --check-config
python examples/run.py --config /path/to/halo.toml
```

运行例子按原资产/单步合同准备，不将任意观测当作这个fixture。

## 可修改参数

| 字段 | 默认 / 条件 |
|---|---|
| `input.fixture` | 同profile的单步CPU tensor文件 |
| `rpu_execution.components.action_expert.action.chunk_size` | auto；原单步准入 |
| `cpu.threads` | 可选正整数；省略继承环境 |
| `run.warmup/runs/seed` | 2 / 3 / 0 |

## 输入与限制

保留原组件范围、权重与fixture语义，不扩展为全HALO推理、量化配置或机器人验收。公共入口只计实际调用并保存结果，不运行CPU质量参考。

运行环境与结果说明见[示例指南](../../README.zh.md)，运行库准备见[算子库规则](../../../docs/runtime_assets.md)。
