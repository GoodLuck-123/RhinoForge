# Llama 配置
[配置索引](../CONFIGURATIONS.md) · [使用](#使用)

入口：`python examples/run.py --config <file>`；选择下表模板。

<a id="weights"></a>
<a id="weight-sources"></a>
## 权重来源与存储类型

| 权重 / 用途 | 磁盘存储精度 | 发布者与下载来源 | 下载到的目录（相对于 `$RPU_MODEL_CACHE`） |
|---|---|---|---|
| 官方 Llama 3.2 1B base，当前模板的标准输入 | **BF16**，已核对官方 HF safetensors 元数据 | [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) · [精度元数据](https://huggingface.co/api/models/meta-llama/Llama-3.2-1B) | `Llama-3.2-1B` |
| 历史登记镜像 | BF16；未证明与官方当前快照相同 | 第三方 [unsloth/Llama-3.2-1B](https://huggingface.co/unsloth/Llama-3.2-1B) | 同一登记目录；不要把两个 snapshot 混装 |

<a id="weight-recipes"></a>
## 模块精度与配方

| config | 运行时模块权重 / 激活 | 输入来源 | 获取方式、源码与完整命令 |
|---|---|---|---|
| [3_2_1b/fp16.toml](3_2_1b/fp16.toml) | decoder Attention/MLP、embedding、norm、lm_head 权重均 FP16；主路径激活 FP16 | 上表原始 BF16 完整 checkpoint | **不涉及低比特量化**；[examples/runners/llama/runner.py](../../runners/llama/runner.py) 的 `RPUModelForCausalLM.from_pretrained(dtype=torch.float16)`；[下载并运行](#quantization) |

仅提供浮点配置，加载时转换 dtype。

<a id="quantization"></a>
## 下载与运行

**本族现有模板无需离线量化，也没有W8/W4转换步骤。** 按[权重来源](#weights)下载完整原浮点权重到 `$RPU_MODEL_CACHE/Llama-3.2-1B`；在仓库根目录、已安装项目依赖的 Python 环境中运行。

```bash
export RPU_MODEL_CACHE=/absolute/path/to/model_cache  # 替换为真实缓存根目录
hf download meta-llama/Llama-3.2-1B --local-dir "$RPU_MODEL_CACHE/Llama-3.2-1B"
python examples/run.py --config examples/configs/llama/3_2_1b/fp16.toml
```

模板的 `example.registry_alias` 自动读取上述缓存目录。加载入口为 [examples/runners/llama/runner.py](../../runners/llama/runner.py)；dtype 转换由加载器完成，无需另存量化目录。

## 选择配置

| 配置 | 模型 / 精度 | 登记状态 |
|---|---|---|
| [3_2_1b/fp16.toml](3_2_1b/fp16.toml) | Llama 3.2 1B，FP16文本生成 | SUPPORTED，精确profile |

此目录无其他尺寸、Vision或W8/W4配置。

## 使用

在 `RPU_MODEL_CACHE/Llama-3.2-1B` 准备原始权重，复制模板修改prompt/输出目录：

```bash
python examples/run.py --config examples/configs/llama/3_2_1b/fp16.toml --check-config
python examples/run.py --config /path/to/my_llama.toml
```

公共runner保存 `report.json` 和生成输出。

采样设置示例：`cpu.threads=8`、`run.warmup=2`、`run.runs=5`；模板默认3次正式运行。

## 可修改参数

| 字段 | 默认 / 条件 |
|---|---|
| `input.prompt` | `Explain what a compiler does.`；真实文本分词 |
| `input.decode_steps` | 4；保持原profile范围 |
| `rpu_execution.prefill.chunk_size` | auto，显式值依native容量检查 |
| `cpu.threads` | 可选正整数；省略继承环境 |
| `run.warmup/runs/seed` | 2 / 3 / 0 |

## 输入与限制

FP16模板使用原浮点资产，修改dtype不能生成量化权重。输入上下文与 cache 由运行时检查；冷配置改变需新进程。

支持边界见[模型矩阵](../../../docs/model_support.md)，运行库准备见[算子库规则](../../../docs/runtime_assets.md)。
