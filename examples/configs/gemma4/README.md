# Gemma4 配置
[配置索引](../CONFIGURATIONS.md) · [使用](#使用)

入口：`python examples/run.py --config <file>`；选择下表模板。

<a id="weights"></a>
<a id="weight-sources"></a>
## 权重来源与存储类型

| 权重 / 用途 | 磁盘存储精度 | 发布者与下载来源 | 下载到的目录（相对于 `$RPU_MODEL_CACHE`） |
|---|---|---|---|
| 官方 Gemma4 E4B Instruct，当前模板的标准输入 | **BF16**，已核对官方 HF safetensors 元数据 | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) · [精度元数据](https://huggingface.co/api/models/google/gemma-4-E4B-it) | `gemma-4-e4b-it` |

使用 Google 官方完整多模态 checkpoint，但本目录只开放文本 MVP；不把官方模型支持的音频/图像能力当作此 runner 的支持。HF 仓库名为大写 `E4B`，本地登记目录为小写 `e4b`；按官方页面要求取得访问权限。

<a id="weight-recipes"></a>
## 模块精度与配方

| config | 运行时模块权重 / 激活 | 输入来源 | 获取方式、源码与完整命令 |
|---|---|---|---|
| [e4b/fp16.toml](e4b/fp16.toml) | 设备 decoder/PLE 权重与主路径激活 FP16；CPU embedding/PLE/head 保留源权重，计算时转 FP32；head 输出 FP32 | 上表原始 BF16 完整 checkpoint | **不涉及低比特量化**；[examples/runners/gemma4/runner.py](../../runners/gemma4/runner.py) 先以 BF16 加载；[python/rpu_backend/adapters/gemma4/__init__.py](../../../python/rpu_backend/adapters/gemma4/__init__.py) 安装 设备 FP16 和 CPU 浮点分支；[下载并运行](#quantization) |

仅提供浮点配置，加载时转换 dtype。

<a id="quantization"></a>
## 下载与运行

**本族现有模板无需离线量化，也没有W8/W4转换步骤。** 按[权重来源](#weights)下载完整原浮点权重到 `$RPU_MODEL_CACHE/gemma-4-e4b-it`；在仓库根目录、已安装项目依赖的 Python 环境中运行。

```bash
export RPU_MODEL_CACHE=/absolute/path/to/model_cache  # 替换为真实缓存根目录
hf download google/gemma-4-E4B-it --local-dir "$RPU_MODEL_CACHE/gemma-4-e4b-it"
python examples/run.py --config examples/configs/gemma4/e4b/fp16.toml
```

模板的 `example.registry_alias` 自动读取上述缓存目录。加载入口为 [examples/runners/gemma4/runner.py](../../runners/gemma4/runner.py)；dtype 转换由加载器完成，无需另存量化目录。

## 选择配置

| 配置 | 模型 / 精度 | 登记状态 |
|---|---|---|
| [e4b/fp16.toml](e4b/fp16.toml) | Gemma4 E4B；设备 FP16 decoder + CPU embedding/PLE/head | SUPPORTED_LIMITED，text MVP |

文本MVP采用CPU/设备混合路径；输出head在CPU以FP32计算，完整请求计时包含这些CPU工作。没有量化模板，未支持图像/多模态或任意长上下文。

## 使用

在 `RPU_MODEL_CACHE/gemma-4-e4b-it` 准备原始HF checkpoint（registry别名 `gemma4-e4b`），复制模板修改prompt/输出目录：

```bash
python examples/run.py --config examples/configs/gemma4/e4b/fp16.toml --check-config
python examples/run.py --config /path/to/my_gemma4.toml
```

公共 runner 保存实际生成输出。

采样设置示例：`cpu.threads=8`、`run.warmup=2`、`run.runs=5`；模板默认3次正式运行。

## 可修改参数

| 字段 | 默认 / 条件 |
|---|---|
| `input.prompt` | `Explain what a compiler does.`；真实文本分词 |
| `input.decode_steps` | 4；保持既有模型准入范围 |
| `rpu_execution.prefill.chunk_size` | auto；显式请求仍由native检查 |
| `cpu.threads` | 可选正整数；省略继承环境 |
| `run.warmup/runs/seed` | 2 / 3 / 0 |

## 输入与限制

原始checkpoint为BF16；入口将decoder安装为设备 FP16，CPU部分保留该混合路径，不要求预先生成一份FP16权重。修改冷配置需新进程。

运行环境与结果说明见[示例指南](../../README.zh.md)，运行库准备见[算子库规则](../../../docs/runtime_assets.md)。
