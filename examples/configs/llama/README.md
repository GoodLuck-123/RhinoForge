# Llama 配置

[配置索引](../CONFIGURATIONS.md) · [准备与运行](#使用)

本目录提供 Llama 3.2 1B base 的 FP16 文本生成配置。

## 配置说明

<a id="weight-recipes"></a>
<a id="配置与权重对应关系"></a>

| 配置 | 计算范围 | 权重获取 |
|---|---|---|
| [3_2_1b/fp16.toml](3_2_1b/fp16.toml) | Llama 3.2 1B base 文本生成，FP16 | [官方 BF16 权重](#weights)，加载时转 FP16 |

<a id="fp16-基准做了什么"></a>

### 精度与执行

Attention、MLP、embedding、norm 和输出 head 的模型权重加载为 FP16，主路径使用 FP16 激活。公共入口通过 `RPUModelForCausalLM` 安装模型，按实际 prompt 与 decode 长度创建 KV cache；prefill 写入整段提示词，之后逐 token decode 并复用已有 K/V。

模板设置 `rpu_execution.prefill.chunk_size="auto"`，由执行规划器选择分块。分块大小改变执行安排，`input.prompt` 和 `input.decode_steps` 决定实际输入与生成工作量。加载器见 [runner](../../runners/llama/runner.py)，生成和 cache 调用见[公共文本入口](../../common/text.py)。

## 准备与运行

<a id="weight-sources"></a>
<a id="weights"></a>
<a id="quantization"></a>
<a id="权重准备"></a>

### 获取权重

`example.registry_alias="llama-3.2-1b"` 固定读取 `$RPU_MODEL_CACHE/Llama-3.2-1B`。权重与 tokenizer 来自同一套 base 1B 资产。

在 [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) 取得访问权限后，在仓库根目录下载完整 checkpoint：

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
hf download meta-llama/Llama-3.2-1B \
  --local-dir "$RPU_MODEL_CACHE/Llama-3.2-1B"
```

官方文件存储为 BF16，运行时由加载器转换，无需另存 FP16 文件。仓库登记记录还使用过 BF16 的 [unsloth 镜像](https://huggingface.co/unsloth/Llama-3.2-1B)；单个缓存目录只放一套完整权重。

<a id="使用"></a>

### 输入与运行

在已安装项目依赖的设备环境中，复制模板并填写文本和输出目录：

```bash
cp examples/configs/llama/3_2_1b/fp16.toml /tmp/llama_fp16.toml
# 修改 input.prompt，并在 [run] 中添加 output_dir。
vi /tmp/llama_fp16.toml
python examples/run.py --config /tmp/llama_fp16.toml --check-config
python examples/run.py --config /tmp/llama_fp16.toml
```

<a id="输入与执行参数"></a>

### 常用参数

| 字段 | 模板值 | 省略时 | 范围与关联 |
|---|---|---|---|
| `input.prompt` | `Explain what a compiler does.` | 未提供 `prompt_token_ids` 时必填 | 用 checkpoint tokenizer 分词 |
| `input.decode_steps` | 4 | 4 | 非负整数；prefill 先产生一个 token，再执行最多指定次数的 decode |
| `input.stop_on_eos` | 未配置 | `true` | 遇 EOS 提前结束 |
| `rpu_execution.prefill.chunk_size` | `"auto"` | `"auto"` | 选择 prefill 分块，显式值由执行规划器检查 |

模板的 `run.warmup/runs/seed` 为 2 / 3 / 0，`cpu.threads` 未配置。通用字段的默认值、输出目录、计时与诊断选项见[公共参数说明](../PARAMETERS.md)。

## 输出

输出生成 token 与运行摘要，产物保存在 `run.output_dir`。
