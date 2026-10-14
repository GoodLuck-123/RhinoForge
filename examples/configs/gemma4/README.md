# Gemma4 配置

[配置索引](../CONFIGURATIONS.md) · [准备与运行](#使用)

本目录提供 Gemma4 E4B Instruct 的文本生成配置，采用 CPU 与设备混合执行。输入为文本，当前入口范围为文本 MVP。

## 配置说明

<a id="weight-recipes"></a>
<a id="配置与权重对应关系"></a>

| 配置 | 计算范围 | 权重获取 |
|---|---|---|
| [e4b/fp16.toml](e4b/fp16.toml) | E4B Instruct 文本生成，设备 FP16 decoder + CPU 浮点前后处理 | [官方 BF16 权重](#weights)，加载时按模块转换 |

<a id="fp16-基准做了什么"></a>

### 精度与执行

| 阶段 | 精度与执行方式 |
|---|---|
| 输入 embedding 与 PLE 侧输入 | 源权重保留在 CPU；按前向逻辑转换为 FP32 参与计算，再生成 decoder 所需的 FP16 输入。PLE 是模型逐层使用的附加 embedding |
| Decoder | decoder 与图内 PLE 投影使用设备 FP16；使用 KV cache 和保留的 Graph 重放 |
| 输出 head | CPU 使用共享 embedding 权重计算 FP32 logits，并应用 softcap |

因此，`fp16` 描述设备 decoder 的精度，完整请求还包含 CPU embedding、PLE 和输出 head 的工作。加载与执行见 [runner](../../runners/gemma4/runner.py) 和 [Gemma4 adapter](../../../python/rpu_backend/adapters/gemma4/__init__.py)。

## 准备与运行

<a id="weight-sources"></a>
<a id="weights"></a>
<a id="quantization"></a>
<a id="权重准备"></a>

### 获取权重

`example.registry_alias="gemma4-e4b"` 固定读取 `$RPU_MODEL_CACHE/gemma-4-e4b-it`。HF 仓库名中的 `E4B` 大写，本地目录名中的 `e4b` 小写；下载完整 config 和 tokenizer。

源权重为 [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)，官方 safetensors 存储为 BF16。先取得模型访问权限，再在仓库根目录执行：

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
hf download google/gemma-4-E4B-it \
  --local-dir "$RPU_MODEL_CACHE/gemma-4-e4b-it"
```

转换在内存中完成，不产生单独的 FP16 目录。

<a id="使用"></a>

### 输入与运行

在已安装项目依赖的设备环境中，复制模板并修改文本和输出目录：

```bash
cp examples/configs/gemma4/e4b/fp16.toml /tmp/gemma4_fp16.toml
# 修改 input.prompt，并在 [run] 中添加 output_dir。
vi /tmp/gemma4_fp16.toml
python examples/run.py --config /tmp/gemma4_fp16.toml --check-config
python examples/run.py --config /tmp/gemma4_fp16.toml
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

模型加载时根据输入长度准备 KV cache，最小容量为 512。更改模型或启动时执行设置后，用新进程运行。

## 输出

输出生成 token 与运行摘要，产物保存在 `run.output_dir`。
