# LingBot 配置

[配置索引](../CONFIGURATIONS.md) · [v2 详细配置](v2/README.md)

先按模型与调用入口选择目录。LingBot v1 使用 4B 权重；v2 使用 6B 权重，其中 runtime Graph 与 exact-Z2 对应不同输入和执行方式。

## 配置说明

<a id="weight-recipes"></a>
<a id="配置与权重对应关系"></a>

| 配置 | 输入与执行范围 | 实际精度 | 权重获取 |
|---|---|---|---|
| [v2 FP16](v2/README.md) | 两/三相机，T16/32/64/128/160，另有三相机 T3 |设备 FP16 | [共用 6B FP32 父权重](v2/README.md#weights) |
| [v2 W8A16](v2/README.md#weight-recipes) | 与对应 FP16 模板配对 | Vision/Text/Action 主要投影 W8A16 | 同一份 [6B 父权重](v2/README.md#weights)，加载时量化 |
| [exact-Z2 W8A16](legacy/exact_z2/6b/w8a16.toml) | 三相机 256×256，真实前缀 215–225，物理 P225 | W8A16 | [配对的 6B FP32 或 BF16 资产](legacy/exact_z2/README.md#weights) |
| [v1 4B](v1/4b/fp16_cpu_vision.toml) | v1 observation，十步去噪 | CPU FP32 Vision +设备 FP16 Text/Action | [官方 4B FP32 权重](#weights) |

T 是有效语言 token 数。各 v2 配置的逐模块量化范围与增量说明放在对应子页。

<a id="v1-混合浮点基准"></a>

### v1 精度与执行

`v1/4b/fp16_cpu_vision.toml` 的 `example.dtype="cpu-fp32+rpu-fp16"` 选择 CPU FP32 视觉编码与设备 FP16 文本/动作计算。前端接收已经分词的 observation，加载时转换浮点类型。前缀和专家使用地址稳定的缓冲与保留的 Graph，跨请求重放。

<a id="v2-两种入口的区别"></a>

### v2 两种入口的区别

| 项目 | runtime Graph | exact-Z2 |
|---|---|---|
| 调用 | `build_lingbot_vla_v2()` / `get_action()` | `Lingbot2Policy.prepare_graphs()` / `infer()` |
| 执行 | Vision/Prefill/Action Graph，稀疏 Top4 专家 | 多视角 SPM、grouped-expert，固定 P225 |
| 精度 | FP16、W8A16 | W8A16 |
| 掩码 | 读取训练 YAML，公开 case 为 false/false | 固定 true/true |

两者均使用 36 层 Action、32 个路由专家及共享专家、Top4 和十步去噪。比较结果时选择相同父权重、输入和掩码。

## 准备与运行

<a id="weight-sources"></a>
<a id="weights"></a>
<a id="quantization"></a>
<a id="权重准备入口"></a>

### 获取 v1 权重

源权重为官方 [robbyant/lingbot-vla-4b](https://huggingface.co/robbyant/lingbot-vla-4b)，FP32，固定 revision `87712223fb463f0f6a6565f8feba64c35418e393`。源码见 [robbyant/lingbot-vla](https://github.com/robbyant/lingbot-vla)。

`example.registry_alias="lingbot-vla-4b"` 读取 `$RPU_MODEL_CACHE/lingbot-vla-4b`。在仓库根目录下载：

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
hf download robbyant/lingbot-vla-4b \
  --revision 87712223fb463f0f6a6565f8feba64c35418e393 \
  --local-dir "$RPU_MODEL_CACHE/lingbot-vla-4b"
```

此配置不需要离线量化。v2 的固定版本、目录与运行命令分别见 [runtime Graph 准备步骤](v2/README.md#weights)与 [exact-Z2 准备步骤](legacy/exact_z2/README.md#weights)。

<a id="当前-v2-w8两相机-t16-observation"></a>

两相机 T16 的 v2 observation 用法见[对应步骤](v2/README.md#当前-v2-w8两相机-t16-observation)。

<a id="使用"></a>
<a id="v1-输入与运行"></a>

### v1 输入与运行

CPU observation 字典包含 `images`、`lang_tokens`、`state`、`noise`；可选 `img_masks` 和 `lang_masks` 标识有效输入，省略时对应输入全部有效。在已安装项目依赖的设备环境中，复制模板并填写路径：

```bash
cp examples/configs/lingbot/v1/4b/fp16_cpu_vision.toml /tmp/lingbot_v1.toml
# input.observation 填配对 CPU 张量文件；在 [run] 中添加 output_dir。
vi /tmp/lingbot_v1.toml
python examples/run.py --config /tmp/lingbot_v1.toml --check-config
python examples/run.py --config /tmp/lingbot_v1.toml
```

### v1 常用参数

| 字段 | 模板值 | 省略时 | 范围与关联 |
|---|---|---|---|
| `input.observation` | 文件占位路径 | 必填 | 已分词的 CPU observation，包含显式噪声 |
| `input.num_steps` | 10 | 10 | 固定十步去噪 |
| `input.max_seq` | 256 | 256 | RoPE/KV 总容量，至少 53，并覆盖真实前缀与 51 行动作后缀；语言前缀最多 178 行 |

模板 `run.warmup/runs/seed=2/3/0`，`cpu.threads` 未配置。通用字段的省略值、日志与诊断选项见[公共参数说明](../PARAMETERS.md)。实现见 [v1 runner](../../runners/lingbot_v1/runner.py) 和 [v1 运行时](../../../python/rpu_backend/adapters/lingbot_vla/runtime.py)。

## 输出

各入口保存归一化动作和运行摘要；输入及输出结构见对应版本说明。
