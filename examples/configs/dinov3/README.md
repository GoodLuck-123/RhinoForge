# DINOv3 配置

[配置索引](../CONFIGURATIONS.md) · [准备与运行](#使用)

本目录提供 DINOv3 ViT-B/16 的 FP16 视觉编码器配置。输入一张图片，输出全部视觉特征和 CLS 池化特征。

## 配置说明

<a id="weight-recipes"></a>
<a id="配置与权重对应关系"></a>

| 配置 | 计算范围 | 权重获取 |
|---|---|---|
| [vit_b/fp16.toml](vit_b/fp16.toml) | ViT-B/16 视觉编码，FP16 | [官方 FP32 权重](#weights)，加载时转 FP16 |

<a id="fp16-基准做了什么"></a>

### 精度与执行

| 阶段 | 精度与执行方式 |
|---|---|
| 图片与 patch embedding | CPU 预处理；将 patch Conv2d 折叠为 CPU FP16 Linear |
| Vision encoder | FP16 权重与激活；12 个 Attention head 补零至 16 个，适配 TP8；12 层在一个保留的 Graph 中执行并重放 |
| 输出 | 公共入口在 CPU 完成末 LayerNorm，保存 `last_hidden_state`，并取第一个 token 作为 `pooler_output` |

每张 224×224 图片产生 196 个 patch，加上 CLS 和 4 个 register token，共 201 个有效 token。执行时按 16 行对齐至 208，补齐不增加有效特征。模板的 `rpu_execution.vision.chunk_size="auto"` 选择这个单块容量。

加载和执行分别见 [runner](../../runners/dinov3/runner.py) 与 [DINOv3 adapter](../../../python/rpu_backend/adapters/dinov3.py)。

## 准备与运行

<a id="weight-sources"></a>
<a id="weights"></a>
<a id="quantization"></a>
<a id="权重准备"></a>

### 获取权重

`example.registry_alias="dinov3-vit-b"` 固定读取 `$RPU_MODEL_CACHE/dinov3-vitb16-pretrain-lvd1689m`。权重、`config.json` 和 `preprocessor_config.json` 使用同一下载版本。

源权重为 [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)，官方 safetensors 存储为 FP32。先在模型页面取得访问权限，再下载完整目录。在已安装项目依赖的环境、仓库根目录执行：

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local-dir "$RPU_MODEL_CACHE/dinov3-vitb16-pretrain-lvd1689m"
```

FP16 转换在加载时完成，仍使用上面的原始目录。

<a id="使用"></a>

### 输入与运行

复制模板，填写一张真实图片和输出目录，然后在 设备环境运行：

```bash
cp examples/configs/dinov3/vit_b/fp16.toml /tmp/dinov3_fp16.toml
# 修改 input.images，并在 [run] 中添加 output_dir。
vi /tmp/dinov3_fp16.toml
python examples/run.py --config /tmp/dinov3_fp16.toml --check-config
python examples/run.py --config /tmp/dinov3_fp16.toml
```

<a id="输入与执行参数"></a>

### 常用参数

| 字段 | 模板值 | 省略时 | 范围与关联 |
|---|---|---|---|
| `input.images` | 一张图片的占位路径 | 必填 | 一张真实图片；processor 生成 B1、224×224 输入 |
| `rpu_execution.vision.chunk_size` | `"auto"` | `"auto"` | 按 201 个有效 token 选择 208 行单块容量；显式值填 208 |

模板的 `run.warmup/runs/seed` 为 2 / 3 / 0，`cpu.threads` 未配置。通用字段的默认值、输出目录、计时与诊断选项见[公共参数说明](../PARAMETERS.md)。

## 输出

输出图像特征与运行摘要，产物保存在 `run.output_dir`。
