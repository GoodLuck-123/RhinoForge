# DINOv3 配置
[配置索引](../CONFIGURATIONS.md) · [使用](#使用)

入口：`python examples/run.py --config <file>`；推荐模板 [vit_b/fp16.toml](vit_b/fp16.toml)。

<a id="weights"></a>
<a id="weight-sources"></a>
## 权重来源与存储类型

| 权重 / 用途 | 磁盘存储精度 | 发布者与下载来源 | 下载到的目录（相对于 `$RPU_MODEL_CACHE`） |
|---|---|---|---|
| 官方 DINOv3 ViT-B/16 LVD-1689M，当前模板的标准输入 | **FP32**，已核对官方 HF safetensors 元数据 | [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) · [精度元数据](https://huggingface.co/api/models/facebook/dinov3-vitb16-pretrain-lvd1689m) | `dinov3-vitb16-pretrain-lvd1689m` |

官方 safetensors 元数据为 **F32（FP32）**，不要因模板名 `fp16.toml` 而寻找“官方 FP16 包”。需先在官方页面取得访问权限；下载同一 snapshot 的 `preprocessor_config.json`，不能替换成 ViT-S/L 或其他预处理配置。

<a id="weight-recipes"></a>
## 模块精度与配方

| config | 运行时模块权重 / 激活 | 输入来源 | 获取方式、源码与完整命令 |
|---|---|---|---|
| [vit_b/fp16.toml](vit_b/fp16.toml) | patch embedding、位置/token 参数、Attention/MLP、norm 权重加载为 FP16；视觉主路径激活 FP16；末 LayerNorm/CLS 提取沿现有 CPU 路径 | 上表原始 FP32 完整 checkpoint | **不涉及低比特量化**；[examples/runners/dinov3/runner.py](../../runners/dinov3/runner.py) 的 `DINOv3ViTModel.from_pretrained(dtype=torch.float16)`；[下载并运行](#quantization) |

仅提供浮点配置，加载时转换 dtype。

<a id="quantization"></a>
## 下载与运行

**本族现有模板无需离线量化，也没有W8/W4转换步骤。** 按[权重来源](#weights)下载完整原浮点权重到 `$RPU_MODEL_CACHE/dinov3-vitb16-pretrain-lvd1689m`；在仓库根目录、已安装项目依赖的 Python 环境中运行。运行前先把该模板的 `input.images` 改为真实图片路径。

```bash
export RPU_MODEL_CACHE=/absolute/path/to/model_cache  # 替换为真实缓存根目录
hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local-dir "$RPU_MODEL_CACHE/dinov3-vitb16-pretrain-lvd1689m"
cp examples/configs/dinov3/vit_b/fp16.toml /tmp/dinov3_fp16.toml
# 编辑 /tmp/dinov3_fp16.toml，把 input.images 改为真实图片路径
python examples/run.py --config /tmp/dinov3_fp16.toml
```

模板的 `example.registry_alias` 自动读取上述缓存目录。加载入口为 [examples/runners/dinov3/runner.py](../../runners/dinov3/runner.py)；dtype 转换由加载器完成，无需另存量化目录。

## 支持范围

DINOv3 ViT-B FP16 视觉编码器，登记状态 `SUPPORTED`；输出视觉特征，不生成文本。没有公开 W8/W4 或 ViT-S/L 配置。

## 使用

在 `RPU_MODEL_CACHE/dinov3-vitb16-pretrain-lvd1689m` 准备原始权重，复制模板并填写 `input.images` 和 `run.output_dir`：

```bash
python examples/run.py --config /path/to/my_dinov3.toml --check-config
python examples/run.py --config /path/to/my_dinov3.toml
```

结果保存输出 tensor 快照；图片由 checkpoint processor 预处理。

采样设置示例：`cpu.threads=8`、`run.warmup=2`、`run.runs=5`；模板默认3次正式运行。

## 可修改参数

| 字段 | 默认 / 条件 |
|---|---|
| `input.images` | 至少一张真实图片；尺寸须符合原 profile |
| `rpu_execution.vision.chunk_size` | auto；显式值须符合运行时容量 |
| `cpu.threads` | 可选正整数；省略继承环境 |
| `run.warmup/runs/seed` | 2 / 3 / 0 |

支持边界见[模型矩阵](../../../docs/model_support.md)，运行库准备见[算子库规则](../../../docs/runtime_assets.md)。
