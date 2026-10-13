# InternVLA-N1 配置

[配置索引](../CONFIGURATIONS.md) · [准备与运行](#使用)

本目录运行 System2 Vision/语言主干、RGBD 编码器与 NavDP 的完整计算链，输入为配对的权重包、CPU fixture 和资产清单。当前提供一种混合精度配方。

## 配置说明

<a id="weight-recipes"></a>
<a id="配置与权重对应关系"></a>

| 配置 | 计算范围 | 权重获取 |
|---|---|---|
| [navdp/w8a16_system2.toml](navdp/w8a16_system2.toml) | System2 → RGBD → NavDP 完整链，混合精度 | [官方父权重与配套源码](#weights)，经[准备脚本](#quantization)生成运行资产 |

System2 decoder 使用 W8A16，其余模块按下面的精度执行：

| 模块 | 实际精度 |
|---|---|
| System2 decoder | Q/K/V/O、Gate/Up/Down 为逐输出通道 INT8 权重、FP16 scale 与激活；norm 为 FP16 |
| System2 Vision | 设备 FP16 权重与激活 |
| RGBD / NavDP | 设备投影为 FP16 |
| CPU 前端与 goal glue | embedding、latent 拼接和目标特征计算为 FP32 |

<a id="基准计算链"></a>

### 执行方式

1. 设备 System2 Vision 编码图片；CPU 拼接词 embedding、图像特征和 4 个 latent query。
2. 设备 System2 decoder 使用 W8A16 得到 latent 表示；CPU goal glue 将其变为导航目标特征。
3. 设备 RGBD 编码器处理 RGB 与深度输入，NavDP 使用目标、RGBD 特征和固定初始噪声输出动作。

入口统一管理六个组件的执行计划与 Graph，另为瞬时 patch 操作准备一个队列区域。`[rpu_execution]` 默认留空，各组件按默认 AUTO 设置规划。模板包含所需的 `RPU_INTERNVLA_N1_ALLOW_NUMERIC_BLOCKED="1"` 启用项。

实现见 [runner](../../runners/internvla/runner.py) 和[组件执行管理](../../../python/rpu_backend/adapters/internvla_n1/execution.py)。

## 准备与运行

<a id="weight-sources"></a>
<a id="weights"></a>
<a id="源权重与配套源码"></a>

### 获取源权重与配套源码

| 资产 | 来源 | 固定版本 |
|---|---|---|
| BF16 父权重 | [InternRobotics/InternVLA-N1-w-NavDP](https://huggingface.co/InternRobotics/InternVLA-N1-w-NavDP) | `6d48df24da45e8b31fdcef9572bbcbf68c0baf64` |
| 模型源码 | [InternRobotics/InternNav](https://github.com/InternRobotics/InternNav) | 干净提交 `7a5c62400ac45b313d9b709c740b64191556a242` |

<a id="quantization"></a>
<a id="转换与输入准备"></a>

### 转换并生成输入

在仓库根目录执行。CPU 准备环境使用 `transformers==4.51.0`、`diffusers==0.33.1` 和 InternNav 的 CPU 依赖；准备过程不加载设备。

```bash
export RPU_MODEL_CACHE=/path/to/new-model-cache
HF_CACHE=/path/to/hf-cache
INTERNNAV_SRC=/path/to/InternNav

git clone https://github.com/InternRobotics/InternNav.git "$INTERNNAV_SRC"
git -C "$INTERNNAV_SRC" checkout 7a5c62400ac45b313d9b709c740b64191556a242
python scripts/prepare_internvla_assets.py \
  --root "$RPU_MODEL_CACHE" \
  --hf-cache "$HF_CACHE" \
  --internnav-root "$INTERNNAV_SRC"
```

[准备脚本](../../../scripts/prepare_internvla_assets.py)下载固定 HF snapshot，提取下表中的运行资产并生成完整 fixture/manifest。`--prepare-only` 只提取资产，运行本模板需要省略该选项。脚本固定使用这一混合精度配方，没有 `--precision` 参数。

默认输入为固定种子的图像、RGBD 和噪声；System2 使用真实模型生成 token 和位置。`--rgb-input` 只替换 System2 图像，RGBD 仍为合成输入；`--instruction` 修改导航指令。`--modelscope-weights` 从官方 ModelScope 镜像获取权重，并按固定 Hugging Face 版本的文件校验和核对。已有完整配对权重包时可直接运行。

### 配置与产物绑定

`example.registry_alias="internvla-n1-controlled"` 读取 `$RPU_MODEL_CACHE/internvla_n1-controlled`。这个目录放转换后的运行资产，原始 HF snapshot 放在准备脚本的 `--hf-cache` 目录。

| 准备产物，相对运行目录 | 内容 | 字段或加载位置 |
|---|---|---|
| `s2_w8a16/` | System2 decoder 量化权重与配置 | runner 自动读取 |
| `s2_vit/` | System2 Vision 权重与配置 | runner 自动读取 |
| `s2_frontend.safetensors` | embedding 与 latent queries | runner 自动读取 |
| `navdp_head.safetensors` | NavDP/RGBD 与目标特征连接层权重 | runner 自动读取 |
| `rgbd_src/` | 配对的纯 Torch DINOv2 源码 | runner 自动读取 |
| `full_chain_fixture.pt` | 完整链输入和噪声 | `input.fixture` |
| `asset_manifest.json` | 资产路径与 SHA256 | `input.asset_manifest` |

<a id="使用"></a>

### 输入与运行

切换到已安装项目依赖的设备环境，复制模板并绑定准备产物：

```bash
cp examples/configs/internvla/navdp/w8a16_system2.toml /tmp/internvla.toml
# input.fixture = /path/to/new-model-cache/internvla_n1-controlled/full_chain_fixture.pt
# input.asset_manifest = /path/to/new-model-cache/internvla_n1-controlled/asset_manifest.json
# 在 [run] 中添加 output_dir。
vi /tmp/internvla.toml
python examples/run.py --config /tmp/internvla.toml --check-config
python examples/run.py --config /tmp/internvla.toml
```

`RPU_MODEL_CACHE` 继续指向准备步骤的根目录。runner 核对资产清单与输入形状。

fixture 必须包含 `pixel_values`、`grid_thw`、`output_ids`、`image_idx`、`position_ids`、`rgb`、`depth`、`init_noise`。System2 Vision grid 为 `[[1,28,28]]`，对应 784 个 patch、196 个图像 token。输入长度、词表与位置字段由入口一起检查。

<a id="输入与执行参数"></a>

### 常用参数

| 字段 | 模板值 | 省略时 | 范围与关联 |
|---|---|---|---|
| `input.fixture` | 文件占位路径 | 必填 | 完整链 CPU 输入，与权重包配套 |
| `input.asset_manifest` | 文件占位路径 | 必填 | 同一权重包的资产清单 |
| `input.sample_num` | 4 | 4 | 必须与 `init_noise` 的第一维 4 一致 |
| `input.seed` | 1 | 1 | 导航采样种子；fixture 提供显式初始噪声 |

模板的 `run.warmup/runs/seed` 为 2 / 3 / 0，`cpu.threads` 未配置。公共默认值、输出和诊断参数见[公共参数说明](../PARAMETERS.md)。

System2 逻辑长度为 1904 时，自动 chunk 使用 176；省略 padding 设置时补到 1936。其他长度默认由共享 planner 在 64 行预算内选择补齐。显式整数 `chunk_size`、`padding_rows` 和自动补齐预算仍按配置传入 planner。

## 输出

`run.output_dir` 保存 `report.json` 和完整动作快照。
