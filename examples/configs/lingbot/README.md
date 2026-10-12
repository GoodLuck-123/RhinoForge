# LingBot 配置

[当前 v2 配置](v2/README.md) · [配置索引](../CONFIGURATIONS.md)

| 入口 | 配置与范围 |
|---|---|
| [v2 runtime Graph](v2/README.md) | FP16/W8A16；2/3 相机；T16/32/64/128/160，另有三相机 T3 |
| [v2 exact-Z2 归档](legacy/exact_z2/README.md) | W8A16；3 相机 256×256；REAL215–225、P225；公开默认对齐掩码 |
| [v1](v1/4b_mixed.toml) | 单图输入；CPU FP32 Vision + 设备 FP16 文本/动作 |

v2 当前入口直接调用 `build_lingbot_vla_v2` / `get_action`，保留 Vision、Prefill、Action Graph，使用稀疏 Top4 路由。exact-Z2 使用旧 facade 和固定布局。两条路径均保留 32 个路由专家、共享专家、36 层 Action 和十步去噪；训练配置、输入及布局分别绑定。

<a id="weights"></a>
<a id="weight-sources"></a>
## 官方来源与原始存储精度

| 模型 / 资产 | 官方下载 / 源码 | 原始存储精度与目录 |
|---|---|---|
| LingBot v2 6B | [robbyant/lingbot-vla-v2-6b](https://huggingface.co/robbyant/lingbot-vla-v2-6b)；[官方源码](https://github.com/robbyant/lingbot-vla-v2) | [HF 元数据](https://huggingface.co/api/models/robbyant/lingbot-vla-v2-6b)中的主要模型 safetensors 为 **F32**，固定版本 `11c703bf6a5c1f45b3b69168482da11fdbba53d7`；当前入口读 `$RPU_MODEL_CACHE/lingbot-vla-v2-6b` |
| LingBot v1 4B | [robbyant/lingbot-vla-4b](https://huggingface.co/robbyant/lingbot-vla-4b)；[官方源码](https://github.com/robbyant/lingbot-vla) | [HF 元数据](https://huggingface.co/api/models/robbyant/lingbot-vla-4b)为 **F32**，本次核对版本 `87712223fb463f0f6a6565f8feba64c35418e393`；registry 读 `$RPU_MODEL_CACHE/lingbot-vla-4b` |
| exact-Z2 历史父资产 | 与 v2 同一模型系列；[归档来源](legacy/exact_z2/README.md#weight-sources) | 原始 F32 和历史 BF16 衍生资产分开记录；BF16 交付不是官方原始 F32 下载，不能互换参考 |

下载原始权重后，按对应配置准备 observation；v2 另需配对训练 YAML。

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
hf download robbyant/lingbot-vla-v2-6b --revision 11c703bf6a5c1f45b3b69168482da11fdbba53d7 --local-dir "$RPU_MODEL_CACHE/lingbot-vla-v2-6b"
hf download robbyant/lingbot-vla-4b --revision 87712223fb463f0f6a6565f8feba64c35418e393 --local-dir "$RPU_MODEL_CACHE/lingbot-vla-4b"
```

<a id="weight-recipes"></a>
## 模块精度与配方

W/A 是权重/激活精度，不能用于反推官方下载文件的存储精度。同一配方的相机数和文本长度变体合并列出。

| 配置组 | 模块实际 W/A 精度 | 输入父权重 → 转换方式与命令 |
|---|---|---|
| `v2/{2,3}cam_t*/fp16.toml`，含 `3cam_t3` | Vision、Text、Action 投影 W16/A16；CPU 输入处理与输出快照另计 | 原始 F32 → 加载时转 FP16，无量化；[FP16 完整命令](v2/README.md#fp16-loading) |
| `v2/{2,3}cam_t*/w8a16.toml`，含 `3cam_t3` | Vision block Q/K/V/O/FC1/FC2、Text 七类投影、Action Q/K/V/O 与共享/路由专家 Gate/Up/Down 为 W8/A16；Vision patch/merger、Action router/IO/条件与归一化参数保留浮点，设备权重为 FP16 | 同一原始 F32 → 加载时逐输出通道 INT8 + FP16 scale；[完整命令](#quantization)。无离线导出 CLI |
| `legacy/exact_z2/6b_w8a16.toml` | 上述 W8 投影范围；grouped-expert/exact-Z2 安装，使用公开默认对齐掩码 | 配对的 F32 或 BF16 父资产 → [旧加载命令](legacy/exact_z2/README.md#quantization)；不导出量化目录 |
| `v1/4b_mixed.toml` | Vision 在 CPU 为 W32/A32；Text/Action 在 设备 为 W16/A16 | 官方 F32 → 浮点类型转换，无低比特量化；[单图配置](v1/4b_mixed.toml) |

当前 v2 的 Text 使用 `_quantize_text_w8a16()`，Action 使用 `build_expert_moe()` / `pack_sparse_expert_projection()`，Vision 使用共享安装器；[源码与完整加载命令](#quantization)在下方。W8 不量化路由决策矩阵，Top4 语义不因精度名改变。

<a id="quantization"></a>
## 加载时量化

以下命令在仓库根目录执行。量化在最后一行加载模型时发生，不生成独立量化目录。先替换实际路径，再按注释用 `vi` 编辑副本并保存；`--check-config` 只校验配置，不能代替最后的运行命令。

### 当前 v2 W8：三相机公开 T3

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
# 上述缓存根下放完整原始 F32 目录 lingbot-vla-v2-6b。
export LINGBOT2_TRAINING_CONFIG=/path/to/matching-training.yaml
export LINGBOT2_PUBLIC_CASE=/path/to/frozen-3cam-t3-case
cp examples/configs/lingbot/v2/3cam_t3/w8a16.toml /tmp/lingbot2_w8.toml
# run.output_dir = 本次独立输出目录。input 的三个变量由上面 export 解析。
vi /tmp/lingbot2_w8.toml
python examples/run.py --config /tmp/lingbot2_w8.toml
```

<a id="当前-v2-w8两相机-t16-observation"></a>
### 当前 v2 W8：两相机 T16 观测输入

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
export LINGBOT2_TRAINING_CONFIG=/path/to/matching-training.yaml
export LINGBOT2_2CAM_T16_OBSERVATION=/path/to/real-2cam-t16-observation.pt
cp examples/configs/lingbot/v2/2cam_t16/w8a16.toml /tmp/lingbot2_2cam_t16_w8.toml
# run.output_dir = 本次独立输出目录。保留 cameras=2 和 expected_language_tokens=16。
vi /tmp/lingbot2_2cam_t16_w8.toml
python examples/run.py --config /tmp/lingbot2_2cam_t16_w8.toml
```

其他相机/T 模板使用匹配的输入，并导出 `input.observation` 指定的变量，如 `LINGBOT2_3CAM_T160_OBSERVATION`。**v2 FP16 无需量化**，同目录 `fp16.toml` 使用原始父权重。v1 单图混合浮点无需量化；exact-Z2 使用独立的 grouped-expert 安装。

**实现位置**：[examples/runners/lingbot2/optimized.py](../../runners/lingbot2/optimized.py)；[python/rpu_backend/adapters/lingbot_vla_v2/runtime.py](../../../python/rpu_backend/adapters/lingbot_vla_v2/runtime.py)；[python/rpu_backend/adapters/lingbot_vla_v2/convert.py](../../../python/rpu_backend/adapters/lingbot_vla_v2/convert.py)；[python/rpu_backend/adapters/qwen3_vl/vision.py](../../../python/rpu_backend/adapters/qwen3_vl/vision.py)。`run_example_config()` → `build_lingbot_vla_v2()`；Text 用 `_quantize_text_w8a16()`，Action 用 `build_expert_moe()` / `pack_sparse_expert_projection()`，Vision 经共享安装器。runner 从 TOML 选择冷量化参数，不需手设额外量化环境变量，也没有离线转换 CLI。

## v1 参数

`input.observation` 指向单图 CPU observation 字典，包含 `images`、`img_masks`、`lang_tokens`、`lang_masks`、`state`、`noise`。模型空间图像为 `[1,256,1176]`，状态为 `[1,75]`，噪声为 `[1,50,75]`；语言 token 与掩码长度一致。`input.num_steps/max_seq` 默认 `10/256`，`run.warmup/runs/seed` 默认 `2/3/0`。`cpu.threads` 为正整数，省略时继承环境。

[运行入口](../../README.zh.md) · [运行库](../../../docs/runtime_assets.md)
