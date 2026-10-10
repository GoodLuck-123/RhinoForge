# GR00T-N1.7 配置
[配置索引](../CONFIGURATIONS.md) · [使用](#使用)

入口：`python examples/run.py --config <file>`；推荐 [三图224 FP16](n1_7_droid/fp16.toml)。固定 B1/TP8、H40，默认四步去噪。

<a id="weights"></a>
<a id="weight-sources"></a>
## 官方来源与原始存储精度

| 资产 | 官方下载 / 源码 | 原始存储精度与用途 |
|---|---|---|
| 策略父权重 | [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B)；[NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | 固定版本 `2fc962b973bccdd5d8ce4f67cc63b264d6886495` 的 [HF safetensors 元数据](https://huggingface.co/api/models/nvidia/GR00T-N1.7-3B)为 **BF16**；不是 FP16 或 INT8 下载 |
| 配套配置与分词器 | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | 本入口只读取配置、processor/tokenizer 文件，不用其权重替换 GR00T 中的主干或动作权重 |

先下载到 registry 对应目录；此步骤不量化。真实 observation 仍需按本页输入合同另行准备。

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
hf download nvidia/GR00T-N1.7-3B --revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495 --local-dir "$RPU_MODEL_CACHE/GR00T-N1.7-3B"
hf download Qwen/Qwen3-VL-2B-Instruct --revision 89644892e4d85e24eaac8bacfd4f463576704203 --include '*.json' '*.txt' '*.jinja' --local-dir "$RPU_MODEL_CACHE/Qwen3-VL-2B-Instruct"
```

<a id="weight-recipes"></a>
## 模块精度与配方

W/A 分别表示权重/激活精度；INT8 权重配 FP16 scale，ACC 累加精度另行选择。224/256 配置使用同一父权重与配方。

| 配置组 | 模块实际 W/A 精度 | 输入父权重 → 转换方式与命令 |
|---|---|---|
| `n1_7_droid/fp16{,_256}.toml` | Vision、Text、VL encoder、DiT 和动作连接层为 W16/A16；CPU 预处理/结果快照不据此改成 FP16 | 官方 BF16 父权重 → 加载时转 FP16；无量化 CLI，使用下方命令并选择 FP16 模板 |
| `n1_7_droid/w8a16{,_256}.toml` | Text 的 Q/K/V/O、Gate/Up/Down；Vision block 的 Q/K/V/O、FC1/FC2；VL encoder 的六类投影；DiT 的 Q/K/V/O、FF1/FF2 和 AdaLN 条件投影均 W8/A16。动作连接层 `se2/ae2a/ae2t/ae3/dec1/dec2/po1/po2` 为 W8/A16；`se1/ae1`（填充后 K=144）、Vision patch/merger、norm/bias/embedding 保持浮点，设备权重为 FP16 | 同一 BF16 父权重 → [加载时量化完整命令](#quantization)；没有独立离线导出 CLI 或量化下载目录 |

实际范围由 [runtime.py](../../../python/rpu_backend/adapters/gr00t/runtime.py) 的 `_quantize_backbone_inplace()`、`_build_vl_encoder()`、`_build_denoise()` 和共享 Vision 安装器决定，不能将 `w8a16` 简写理解为全部张量均 INT8。

<a id="quantization"></a>
## 加载时量化

以下命令在仓库根目录执行。量化在最后一行加载模型时发生，不生成独立量化目录。先替换实际路径，再按注释用 `vi` 编辑副本并保存；`--check-config` 只校验配置，不能代替最后的运行命令。

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
# 缓存根下必须有 GR00T-N1.7-3B 原始策略权重，以及 Qwen3-VL-2B-Instruct 配套配置/tokenizer。
cp examples/configs/gr00t/n1_7_droid/w8a16.toml /tmp/gr00t_w8.toml
# input.observation = 真实三图 224/T32 CPU observation 文件。
# run.output_dir = 本次输出目录；保留原 profile_id、registry_alias 和 dtype。
vi /tmp/gr00t_w8.toml
python examples/run.py --config /tmp/gr00t_w8.toml
```
以上为 W8A16 224 配方。256 配方复制 `n1_7_droid/w8a16_256.toml` 并改用真实三图 256 observation；二者从同一原始 GR00T 目录加载。**FP16 无需量化**，不需要转换脚本或新输出权重目录。

**实现位置**：[examples/runners/gr00t/runner.py](../../runners/gr00t/runner.py)；[python/rpu_backend/adapters/gr00t/runtime.py](../../../python/rpu_backend/adapters/gr00t/runtime.py)；[python/rpu_backend/adapters/qwen3_vl/vision.py](../../../python/rpu_backend/adapters/qwen3_vl/vision.py)。`_gr00t()` → `build_gr00t_vla()`；Text 为 `_quantize_backbone_inplace()`，Vision 为 `_convert_vision_block_weights_for_rpu()`，其余组件由 `_build_vl_encoder()` / `_build_denoise()` 安装；Vision merger 保持 FP16。无需直接运行这些实现文件。

## 选择配置

| 三图尺寸 | FP16 | W8A16 | 区别 |
|---|---|---|---|
| 224×224 | [配置](n1_7_droid/fp16.toml) | [配置](n1_7_droid/w8a16.toml) | Qwen3-VL grid每图 `[1,14,14]` |
| 256×256 | [配置](n1_7_droid/fp16_256.toml) | [配置](n1_7_droid/w8a16_256.toml) | grid每图 `[1,16,16]` |

W8 的 Vision merger 仍为 FP16。T 是纯 instruction token 数，不含图像/模板；选择32/64/128时须更换匹配 observation，入口不补 token 或缩放图片。登记支持见 [模型矩阵](../../../docs/model_support.md)。

## 使用

准备registry资产（`RPU_MODEL_CACHE`），复制模板、填写真实observation和输出目录：

```bash
cp examples/configs/gr00t/n1_7_droid/w8a16.toml /tmp/gr00t.toml
python examples/run.py --config /tmp/gr00t.toml --check-config
python examples/run.py --config /tmp/gr00t.toml
```

输出独立子目录的 `report.json`、`output.pt`；`--profile/--hwperf` 用于诊断，`--check-config` 仅检查配置。运行需匹配 native、空闲板卡和既有设备锁；冷配置变化后使用新进程。

## 可修改参数

| 字段 | 默认 / 范围 | 条件 |
|---|---|---|
| `input.text_tokens` | 32 / 64 / 128 | 同步替换真实observation |
| `input.num_steps` | 4，可冷选10 | 构建后固定步数 |
| `input.max_prefix_len` | 自动16对齐、至少256、最多512 | 显式容量必须覆盖实际prefix |
| `cpu.threads` | 8 | 与设备核数无关 |
| `run.warmup/runs/seed` | 2 / 6 / 0 | 使用独立进程/输出目录 |

## 输入与限制

observation为CPU tensor字典或已有 `{"inputs": 字典, "meta": ...}` 包装：B1、无padding的input_ids/attention_mask，一个user消息先放三图再instruction并带assistant generation prompt；pixel_values为224 `[588,1536]` 或256 `[768,1536]`，由Qwen3-VL processor准备；state为有限浮点 `[1,1,132]`。可选mm_token_type_ids、embodiment_id必须与真实输入/配置一致，meta不能替代tensor验证。

T32/64/128 的真实 prefix 分别为224 P193/225/289、256 P238/270/334。仅使用 [正式 main/DDR 两份 REF](../../../docs/runtime_assets.md)，不接受 `expansion_ref` 配置字段。
