# InternVLA-N1 配置
[配置索引](../CONFIGURATIONS.md) · [使用](#使用)

入口：`python examples/run.py --config <file>`；选择下表配置。

<a id="weights"></a>
<a id="weight-sources"></a>
## 官方来源与原始存储精度

| 资产 | 官方下载 / 源码 | 原始存储精度与用途 |
|---|---|---|
| System2 与 NavDP 完整父权重 | [InternRobotics/InternVLA-N1-w-NavDP](https://huggingface.co/InternRobotics/InternVLA-N1-w-NavDP) | 固定版本 `6d48df24da45e8b31fdcef9572bbcbf68c0baf64` 的 [HF safetensors 元数据](https://huggingface.co/api/models/InternRobotics/InternVLA-N1-w-NavDP)为 **BF16**；[官方配置](https://huggingface.co/InternRobotics/InternVLA-N1-w-NavDP/blob/main/config.json)也声明 `bfloat16` |
| 配套模型源码 | [InternRobotics/InternNav](https://github.com/InternRobotics/InternNav) | 配套源码使用提交 `7a5c62400ac45b313d9b709c740b64191556a242`，不接受任意新版源码替换 |

下载固定源码和 checkpoint：

```bash
HF_CACHE=/path/to/hf-cache
INTERNNAV_SRC=/path/to/InternNav
git clone https://github.com/InternRobotics/InternNav.git "$INTERNNAV_SRC"
git -C "$INTERNNAV_SRC" checkout 7a5c62400ac45b313d9b709c740b64191556a242
hf download InternRobotics/InternVLA-N1-w-NavDP --revision 6d48df24da45e8b31fdcef9572bbcbf68c0baf64 --cache-dir "$HF_CACHE"
```

<a id="weight-recipes"></a>
## 模块精度与配方

| 配置 | 模块实际 W/A 精度 | 输入父权重 → 转换方式与命令 |
|---|---|---|
| `navdp/fp16.toml`（唯一配方） | System2 decoder 的 Q/K/V/O、Gate/Up/Down 为 **W8/A16**，逐输出通道 INT8 + FP16 scale；decoder norm 为 FP16。System2 Vision、RGBD 的 设备 投影和 NavDP 为 W16/A16；CPU embedding/latent 拼接与 goal glue 保持 FP32 浮点，不量化 | 官方 BF16 父权重 → [配对资产](#quantization)，包含 `s2_w8a16/`、`s2_vit/`、`s2_frontend.safetensors`、`navdp_head.safetensors` 与配对 fixture/manifest；随后加载运行 |

模板名 `fp16` 对应 Vision/RGBD/NavDP；System2 decoder 使用 W8。

<a id="quantization"></a>
## 准备配对资产并运行

运行使用完整的 `internvla_n1-controlled/` 包：`s2_w8a16/` 为逐通道 W8 decoder，`s2_vit/`、`s2_frontend.safetensors`、`navdp_head.safetensors` 为配套浮点组件，另需 `full_chain_fixture.pt` 和 `asset_manifest.json`。由资产提供方按上述固定源码和 checkpoint 准备。

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
cp examples/configs/internvla/navdp/fp16.toml /tmp/internvla.toml
# 填 input.fixture、input.asset_manifest 和 run.output_dir。
vi /tmp/internvla.toml
python examples/run.py --config /tmp/internvla.toml --check-config
python examples/run.py --config /tmp/internvla.toml
```

## 选择配置

| 配置 | 范围 / 精度 | 登记状态 |
|---|---|---|
| [navdp/fp16.toml](navdp/fp16.toml) | Vision/RGBD/NavDP FP16；System2 decoder固定W8A16 | CONTROLLED_EVALUATION，受控配对 |

仅支持配对frontend、NavDP权重及完整fixture/asset manifest的受控输入。

## 使用

将完整bundle放入 `$RPU_MODEL_CACHE/internvla_n1-controlled`，填写fixture与asset_manifest。完整资产需由提供方准备，包含配对模型、fixture 与 asset_manifest；不能用任意权重目录替代。

```bash
python examples/run.py --config /path/to/internvla.toml --check-config
python examples/run.py --config /path/to/internvla.toml
```

asset manifest用于核对权重与fixture身份，必须与本次bundle配对。

采样设置示例：`cpu.threads=8`、`run.warmup=2`、`run.runs=5`；模板默认3次正式运行。

## 可修改参数

| 字段 | 默认 / 条件 |
|---|---|
| `input.fixture/asset_manifest` | 受控配对资产，CPU文件 |
| `input.sample_num/seed` | sample_num固定4；seed默认1 |
| `cpu.threads` | 可选正整数；省略继承环境 |
| `run.warmup/runs/seed` | 2 / 3 / 0 |

## 输入与限制

full-chain fixture必须包含pixel_values、grid_thw、output_ids、image_idx、position_ids、rgb、depth、init_noise；Vision grid为 `[[1,28,28]]`、784 patches/196 image tokens。入口核对tensor、词表、位置和组件形状；模板保留受控opt-in。

运行环境与结果说明见[示例指南](../../README.zh.md)，运行库准备见[算子库规则](../../../docs/runtime_assets.md)。
