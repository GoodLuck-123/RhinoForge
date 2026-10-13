# InternVLA-N1

[配置索引](../CONFIGURATIONS.md) · [W8 System2 模板](navdp/w8a16_system2.toml)

System2 Vision 编码图像，decoder 生成 latent，经 CPU goal glue 交给 RGBD 编码器和 NavDP 生成导航动作。System2 decoder 的 Q/K/V/O、Gate/Up/Down 使用逐输出通道 W8A16；其余设备组件为 FP16，CPU embedding 和 goal glue 为 FP32。

<a id="weights"></a>
<a id="weight-sources"></a>
<a id="weight-recipes"></a>
<a id="quantization"></a>
## 准备配对资产

父权重为 [InternRobotics/InternVLA-N1-w-NavDP](https://huggingface.co/InternRobotics/InternVLA-N1-w-NavDP) 的 BF16 版本 `6d48df24da45e8b31fdcef9572bbcbf68c0baf64`；配套 [InternNav](https://github.com/InternRobotics/InternNav) 固定到 `7a5c62400ac45b313d9b709c740b64191556a242`。

在安装 InternNav CPU 依赖、`transformers==4.51.0` 的环境中执行：

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
HF_CACHE=/path/to/hf-cache
INTERNNAV_SRC=/path/to/InternNav
git clone https://github.com/InternRobotics/InternNav.git "$INTERNNAV_SRC"
git -C "$INTERNNAV_SRC" checkout 7a5c62400ac45b313d9b709c740b64191556a242
python scripts/prepare_internvla_assets.py \
  --root "$RPU_MODEL_CACHE" --hf-cache "$HF_CACHE" --internnav-root "$INTERNNAV_SRC"
```

工具下载固定权重并生成 `internvla_n1-controlled/`，包括 `s2_w8a16/`、`s2_vit/`、`s2_frontend.safetensors`、`navdp_head.safetensors`、`rgbd_src/`、`full_chain_fixture.pt` 和 `asset_manifest.json`。`--prepare-only` 仅准备权重。`--modelscope-weights` 从官方 ModelScope 镜像获取权重，并按固定 Hugging Face 版本的文件校验和核对。

默认输入为固定种子的图像、RGBD 和噪声；System2 使用真实模型生成 token 和位置。`--rgb-input` 只替换 System2 图像，RGBD 仍为合成输入；`--instruction` 修改导航指令。fixture 保存 `pixel_values`、`grid_thw`、`output_ids`、`image_idx`、`position_ids`、`rgb`、`depth` 和 `init_noise`。

## 使用

```bash
cp examples/configs/internvla/navdp/w8a16_system2.toml /tmp/internvla.toml
# 填入 input.fixture、input.asset_manifest 和 run.output_dir。
vi /tmp/internvla.toml
python examples/run.py --config /tmp/internvla.toml --check-config
python examples/run.py --config /tmp/internvla.toml
```

fixture 与 manifest 必须来自同一次资产准备；入口核对权重身份、词表、位置和组件形状。Vision grid 为 `[[1,28,28]]`，对应 784 patches 和 196 image tokens。模板为 System2 prefill 启用最多 64 行自动补齐，逻辑输入和输出长度不变。

System2 逻辑长度为 1904 时，自动 chunk 使用 176；省略 padding 设置时补到 1936。其他长度默认由共享 planner 在 64 行预算内选择补齐。显式整数 `chunk_size`、`padding_rows` 和自动补齐预算仍按配置传入 planner。

| 字段 | 设置 |
|---|---|
| `input.sample_num/seed` | `sample_num=4`，seed 默认 1 |
| `cpu.threads` | 可选正整数，省略时继承环境 |
| `run.warmup/runs/seed` | 默认 2 / 3 / 0 |

运行环境见[示例指南](../../README.zh.md)，运行库见[算子库说明](../../../docs/runtime_assets.md)。
