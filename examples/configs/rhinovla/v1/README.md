# RhinoVLA v1

[官方 FP16](official_fp16.toml) · [caller-factory 配置](fp16.toml) · [v3](../v3/README.md)

官方入口使用两图、P171、H30 和十步非 RTC 去噪；动作完整维度为 72，有效维度为 64。权重在设备上以 FP16 运行。

<a id="weights"></a>
<a id="weight-sources"></a>
<a id="weight-recipes"></a>
<a id="quantization"></a>
## 准备权重与输入

| 资产 | 公开来源与固定版本 |
|---|---|
| 源码 | [HUIXI-AI/RhinoVLA](https://github.com/HUIXI-AI/RhinoVLA)，`a3edf0c1addf4bf849192642bff3ad15559db3f4` |
| checkpoint | [HuixiAI/RhinoVLA](https://huggingface.co/HuixiAI/RhinoVLA)，revision `7f3708650e302f395e7c63607b48b56fb96b8076` |
| 输入图像 | [lerobot/libero_10_image](https://huggingface.co/datasets/lerobot/libero_10_image)，revision `7e324b526699f444044952c82ce3f438e8d300d0`，episode/frame/task 均为 0 |

在安装模型源码所需 CPU 依赖和 `pyarrow` 的环境中执行。准备工具核对源码提交、干净状态、checkpoint 与 parquet 的 SHA-256，使用官方 processor 生成输入。

```bash
export RPU_MODEL_CACHE=/path/to/model-cache
RHINO_ROOT="$RPU_MODEL_CACHE/rhinovla-controlled"
git clone https://github.com/HUIXI-AI/RhinoVLA.git "$RHINO_ROOT/source"
git -C "$RHINO_ROOT/source" checkout a3edf0c1addf4bf849192642bff3ad15559db3f4
hf download HuixiAI/RhinoVLA rhinovla_pretrain.ckpt \
  --revision 7f3708650e302f395e7c63607b48b56fb96b8076 --local-dir "$RHINO_ROOT/weights"
hf download lerobot/libero_10_image data/chunk-000/file-000.parquet \
  --repo-type dataset --revision 7e324b526699f444044952c82ce3f438e8d300d0 \
  --local-dir "$RHINO_ROOT/libero"
python scripts/prepare_rhinovla_assets.py \
  --source "$RHINO_ROOT/source" \
  --checkpoint "$RHINO_ROOT/weights/rhinovla_pretrain.ckpt" \
  --libero-parquet "$RHINO_ROOT/libero/data/chunk-000/file-000.parquet" \
  --output-dir "$RHINO_ROOT/fixture"
```

输出目录须不存在。生成的 `cpu_fixture.pt` 包含两图的 `qwen_inputs`、固定种子的 state/noise、state/action mask 及来源信息；PNG 保留原数据字节，`manifest.json` 记录文件校验和。工具只准备输入，不加载模型权重。`--inputs-only` 仅导出 PNG 和清单，不生成运行所需的 fixture。

## 使用

```bash
python examples/run.py --config examples/configs/rhinovla/v1/official_fp16.toml --check-config
python examples/run.py --config examples/configs/rhinovla/v1/official_fp16.toml
```

`rhinovla-controlled` 从 `$RPU_MODEL_CACHE` 查找 `source/`、`weights/rhinovla_pretrain.ckpt` 和 `fixture/cpu_fixture.pt`。模板启用 `trust_model_code=true`，`input.num_steps=10`；可调整 `cpu.threads`、`run.warmup/runs/output_dir`。

## caller-factory 入口

`fp16.toml` 接受自备 checkpoint、请求文件、`input.runtime_factory` 和 `input.factory_config`。切换官方入口时使用上面的固定资产，迁移步骤见[升级指南](../../../../docs/upgrading_1_2.zh.md)。v3 使用另一套 96D 模型和输入。
