# Legacy 27B W8A16 配置
[Qwen3.5 配置](../../README.md)

已供 Qwen3.8-27B 混合版本权重的 W8主干/FP16 GDN 受控入口。差别是模型核数和生成上限，设备 CPU 线程均为8。

| 配置 | 模型核数 | decode上限 / 固定长度输出 |
|---|---:|---|
| [w8a16.toml](w8a16.toml) | 8 | 4 / 5 |
| [w8a16_tp4.toml](w8a16_tp4.toml) | 4 | 31 / 32 |
| [w8a16_tp6.toml](w8a16_tp6.toml) | 6 | 31 / 32 |

## 使用

准备匹配的 `qwen3_8-27b-w8a16` 缓存权重，即 `$RPU_MODEL_CACHE/Qwen3.8-27B-w8a16/`。不要把普通FP16或其他模型的W8权重放入此别名。

```bash
python examples/run.py --config examples/configs/qwen3_5/legacy/27b/w8a16.toml --check-config
python examples/run.py --config examples/configs/qwen3_5/legacy/27b/w8a16.toml
```

可在配置副本修改 `input.prompt`、`input.decode_steps`、`run.output_dir`、`run.warmup/runs` 和 `cpu.threads`。原模板为W2/R3；可设 `cpu.threads=8`、`run.warmup=2`、`run.runs=5`，保留输入和模板核数。核数、累加精度和长度按对应模板使用。模板已声明受控 opt-in；`--check-config` 检查配置字段与准入。

## 权重与转换

原已供 exact legacy27 BF16 父通过历史混合配方转换：主干 W8/A16，GDN、Vision、embedding、head、norm、conv 保留浮点。使用原始配置和配对权重，不能换成官方 Qwen3.5-27B。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python python/rpu_backend/quant/convert_qwen3_5.py \
  --src /path/to/original-legacy27-bf16 \
  --dst "$RPU_MODEL_CACHE/Qwen3.8-27B-w8a16" \
  --profile w8_main_fp16_gdn_legacy
```

转换后选择本目录的 W8 模板；完整说明见[Qwen3.5 权重与转换](../../README.md#quantization)。
